#pragma once
// Speculative-verify decode attention for Intel Xe2 in ESIMD.
//
// vLLM's XPU spec-decode path runs the split-K *decode* kernel once per (sequence, query position), so the paged
// KV cache is streamed q_len times per layer (measured 2.2x the read-once floor at 15k-64k keys). Here one
// work-group owns (sequence, KV head, key split) and all q_len <= 8 query positions of all G query heads:
//   thread g = query head kvh*G + g, its 8 DPAS rows = the q_len positions (pad rows zero, discarded),
//   the G threads of a work-group read the same K/V tiles, so the KV cache comes from DRAM about once.
// Per 32-key block (never crosses a page: block_size % 32 == 0), as in fa_esimd.h:
//   S = Q K^T (2 key tiles x 16 d-chunks DPAS, B = transposed 2D load of K), online softmax in base 2,
//   O += P V (16 d-chunks x 2 key tiles DPAS, B = VNNI 2D load of V).
// Causal, bottom-right aligned: position r (0..QL-1) of a sequence with L keys sees keys [0, L-QL+r].
// Output per split: normalised O (fp16) and LSE (natural log, -inf for an empty split); FaSpecMerge combines.
// Graph-safe: the grid depends only on (batch, heads, splits); lengths and pages are read on the device.

#include <sycl/sycl.hpp>
#include <sycl/ext/intel/esimd.hpp>
#include <sycl/ext/intel/esimd/xmx/dpas.hpp>
#include <sycl/ext/intel/experimental/esimd/memory.hpp>

namespace exl3fa {

using namespace sycl::ext::intel::esimd;
namespace xesimd = sycl::ext::intel::experimental::esimd;
using fp16 = sycl::half;

// F8 = fp8 (e4m3fn) KV cache: K/V tiles are loaded as bytes (half the traffic) and converted
// to fp16 in registers (bits (b&0x80)<<8 | (b&0x7F)<<7, times 2^8: exact incl. subnormals), then the same fp16 DPAS
// math runs; per-tensor k/v scales are read on the device (ks/vs, graph-safe).
// KVM 2: vLLM int8_per_token_head cache (NHD, inline scales): per (block, slot, head) row of
// 2*(256+4) bytes = [K int8 x256 | K scale f32 | V int8 x256 | V scale f32]. kc/vc point at the block base of K/V views;
// kbs/tstride/hstride are byte strides; K/V tiles load from the block base with head/column offsets in x; per-key
// scales multiply the scores (K) and the probabilities after the row sums (V).
// in-register fast Walsh-Hadamard transform (Sylvester order, unnormalised) of 256 values
template <int L>
ESIMD_INLINE void fwht256_stage(simd<float, 256>& x) {
    auto m = x.template bit_cast_view<float, 256 / (2 * L), 2 * L>();
    simd<float, 128> lo = m.template select<256 / (2 * L), 1, L, 1>(0, 0);
    simd<float, 128> hi = m.template select<256 / (2 * L), 1, L, 1>(0, L);
    m.template select<256 / (2 * L), 1, L, 1>(0, 0) = lo + hi;
    m.template select<256 / (2 * L), 1, L, 1>(0, L) = lo - hi;
}
ESIMD_INLINE void fwht256(simd<float, 256>& x) {
    fwht256_stage<1>(x); fwht256_stage<2>(x); fwht256_stage<4>(x); fwht256_stage<8>(x);
    fwht256_stage<16>(x); fwht256_stage<32>(x); fwht256_stage<64>(x); fwht256_stage<128>(x);
}

#ifdef EXL3_SPEC_QSLM
#define EXL3_QA_DECL(a, c8) simd<uint32_t, 64> a##_u = slm_block_load<uint32_t, 64>(g * 2048 + (c8) * 256); \
    simd<int8_t, QR * 32> a = a##_u.template bit_cast_view<int8_t>()
#else
#define EXL3_QA_DECL(a, c8) simd<int8_t, QR * 32> a = Qi8.template select<QR * 32, 1>((c8) * QR * 32)
#endif
template <int KVM = 0, bool VA = false>
struct FaSpecKernelT {
    int varr_maxl = 1 << 30;
    int va_min_b = 1;          // VARR loads only for B >= this (the VA instance at B=1 runs plain loads)
    // rsign = vLLM's RHT signs [256] (device): the kernel rotates q itself
    // (q' = H (s * q) / 256, in registers) instead of the host's fp32 matmul; the merge un-rotates the output. nullptr = off.
    const float* rsign = nullptr;   // VARR only for sequences up to this many tokens (in-kernel gate)
    // q already rotated (and / 256) by the host's one-pass pre-kernel, fp32 [rows, Hq, 256];
    // the kernel reads it instead of rotating per split thread. nullptr = off.
    const float* qrot = nullptr;
    // KVM 3: vLLM int4_per_token_head cache: rows of 2*(128+4) bytes = [K nibbles x256 | K scale f32 |
    // V nibbles x256 | V scale f32]; value 2j = low nibble of byte j; the 4-bit zero point sits in the scale's low 4
    // mantissa bits. I4 runs the I8 math: nibbles (0..15) feed the int8 DPAS as is, the zero points are corrected on
    // the score side (S -= zp_k * sum(q8_r)) and on the output (O -= sum_k P8_rk * zp_k); values are in vLLM's rotated
    // space (the host rotates q and un-rotates the output).
    static constexpr bool F8 = KVM == 1, I4 = KVM == 3, I8 = KVM == 2 || KVM == 3;
    static constexpr int D = 256, QR = 8, KT = 16, KBT = 2, KB = KT * KBT, NC = D / 16;
    const fp16* q;                 // [B*QL, Hq, D] (row b*QL + r)
    const fp16* kc; const fp16* vc;  // [nblocks, BS, Hk, D] (F8: really uint8 e4m3fn; element strides are bytes)
    const int* block_table;        // [B, bt_stride]
    const int* seqused;            // [B]
    fp16* opart;                   // [B, S, QL, Hq, D]
    float* lpart;                  // [B, S, QL, Hq]
    int B, QL, Hq, Hk, G, BS, S, bt_stride;
    float scale_log2;              // softmax scale * log2(e)
    size_t kbs, vbs;               // block strides of k/v cache in elements (pages may be padded / interleaved)
    int tstride, hstride;          // token and head strides in elements (vLLM XPU interleaves K/V: 2048 / 512)
    const float* ks = nullptr; const float* vs = nullptr;   // F8: per-tensor k/v scales (device scalars)
    // varlen: qsl = query_start_loc [B+1] on the device; request b has rows qsl[b]..qsl[b+1]
    // (<= QR, else skipped: a prefill handled elsewhere); QL is then only the partial-buffer row stride (8)
    const int* qsl = nullptr;
    // HEAD-PACK (hp = 1: plain decode with QL * G <= QR, no varlen): a
    // thread's DPAS rows are the G query heads x QL positions of its KV head (row r = head kvh*G + r / QL, position
    // r % QL) instead of QL positions of one head (7 of 8 rows idle at QL = 1), and the G threads of a work-group split
    // the keys (effective split sp = s * G + g of S * G) instead of sharing them. Same per-(head, key) math.
    int hp = 0;

    // I8 helper: [32 keys][16 d] int8 (row-major, 4 dwords per key) -> fp16 [32][16] (native conversion, exact)
    static ESIMD_INLINE simd<fp16, 512> i8_tile(simd<uint32_t, 128> raw) {
        simd<int8_t, 512> by = raw.template bit_cast_view<int8_t>();
        return convert<fp16>(by);
    }
    // F8 helper: [32 keys][16 d] e4m3fn bytes (row-major, 4 dwords per key) -> fp16 [32][16]
    static ESIMD_INLINE simd<fp16, 512> f8_tile(simd<uint32_t, 128> raw) {
        simd<uint8_t, 512> by = raw.template bit_cast_view<uint8_t>();
        simd<uint16_t, 512> w = by;
        simd<uint16_t, 512> hb = ((w & 0x80) << 8) | ((w & 0x7F) << 7);
        simd<fp16, 512> f = hb.template bit_cast_view<fp16>();
        return f * fp16(256.0f);
    }

    void operator()(sycl::nd_item<1> it) const SYCL_ESIMD_KERNEL {
#ifdef EXL3_SPEC_QSLM
        // the int8 Q (2 KB per thread) lives in SLM, read per 32-d chunk during Q.K, so its
        // 32 registers are free for P.V (the int4 kernel spilled 5.8 KB inside the key-block loop)
        slm_init<8 * 2048>();
#endif
        int id = it.get_global_id(0);
        int g = id % G;
        int rest = id / G;
        int s = rest % S; rest /= S;
        int kvh = rest % Hk;
        int b = rest / Hk;
        int h = kvh * G + g;
        int L = seqused[b];
        const bool va = VA && !I4 && L <= varr_maxl && B >= va_min_b;   // per work-group, uniform: graph-safe context gate
        int q0 = b * QL, QLb = QL;
        if (qsl) { q0 = qsl[b]; QLb = qsl[b + 1] - q0; if (QLb <= 0 || QLb > QR) return; }
        const int sp = hp ? s * G + g : s, Sp = hp ? S * G : S;   // effective split (HEAD-PACK: threads split keys)
        const int nrows = hp ? G * QLb : QLb;                     // valid DPAS rows
        int nkb = (L + KB - 1) / KB;
        int per = (nkb + Sp - 1) / Sp;
        if (per < 4) per = 4;                          // at least 128 keys per split; later splits stay empty
        int kb_lo = sp * per, kb_hi = kb_lo + per;
        if (kb_hi > nkb) kb_hi = nkb;

        simd<fp16, NC * QR * 16> Qc = 0;
        // I8: Q quantized per row to int8 for an int8 x int8 Q.K^T on XMX (K tiles feed the B operand straight from
        // the transposed 2D load: no conversion, no shuffles). Layout [8 chunks of 32 d][QR rows][32].
        simd<int8_t, (I8 ? 8 * QR * 32 : 32)> Qi8 = 0;
        simd<float, QR> qs = 0.0f;
        simd<int, QR> qsum = 0;                        // I4: sum of each row's int8 q (zero-point correction)
        if constexpr (I8) {
            for (int r = 0; r < QR; ++r) {
                if (r < nrows) {
                    int rp = hp ? r % QLb : r, rh = hp ? kvh * G + r / QLb : h;
                    simd<float, D> rowf;
                    if (I4 && qrot) rowf = block_load<float, D>(qrot + ((size_t)(q0 + rp) * Hq + rh) * D);
                    else rowf = convert<float>(simd<fp16, D>(block_load<fp16, D>(q + ((size_t)(q0 + rp) * Hq + rh) * D)));
                    if constexpr (I4) {
                        if (rsign && !qrot) {   // I4-ROT (uniform per work-group)
                            rowf *= simd<float, D>(block_load<float, D>(rsign));
                            fwht256(rowf);
                            rowf *= 1.0f / 256.0f;
                        }
                    }
                    float amax = hmax<float>(abs(rowf));
                    float sc = amax > 0.0f ? amax / 127.0f : 1.0f;
                    simd<int8_t, D> q8 = convert<int8_t>(rnde<float>(rowf * (1.0f / sc)));
#ifndef EXL3_I4_U4DPAS
                    if constexpr (I4) {   // per 8 d: [d0 d2 d4 d6 | d1 d3 d5 d7] = the low / high nibble dwords of K
                        simd<int8_t, D / 2> ev = q8.template select<D / 2, 2>(0), od = q8.template select<D / 2, 2>(1);
                        simd<uint32_t, D / 4> qp;
                        qp.template select<D / 8, 2>(0) = ev.template bit_cast_view<uint32_t>();
                        qp.template select<D / 8, 2>(1) = od.template bit_cast_view<uint32_t>();
                        q8 = qp.template bit_cast_view<int8_t>();
                    }
#endif
#pragma unroll
                    for (int c8 = 0; c8 < 8; ++c8)
                        Qi8.template select<32, 1>((c8 * QR + r) * 32) = q8.template select<32, 1>(c8 * 32);
                    qs[r] = sc;
                    if constexpr (I4) qsum[r] = reduce<int>(convert<int>(q8), std::plus<>());
                }
            }
#ifdef EXL3_SPEC_QSLM
#pragma unroll
            for (int i = 0; i < 8; ++i)
                slm_block_store<uint32_t, 64>(g * 2048 + i * 256, Qi8.template bit_cast_view<uint32_t>().template select<64, 1>(i * 64));
            barrier();
#endif
        }
#if defined(EXL3_SPEC_NOQC) && !defined(EXL3_I8_FP16QK)
        if constexpr (!I8)   // the int8 / int4 paths never read the fp16 Q copy (4 KB of registers)
#endif
        for (int r = 0; r < QR; ++r) {
            if (r < nrows) {
                int rp = hp ? r % QLb : r, rh = hp ? kvh * G + r / QLb : h;
                simd<fp16, D> row = block_load<fp16, D>(q + ((size_t)(q0 + rp) * Hq + rh) * D);
#pragma unroll
                for (int c = 0; c < NC; ++c)
                    Qc.template select<16, 1>((c * QR + r) * 16) = row.template select<16, 1>(c * 16);
            }
        }
        simd<float, NC * QR * 16> O = 0.0f;
        simd<float, QR> m = -INFINITY, l = 0.0f;
#ifdef EXL3_I4_ZPFOLD
        simd<float, QR> zc = 0.0f;                     // I4 ZPFOLD: sum over blocks of pscale * sum_k P8 zp_k (rescaled)
#endif
        int off = L - QLb;                             // position r sees keys <= off + r

        constexpr int EB = (F8 || I8) ? 1 : 2;         // bytes per cache element
        // I8: the 2D surface is the whole block (rows = slots, width = all heads' rows); head/col offsets go in x
        unsigned surfW = I8 ? (unsigned)tstride - 1 : D * EB - 1, surfH = (unsigned)BS - 1, pitch = (unsigned)tstride * EB - 1;
        const int hx = I8 ? kvh * hstride / 4 : 0;     // head offset in dwords (I8)
        float sl2 = scale_log2;
        if constexpr (F8) sl2 *= ks[0];
        const int* bt = block_table + (size_t)b * bt_stride;

        // cooperative L1 prefetch of a 32-key block's K and V: 32 rows x 512 B each = 8 column slices of
        // 64 B x 32 rows per tensor; the G threads of the work-group split the 16 slices between them
        auto prefetch_block = [&](int kbi) SYCL_ESIMD_FUNCTION {
            int key0 = kbi * KB;
            int page = key0 / BS, y0 = key0 - page * BS;
            size_t row0 = (size_t)kvh * hstride;
            if constexpr (I4) {   // int4 rows: 66 dwords (K 32 + scale, V 32 + scale): 5 slices of 16 dwords x 32 rows
                const uint32_t* bb = reinterpret_cast<const uint32_t*>(reinterpret_cast<const uint8_t*>(kc) + (size_t)bt[page] * kbs);
                for (int sl = hp ? 0 : g; sl < 5; sl += hp ? 1 : G)
                    xesimd::lsc_prefetch_2d<uint32_t, 16, 32, 1, xesimd::cache_hint::cached, xesimd::cache_hint::cached>(
                        bb, surfW, surfH, pitch, hx + sl * 16, y0);
                return;
            } else if constexpr (I8) {   // int8 layout (byte strides, head offset hx in dwords): K dwords hx..hx+63,
                                  // V dwords hx+65..hx+128 (also covers both scale dwords); 8 slices of 16 dwords x 32 rows
                const uint32_t* bb = reinterpret_cast<const uint32_t*>(reinterpret_cast<const uint8_t*>(kc) + (size_t)bt[page] * kbs);
                for (int sl = hp ? 0 : g; sl < 8; sl += hp ? 1 : G)
                    xesimd::lsc_prefetch_2d<uint32_t, 16, 32, 1, xesimd::cache_hint::cached, xesimd::cache_hint::cached>(
                        bb, surfW, surfH, pitch, hx + (sl < 4 ? sl * 16 : 65 + (sl - 4) * 16), y0);
                return;
            }
            for (int sl = hp ? 0 : g; sl < 16; sl += hp ? 1 : G) {
                const fp16* base = sl < 8 ? kc + (size_t)bt[page] * kbs + row0 : vc + (size_t)bt[page] * vbs + row0;
                xesimd::lsc_prefetch_2d<fp16, 32, 32, 1, xesimd::cache_hint::cached, xesimd::cache_hint::cached>(
                    base, surfW, surfH, pitch, (sl & 7) * 32, y0);
            }
        };
#ifdef EXL3_SPEC_PF
#ifndef EXL3_SPEC_PF_DIST
#define EXL3_SPEC_PF_DIST 1
#endif
        // prefetch distance EXL3_SPEC_PF_DIST blocks (long-context verify bandwidth experiment)
        for (int pk = kb_lo; pk < kb_hi && pk < kb_lo + EXL3_SPEC_PF_DIST; ++pk) prefetch_block(pk);
#endif
#ifdef EXL3_I4_SNEXT
        // SNEXT (I4): the next block's page index and scale / zero-point gathers are issued one
        // iteration ahead (the block-table read and the gathers were a dependent load chain at each block start)
        int blkn = 0;
        simd<uint32_t, KB> krn, vrn;
        auto ld_next = [&](int kbn) SYCL_ESIMD_FUNCTION {
            int k0n = kbn * KB, pgn = k0n / BS, y0n = k0n - pgn * BS;
            blkn = bt[pgn];
            simd<uint32_t, KB> offs(0, (uint32_t)tstride);
            offs += (uint32_t)y0n * (uint32_t)tstride;
            krn = gather<uint32_t, KB>(reinterpret_cast<const uint32_t*>(reinterpret_cast<const uint8_t*>(kc) + (size_t)blkn * kbs + (size_t)kvh * hstride + D / 2), offs);
            vrn = gather<uint32_t, KB>(reinterpret_cast<const uint32_t*>(reinterpret_cast<const uint8_t*>(vc) + (size_t)blkn * vbs + (size_t)kvh * hstride + D / 2), offs);
        };
        if constexpr (I4) { if (kb_lo < kb_hi) ld_next(kb_lo); }
#endif
#if defined(EXL3_I4_KNEXT) && defined(EXL3_I4_SNEXT) && defined(EXL3_I4_K2T)
        // KNEXT (I4): the first K line pair (d 0..127) of the NEXT block is issued during the current
        // block's P.V (the Q.K registers are free then), so each block starts with its first K tiles already in flight
        simd<uint32_t, (I4 ? 16 * KB : 4)> k2tn;
        auto ld_k2tn = [&](int kbn) SYCL_ESIMD_FUNCTION {
          if constexpr (I4) {
            int k0n = kbn * KB, pgn = k0n / BS, y0n = k0n - pgn * BS;
            const uint32_t* kbn32 = reinterpret_cast<const uint32_t*>(reinterpret_cast<const uint8_t*>(kc) + (size_t)blkn * kbs);
            k2tn.template select<8 * KB, 1>(0) = xesimd::lsc_load_2d<uint32_t, 8, KB, 1, true, false,
                xesimd::cache_hint::cached, xesimd::cache_hint::cached>(kbn32, surfW, surfH, pitch, hx, y0n);
            k2tn.template select<8 * KB, 1>(8 * KB) = xesimd::lsc_load_2d<uint32_t, 8, KB, 1, true, false,
                xesimd::cache_hint::cached, xesimd::cache_hint::cached>(kbn32, surfW, surfH, pitch, hx + 8, y0n);
          }
        };
        if constexpr (I4) { if (kb_lo < kb_hi) ld_k2tn(kb_lo); }
#define EXL3_I4_KNEXT_ON 1
#endif
        for (int kbi = kb_lo; kbi < kb_hi; ++kbi) {
            int key0 = kbi * KB;
            int page = key0 / BS, y0 = key0 - page * BS;
#ifdef EXL3_I4_SNEXT
            int blk = I4 ? blkn : bt[page];
            simd<uint32_t, KB> krc = krn, vrc = vrn;
            if constexpr (I4) { if (kbi + 1 < kb_hi) ld_next(kbi + 1); }
#else
            int blk = bt[page];
#endif
            const uint32_t* kb32 = I8 ? reinterpret_cast<const uint32_t*>(reinterpret_cast<const uint8_t*>(kc) + (size_t)blk * kbs)
                                 : F8 ? reinterpret_cast<const uint32_t*>(reinterpret_cast<const uint8_t*>(kc) + (size_t)blk * kbs + (size_t)kvh * hstride)
                                      : reinterpret_cast<const uint32_t*>(kc + (size_t)blk * kbs + (size_t)kvh * hstride);
            simd<float, KB> ksv, vsv;                  // I8: per-key scales of this 32-key block
            simd<int, KB> kzp, vzp;                    // I4: per-key zero points
            if constexpr (I4) {
                const uint8_t* kbase = reinterpret_cast<const uint8_t*>(kc) + (size_t)blk * kbs + (size_t)kvh * hstride + D / 2;
                const uint8_t* vbase8 = reinterpret_cast<const uint8_t*>(vc) + (size_t)blk * vbs + (size_t)kvh * hstride + D / 2;
                simd<uint32_t, KB> offs(0, (uint32_t)tstride);
                offs += (uint32_t)y0 * (uint32_t)tstride;
#ifdef EXL3_I4_SNEXT
                simd<uint32_t, KB> kr = krc, vr = vrc;
                (void)kbase; (void)vbase8;
#else
                simd<uint32_t, KB> kr = gather<uint32_t, KB>(reinterpret_cast<const uint32_t*>(kbase), offs);
                simd<uint32_t, KB> vr = gather<uint32_t, KB>(reinterpret_cast<const uint32_t*>(vbase8), offs);
#endif
                kzp = convert<int>(kr & 0xFu); vzp = convert<int>(vr & 0xFu);
                simd<uint32_t, KB> km = kr & 0xFFFFFFF0u, vm = vr & 0xFFFFFFF0u;
                ksv = km.template bit_cast_view<float>(); vsv = vm.template bit_cast_view<float>();
            } else if constexpr (I8) {
                const uint8_t* kbase = reinterpret_cast<const uint8_t*>(kc) + (size_t)blk * kbs + (size_t)kvh * hstride + D;
                const uint8_t* vbase8 = reinterpret_cast<const uint8_t*>(vc) + (size_t)blk * vbs + (size_t)kvh * hstride + D;
                simd<uint32_t, KB> offs(0, (uint32_t)tstride);
                offs += (uint32_t)y0 * (uint32_t)tstride;
#ifdef EXL3_I8_PROBE_S   // timing probe only: no scale gathers
                ksv = 1.0f; vsv = 1.0f;
#else
                ksv = gather<float, KB>(reinterpret_cast<const float*>(kbase), offs);
                vsv = gather<float, KB>(reinterpret_cast<const float*>(vbase8), offs);
#endif
            }
            const fp16* vbase = vc + (size_t)blk * vbs + (size_t)kvh * hstride;
            const uint32_t* vb32 = I8 ? reinterpret_cast<const uint32_t*>(reinterpret_cast<const uint8_t*>(vc) + (size_t)blk * vbs)
                                      : reinterpret_cast<const uint32_t*>(reinterpret_cast<const uint8_t*>(vc) + (size_t)blk * vbs + (size_t)kvh * hstride);
#ifdef EXL3_SPEC_PF
            if (kbi + EXL3_SPEC_PF_DIST < kb_hi) prefetch_block(kbi + EXL3_SPEC_PF_DIST);
#endif

#ifdef EXL3_I4_VEARLY
            // the block's first V message (d 0..127) is issued with the K loads, so its
            // latency hides behind Q.K + softmax (the kernel is load-latency bound at 4 threads per XVE)
            simd<uint8_t, (I4 ? 4 * 512 : 4)> vb64;
            if constexpr (I4)
                vb64 = xesimd::lsc_load_2d<uint8_t, 16, KB, 4, false, true, xesimd::cache_hint::cached, xesimd::cache_hint::cached>(
                    reinterpret_cast<const uint8_t*>(kb32), surfW, surfH, pitch, hx * 4 + 132, y0);
#endif
            simd<float, KBT * QR * KT> Sm;
            {
                simd<float, QR * KT> acc0 = 0.0f, acc1 = 0.0f;
#ifndef EXL3_I8_FP16QK
                if constexpr (I8) {
                    simd<int, QR * KT> ia0 = 0, ia1 = 0;
                    simd<uint32_t, (I4 ? 8 * KB : 4)> k8w;     // I4: the current K chunk pair (transposed dwords)
#ifdef EXL3_I4_KALL
                    simd<uint32_t, (I4 ? 32 * KB : 4)> kall;    // I4: the whole block's K (4 KB)
#endif
#ifdef EXL3_I4_K64
                    simd<uint32_t, (I4 ? 16 * KB : 4)> k64;     // I4: 64-byte K lines of 32 keys (2 KB)
#endif
#ifdef EXL3_I4_K64M
                    simd<uint32_t, (I4 ? 16 * KB : 4)> k64m;    // I4 mask path: 64-byte K lines of 32 keys (2 KB)
#endif
#ifdef EXL3_I4_K2T
                    simd<uint32_t, (I4 ? 16 * KB : 4)> k2t;     // I4: both transposed halves of the K lines (2 KB)
#endif
#pragma unroll
                    for (int c8 = 0; c8 < 8; ++c8) {
                        // transposed [8 d-quads][32 keys] = int8 VNNI B operand for both 16-key tiles
                        simd<uint32_t, 8 * KB> kt;
#ifdef EXL3_I4_U4DPAS
                        if constexpr (I4) {
                            // K as the native u4 B operand (s8 Q x u4 K: 4 elements per 16-bit channel,
                            // K = 32 per DPAS): depth row 2w = the low 16 bits (d 8w..8w+3) of each key's dword w, row
                            // 2w+1 = the high 16 bits; no nibble masks, Q in natural order
#ifdef EXL3_I4_K64   // whole 64-byte lines: [32 keys][16 dwords] (d 0..127 or 128..255) every 4 chunks, operand rows
                     // picked by a 16-dword register stride (one cache line per key row per load: no half-line re-reads)
                            {
                            if ((c8 & 3) == 0)
                                k64 = xesimd::lsc_load_2d<uint32_t, 16, KB, 1, false, false,
                                    xesimd::cache_hint::cached, xesimd::cache_hint::cached>(kb32, surfW, surfH, pitch, hx + (c8 >> 2) * 16, y0);
                            simd<uint32_t, 4 * KT> b0, b1;
#pragma unroll
                            for (int w = 0; w < 4; ++w) {
                                b0.template select<KT, 1>(w * KT) = k64.template select<KT, 16>((c8 & 3) * 4 + w);
                                b1.template select<KT, 1>(w * KT) = k64.template select<KT, 16>(KT * 16 + (c8 & 3) * 4 + w);
                            }
                            EXL3_QA_DECL(a, c8);
                            ia0 = xmx::dpas<8, 8, int, int, uint32_t, int8_t, xmx::dpas_argument_type::u4, xmx::dpas_argument_type::s8>(ia0, b0, a);
                            ia1 = xmx::dpas<8, 8, int, int, uint32_t, int8_t, xmx::dpas_argument_type::u4, xmx::dpas_argument_type::s8>(ia1, b1, a);
                            continue;
                            }
#endif
#ifdef EXL3_I4_K2T   // both 32-byte transposed halves of a 64-byte line issued back-to-back every 4 chunks
#ifdef EXL3_I4_KNEXT_ON
                            if (c8 == 0) k2t = k2tn;
                            else
#endif
#ifdef EXL3_I4_PROBE_ONELOAD   // timing probe only (wrong results): one thread per work-group loads K
                            if ((c8 & 3) == 0 && g == 0) {
#else
                            if ((c8 & 3) == 0) {
#endif
                                k2t.template select<8 * KB, 1>(0) = xesimd::lsc_load_2d<uint32_t, 8, KB, 1, true, false,
                                    xesimd::cache_hint::cached, xesimd::cache_hint::cached>(kb32, surfW, surfH, pitch, hx + (c8 >> 2) * 16, y0);
                                k2t.template select<8 * KB, 1>(8 * KB) = xesimd::lsc_load_2d<uint32_t, 8, KB, 1, true, false,
                                    xesimd::cache_hint::cached, xesimd::cache_hint::cached>(kb32, surfW, surfH, pitch, hx + (c8 >> 2) * 16 + 8, y0);
                            }
                            simd<uint32_t, 4 * KB> k4 = k2t.template select<4 * KB, 1>((c8 & 3) * 4 * KB);
#elif defined(EXL3_I4_KALL)   // all four 32-byte K loads of the block issued up front (4 loads in flight)
                            if (c8 == 0) {
#pragma unroll
                                for (int ld = 0; ld < 4; ++ld)
                                    kall.template select<8 * KB, 1>(ld * 8 * KB) = xesimd::lsc_load_2d<uint32_t, 8, KB, 1, true, false,
                                        xesimd::cache_hint::cached, xesimd::cache_hint::cached>(kb32, surfW, surfH, pitch, hx + ld * 8, y0);
                            }
                            simd<uint32_t, 4 * KB> k4 = kall.template select<4 * KB, 1>(c8 * 4 * KB);
#else
                            if ((c8 & 1) == 0)
                                k8w = xesimd::lsc_load_2d<uint32_t, 8, KB, 1, true, false,
                                    xesimd::cache_hint::cached, xesimd::cache_hint::cached>(kb32, surfW, surfH, pitch, hx + (c8 >> 1) * 8, y0);
                            simd<uint32_t, 4 * KB> k4 = k8w.template select<4 * KB, 1>((c8 & 1) * 4 * KB);
#endif
                            // operand layout (probe): dword [w][n] = d 8w..8w+7 of key n, low nibble first
                            // = the cache's own layout: tile t = keys 16t..16t+15 of each of the 4 rows
                            auto km4 = k4.template bit_cast_view<uint32_t, 4, KB>();
#ifdef EXL3_I4_U4_NOSEL   // timing probe only (wrong results): contiguous halves of the load, no 2D select
                            simd<uint32_t, 4 * KT> b0 = k4.template select<4 * KT, 1>(0);
                            simd<uint32_t, 4 * KT> b1 = k4.template select<4 * KT, 1>(4 * KT);
#else
                            simd<uint32_t, 4 * KT> b0 = km4.template select<4, 1, KT, 1>(0, 0);
                            simd<uint32_t, 4 * KT> b1 = km4.template select<4, 1, KT, 1>(0, KT);
#endif
                            EXL3_QA_DECL(a, c8);
                            ia0 = xmx::dpas<8, 8, int, int, uint32_t, int8_t, xmx::dpas_argument_type::u4, xmx::dpas_argument_type::s8>(ia0, b0, a);
                            ia1 = xmx::dpas<8, 8, int, int, uint32_t, int8_t, xmx::dpas_argument_type::u4, xmx::dpas_argument_type::s8>(ia1, b1, a);
                            continue;
                        }
#endif
                        if constexpr (I4) {
                            // transposed [4 dwords (8 nibbles = 8 d each)][32 keys] -> spread each 16-bit half into a
                            // dword of 4 bytes (d order) = the same VNNI quads as the int8 layout
#ifdef EXL3_I4_K64M   // whole 64-byte lines [32 keys][16 dwords] every 4 chunks; rows gathered by a 16-dword stride
                            if ((c8 & 3) == 0)
                                k64m = xesimd::lsc_load_2d<uint32_t, 16, KB, 1, false, false,
                                    xesimd::cache_hint::cached, xesimd::cache_hint::cached>(kb32, surfW, surfH, pitch, hx + (c8 >> 2) * 16, y0);
                            simd<uint32_t, 4 * KB> k4;
#pragma unroll
                            for (int w = 0; w < 4; ++w)
                                k4.template select<KB, 1>(w * KB) = k64m.template select<KB, 16>((c8 & 3) * 4 + w);
#else
                            // one 32-byte (64 d) transposed load per two chunks: whole cache-line halves per key row
                            if ((c8 & 1) == 0)
                                k8w = xesimd::lsc_load_2d<uint32_t, 8, KB, 1, true, false,
                                    xesimd::cache_hint::cached, xesimd::cache_hint::cached>(kb32, surfW, surfH, pitch, hx + (c8 >> 1) * 8, y0);
                            simd<uint32_t, 4 * KB> k4 = k8w.template select<4 * KB, 1>((c8 & 1) * 4 * KB);
#endif
#pragma unroll
                            for (int w = 0; w < 4; ++w) {   // low nibbles = d 8w+{0,2,4,6}, high = 8w+{1,3,5,7} (Q permuted to match)
                                simd<uint32_t, KB> x = k4.template select<KB, 1>(w * KB);
#ifdef EXL3_I4_PROBE_K   // timing probe only (wrong results): no nibble masks on K
                                kt.template select<KB, 1>((2 * w) * KB) = x;
                                kt.template select<KB, 1>((2 * w + 1) * KB) = x;
#else
                                kt.template select<KB, 1>((2 * w) * KB) = x & 0x0F0F0F0Fu;
                                kt.template select<KB, 1>((2 * w + 1) * KB) = (x >> 4) & 0x0F0F0F0Fu;
#endif
                            }
                        } else {
                            kt = xesimd::lsc_load_2d<uint32_t, 8, KB, 1, true, false,
                                xesimd::cache_hint::cached, xesimd::cache_hint::cached>(kb32, surfW, surfH, pitch, hx + c8 * 8, y0);
                        }
                        auto km = kt.template bit_cast_view<uint32_t, 8, KB>();
                        simd<uint32_t, 8 * KT> k0 = km.template select<8, 1, KT, 1>(0, 0);
                        simd<uint32_t, 8 * KT> k1 = km.template select<8, 1, KT, 1>(0, KT);
                        EXL3_QA_DECL(a, c8);
                        ia0 = xmx::dpas<8, 8, int, int, int8_t, int8_t>(ia0, k0.template bit_cast_view<int8_t>().read(), a);
                        ia1 = xmx::dpas<8, 8, int, int, int8_t, int8_t>(ia1, k1.template bit_cast_view<int8_t>().read(), a);
                    }
                    if constexpr (I4) {   // zero points: S_rk -= zp_k * sum(q8_r)
                        simd<int, QR * KT> qsr = qsum.template replicate_vs_w_hs<QR, 1, KT, 0>(0);
                        ia0 -= simd<int, KT>(kzp.template select<KT, 1>(0)).template replicate<QR>() * qsr;
                        ia1 -= simd<int, KT>(kzp.template select<KT, 1>(KT)).template replicate<QR>() * qsr;
                    }
                    simd<float, QR * KT> qrep = qs.template replicate_vs_w_hs<QR, 1, KT, 0>(0);
                    acc0 = convert<float>(ia0) * qrep;
                    acc1 = convert<float>(ia1) * qrep;
                } else
#endif
                {
#ifdef EXL3_SPEC_FEWLOADS
                simd<uint32_t, 8 * KB> ktp;
#endif
#pragma unroll
                for (int c = 0; c < NC; ++c) {
                    // transposed [8 d-pairs][32 keys]: one message feeds both 16-key tiles
#ifdef EXL3_SPEC_FEWLOADS   // timing probe only (wrong results): each thread issues 1/G of the transposed K loads
                    if (c % G == g || c == 0) ktp = xesimd::lsc_load_2d<uint32_t, 8, KB, 1, true, false,
                        xesimd::cache_hint::cached, xesimd::cache_hint::cached>(kb32, surfW, surfH, pitch, c * 8, y0);
                    simd<uint32_t, 8 * KB> kt = ktp;
#else
                    simd<uint32_t, 8 * KT> k0, k1;
                    if constexpr (F8 || I8) {
                        // bytes [32 keys][16 d] -> fp16 -> d-pair words [32 keys][8 pairs] -> [8 pairs][16 keys] per tile
                        simd<uint32_t, 128> kraw = xesimd::lsc_load_2d<uint32_t, 4, KB, 1, false, false,
                            xesimd::cache_hint::cached, xesimd::cache_hint::cached>(kb32, surfW, surfH, pitch, hx + c * 4, y0);
                        simd<fp16, 512> kf;
#ifdef EXL3_I8_PROBE_K   // timing probe only (wrong results): skip the K conversion + rearrangement
                        if constexpr (I8) { k0 = kraw.template select<128, 1>(0); k1 = kraw.template select<128, 1>(0); }
                        else
#endif
                        {
                        if constexpr (I8) kf = i8_tile(kraw); else kf = f8_tile(kraw);
                        simd<uint32_t, 256> kw = kf.template bit_cast_view<uint32_t>();
#pragma unroll
                        for (int pp = 0; pp < 8; ++pp) {
                            k0.template select<KT, 1>(pp * KT) = kw.template select<KT, 8>(pp);
                            k1.template select<KT, 1>(pp * KT) = kw.template select<KT, 8>(KT * 8 + pp);
                        }
                        }
                    } else {
                        simd<uint32_t, 8 * KB> kt = xesimd::lsc_load_2d<uint32_t, 8, KB, 1, true, false,
                            xesimd::cache_hint::cached, xesimd::cache_hint::cached>(kb32, surfW, surfH, pitch, c * 8, y0);
                        auto km = kt.template bit_cast_view<uint32_t, 8, KB>();
                        k0 = km.template select<8, 1, KT, 1>(0, 0);
                        k1 = km.template select<8, 1, KT, 1>(0, KT);
                    }
#endif
                    simd<fp16, QR * 16> a = Qc.template select<QR * 16, 1>(c * QR * 16);
                    acc0 = xmx::dpas<8, 8, float, float, fp16, fp16>(acc0, k0.template bit_cast_view<fp16>().read(), a);
                    acc1 = xmx::dpas<8, 8, float, float, fp16, fp16>(acc1, k1.template bit_cast_view<fp16>().read(), a);
                }
                }
                if constexpr (I8) {   // per-key K scales (both the int8 and the fp16-converted K paths)
                    acc0 *= simd<float, KT>(ksv.template select<KT, 1>(0)).template replicate<QR>();
                    acc1 *= simd<float, KT>(ksv.template select<KT, 1>(KT)).template replicate<QR>();
                }
                Sm.template select<QR * KT, 1>(0) = acc0 * sl2;
                Sm.template select<QR * KT, 1>(QR * KT) = acc1 * sl2;
            }
#ifdef EXL3_I4_KNEXT_ON
            if constexpr (I4) { if (kbi + 1 < kb_hi) ld_k2tn(kbi + 1); }   // Q.K is done: K registers are free
#endif
            // mask keys past each row's causal limit (and past L); only the last blocks need it
            if (key0 + KB > off + 1) {
                simd<int, KT> lane(0, 1);
#pragma unroll
                for (int t = 0; t < KBT; ++t)
#pragma unroll
                    for (int r = 0; r < QR; ++r) {
                        simd<int, KT> kj = lane + (key0 + t * KT);
                        int lim = off + (hp ? r % QLb : r);    // pad rows (r >= nrows) are discarded anyway
                        simd<float, KT> sv = Sm.template select<KT, 1>((t * QR + r) * KT);
                        sv.merge(simd<float, KT>(-INFINITY), kj > lim);
                        Sm.template select<KT, 1>((t * QR + r) * KT) = sv;
                    }
            }
            simd<float, QR> mb;
#pragma unroll
            for (int r = 0; r < QR; ++r) {
                simd<float, KT> mx = Sm.template select<KT, 1>(r * KT);
#pragma unroll
                for (int t = 1; t < KBT; ++t) mx = max(mx, simd<float, KT>(Sm.template select<KT, 1>((t * QR + r) * KT)));
                mb[r] = hmax<float>(mx);
            }
#ifdef EXL3_SPEC_LAZY
            // LAZY rescale: keep the running max (and skip the O rescale) unless some row's block
            // max exceeds it by more than EXL3_SPEC_LAZY log2 units; p = exp2(s - m) <= 2^LAZY stays exact in float
            const bool need = (mb > m + (float)EXL3_SPEC_LAZY).any();
            simd<float, QR> m_new = m;
            if (need) m_new = max(m, mb);
#else
            constexpr bool need = true;
            simd<float, QR> m_new = max(m, mb);
#endif
            // rows with nothing visible yet keep m = -inf: use 0 as the exponent base so exp2 gives 0, not NaN
            simd<float, QR> m_use = m_new;
            m_use.merge(simd<float, QR>(0.0f), m_new == -INFINITY);
            simd<float, QR> alpha = exp2(m - m_use);
            simd<float, QR * KT> mrep = m_use.template replicate_vs_w_hs<QR, 1, KT, 0>(0);
            simd<float, QR> rs = 0.0f;
            simd<fp16, KBT * QR * KT> P;
            simd<float, (I8 ? KBT * QR * KT : 16)> Pf;        // I8: v-scaled probabilities [t][r][16]
#pragma unroll
            for (int t = 0; t < KBT; ++t) {
                simd<float, QR * KT> p = exp2(simd<float, QR * KT>(Sm.template select<QR * KT, 1>(t * QR * KT)) - mrep);
#pragma unroll
                for (int r = 0; r < QR; ++r) rs[r] += reduce<float>(simd<float, KT>(p.template select<KT, 1>(r * KT)), std::plus<>());
                if constexpr (I8) {
                    p *= simd<float, KT>(vsv.template select<KT, 1>(t * KT)).template replicate<QR>();   // after the row sums
                    Pf.template select<QR * KT, 1>(t * QR * KT) = p;
                }
                P.template select<QR * KT, 1>(t * QR * KT) = convert<fp16>(p);
            }
#ifndef EXL3_I8_FP16PV
            // I8: int8 P.V on XMX. P' = p * v_scale[key] quantized per (row, 32-key block): A = [QR rows][32 keys] int8
            // (key order = tile 0 keys 0..15, tile 1 keys 16..31 = the V quad order); the V tile from the 8-bit
            // VNNI-transform load is the int8 B operand as is (no conversion). O += int32 * row scale.
            simd<int8_t, (I8 ? QR * KB : 32)> P8;
            simd<float, QR> pscale;
            simd<int, QR> vcorr = 0;                   // I4: sum_k P8_rk * zp_k per row
            if constexpr (I8) {
#pragma unroll
                for (int r = 0; r < QR; ++r) {
                    simd<float, KB> pr;
                    pr.template select<KT, 1>(0) = Pf.template select<KT, 1>(r * KT);
                    pr.template select<KT, 1>(KT) = Pf.template select<KT, 1>(QR * KT + r * KT);
                    float pm = hmax<float>(pr);
                    float sc = pm > 0.0f ? pm / 127.0f : 1.0f;
                    pscale[r] = sc;
                    P8.template select<KB, 1>(r * KB) = convert<int8_t>(rnde<float>(pr * (1.0f / sc)));
                    if constexpr (I4) vcorr[r] = reduce<int>(convert<int>(simd<int8_t, KB>(P8.template select<KB, 1>(r * KB))) * vzp, std::plus<>());
                }
            }
#endif
            if (need) l = l * alpha + rs; else l += rs;
#ifdef EXL3_I4_ZPFOLD
            if constexpr (I4) { if (need) zc *= alpha; zc += convert<float>(vcorr) * pscale; }
#endif
            m = m_new;
            simd<float, QR * 16> arep = alpha.template replicate_vs_w_hs<QR, 1, 16, 0>(0);
            // VARR (VA): 4 V tiles per 2D message (array_len 4: 4 x 16 B wide); bit-exact, -3% at B=2,
            // -6% at B=4, +1.5-3% at B=1 (microbench), so the host picks it for B >= 2
            simd<uint8_t, ((I8 && VA) ? 4 * 512 : 4)> vq4;
#ifndef EXL3_I4_VEARLY
#ifdef EXL3_I4_V2
            simd<uint8_t, (I4 ? 2 * 512 : 4)> vb64;  // I4 V2: 2 chunk pairs
#else
            simd<uint8_t, (I4 ? 4 * 512 : 4)> vb64;  // I4: 4 chunk pairs of VNNI-transformed V bytes
#endif
#endif
#ifdef EXL3_SPEC_OINPLACE
            // rescale all of O once, then each chunk adds its P.V straight into its O
            // slice (I8 path). IGC did not coalesce the loop-carried O with the per-chunk 'acc = oc * alpha + ...' copies:
            // the new O (8 KB) went to scratch every block and was copied back at the loop end (int4 5.8 KB spill)
            if constexpr (I8) {
                if (need) {
#pragma unroll
                    for (int c = 0; c < NC; ++c) O.template select<QR * 16, 1>(c * QR * 16) *= arep;
                }
            }
            simd<float, QR * 16> psrep;
            if constexpr (I8) psrep = pscale.template replicate_vs_w_hs<QR, 1, 16, 0>(0);
#endif
#pragma unroll
            for (int c = 0; c < NC; ++c) {
                auto oc = O.template select<QR * 16, 1>(c * QR * 16);
#ifdef EXL3_SPEC_OINPLACE
                simd<float, QR * 16> acc;
                if constexpr (!I8) { acc = oc; if (need) acc *= arep; }
#else
                simd<float, QR * 16> acc = oc;
                if (need) acc *= arep;
#endif
#ifndef EXL3_I8_FP16PV
                if constexpr (I8) {
                    simd<uint8_t, 512> vq;
                    bool va_done = false;
                    if constexpr (I4) {
                        // 8-bit VNNI-transform load of 16 bytes (= d 32p..32p+31, chunks 2p and 2p+1) x 32 keys every
                        // second chunk: dword [kq][b] = byte b of keys 4kq..4kq+3; its low nibbles are d 2b, the high
                        // nibbles d 2b+1 -> VNNI [kq][16 d][4 keys] int8 with two masks
                        // 4 x 16 bytes per message (array_len 4: 64 B = d 128 per load, every 8 chunks), as VARR
#ifdef EXL3_I4_V2   // 2 x 16 B per message (1 KB of registers instead of 2), every 4 chunks
                        if ((c & 3) == 0)
                            vb64.template select<1024, 1>(0) = xesimd::lsc_load_2d<uint8_t, 16, KB, 2, false, true,
                                xesimd::cache_hint::cached, xesimd::cache_hint::cached>(
                                reinterpret_cast<const uint8_t*>(kb32), surfW, surfH, pitch, hx * 4 + 132 + (c >> 2) * 32, y0);
                        simd<uint8_t, 512> vb16 = vb64.template select<512, 1>(((c & 3) >> 1) * 512);
#else
#ifdef EXL3_I4_VEARLY
                        if (c == 8)
#elif defined(EXL3_I4_PROBE_ONELOAD)
                        if ((c & 7) == 0 && g == 0)
#else
                        if ((c & 7) == 0)
#endif
                            vb64 = xesimd::lsc_load_2d<uint8_t, 16, KB, 4, false, true,
                                xesimd::cache_hint::cached, xesimd::cache_hint::cached>(
                                reinterpret_cast<const uint8_t*>(kb32), surfW, surfH, pitch, hx * 4 + 132 + (c >> 3) * 64, y0);
                        simd<uint8_t, 512> vb16 = vb64.template select<512, 1>(((c & 7) >> 1) * 512);
#endif
                        auto vm = vb16.template bit_cast_view<uint32_t, KB / 4, 16>();
                        simd<uint32_t, 64> x = vm.template select<KB / 4, 1, 8, 1>(0, (c & 1) * 8);
                        simd<uint32_t, 128> vo;
                        auto vom = vo.template bit_cast_view<uint32_t, KB / 4, 16>();
#ifdef EXL3_I4_PROBE_V   // timing probe only (wrong results): no nibble masks on V
                        vom.template select<KB / 4, 1, 8, 2>(0, 0) = x;
                        vom.template select<KB / 4, 1, 8, 2>(0, 1) = x;
#else
                        vom.template select<KB / 4, 1, 8, 2>(0, 0) = x & 0x0F0F0F0Fu;
                        vom.template select<KB / 4, 1, 8, 2>(0, 1) = (x >> 4) & 0x0F0F0F0Fu;
#endif
                        vq = vo.template bit_cast_view<uint8_t>();
                        va_done = true;
                    }
                    if constexpr (VA && !I4) {
                        if (va) {
                            if ((c & 3) == 0)
                                vq4 = xesimd::lsc_load_2d<uint8_t, 16, KB, 4, false, true,
                                    xesimd::cache_hint::cached, xesimd::cache_hint::cached>(
                                    reinterpret_cast<const uint8_t*>(kb32), surfW, surfH, pitch, hx * 4 + 260 + c * 16, y0);
                            vq = vq4.template select<512, 1>((c & 3) * 512);
                            va_done = true;
                        }
                    }
                    if (!va_done) {
                        vq = xesimd::lsc_load_2d<uint8_t, 16, KB, 1, false, true,
                            xesimd::cache_hint::cached, xesimd::cache_hint::cached>(
                            reinterpret_cast<const uint8_t*>(kb32), surfW, surfH, pitch, hx * 4 + 260 + c * 16, y0);
                    }
                    simd<int, QR * 16> ai = 0;
                    ai = xmx::dpas<8, 8, int, int, int8_t, int8_t>(ai, vq.template bit_cast_view<int8_t>().read(), P8);
#ifndef EXL3_I4_ZPFOLD
                    if constexpr (I4) ai -= vcorr.template replicate_vs_w_hs<QR, 1, 16, 0>(0);
#endif
#ifdef EXL3_SPEC_OINPLACE
                    O.template select<QR * 16, 1>(c * QR * 16) += convert<float>(ai) * psrep;
#else
                    acc += convert<float>(ai) * pscale.template replicate_vs_w_hs<QR, 1, 16, 0>(0);
                    oc = acc;
#endif
                    continue;
                }
#endif
                // VNNI [16 key pairs][16 d][2]: key tile t = key pairs 8t..8t+7, a contiguous 256-half slice
                simd<fp16, KB * 16> vt;
#ifndef EXL3_I8_V_PLAIN
                if constexpr (I8) {
                    // 8-bit VNNI-transform load: [8 key quads][16 d][4] int8 -> fp16 (positions kept) -> as dwords
                    // [kq][d][2 pairs] -> fp16 VNNI [16 key pairs][16 d][2] with kp = 2*kq + j (stride-2 dword moves)
                    simd<uint8_t, 512> vq = xesimd::lsc_load_2d<uint8_t, 16, KB, 1, false, true,
                        xesimd::cache_hint::cached, xesimd::cache_hint::cached>(
                        reinterpret_cast<const uint8_t*>(kb32), surfW, surfH, pitch, hx * 4 + 260 + c * 16, y0);
#ifdef EXL3_I8_PROBE_V   // timing probe only (wrong results): no V conversion / regrouping
                    vt = vq.template bit_cast_view<fp16>().read().template select<512, 1>(0);
#else
                    simd<fp16, 512> vh = convert<fp16>(vq.template bit_cast_view<int8_t>().read());
                    simd<uint32_t, 256> vw = vh.template bit_cast_view<uint32_t>();
                    simd<uint32_t, 256> vo;
#pragma unroll
                    for (int kq = 0; kq < KB / 4; ++kq) {
                        vo.template select<16, 1>((2 * kq) * 16) = vw.template select<16, 2>(kq * 32);
                        vo.template select<16, 1>((2 * kq + 1) * 16) = vw.template select<16, 2>(kq * 32 + 1);
                    }
                    vt = vo.template bit_cast_view<fp16>();
#endif
                } else
#endif
                if constexpr (F8 || I8) {
                    // bytes [32 keys][16 d] -> fp16 -> VNNI [16 key pairs][16 d][2]
                    simd<uint32_t, 128> vraw;
                    if constexpr (I8)   // V bytes sit 260 B (65 dwords) after K in the head row; load from the K block base
                        vraw = xesimd::lsc_load_2d<uint32_t, 4, KB, 1, false, false,
                            xesimd::cache_hint::cached, xesimd::cache_hint::cached>(kb32, surfW, surfH, pitch, hx + 65 + c * 4, y0);
                    else
                        vraw = xesimd::lsc_load_2d<uint32_t, 4, KB, 1, false, false,
                            xesimd::cache_hint::cached, xesimd::cache_hint::cached>(vb32, surfW, surfH, pitch, c * 4, y0);
                    simd<fp16, 512> vf;
                    if constexpr (I8) vf = i8_tile(vraw); else vf = f8_tile(vraw);
#pragma unroll
                    for (int kp = 0; kp < KB / 2; ++kp) {
                        vt.template select<16, 2>(kp * 32) = vf.template select<16, 1>(2 * kp * 16);
                        vt.template select<16, 2>(kp * 32 + 1) = vf.template select<16, 1>((2 * kp + 1) * 16);
                    }
                } else {
                    vt = xesimd::lsc_load_2d<fp16, 16, KB, 1, false, true,
                        xesimd::cache_hint::cached, xesimd::cache_hint::cached>(vbase, surfW, surfH, pitch, c * 16, y0);
                }
#pragma unroll
                for (int t = 0; t < KBT; ++t)
                    acc = xmx::dpas<8, 8, float, float, fp16, fp16>(
                        acc, simd<fp16, KT * 16>(vt.template select<KT * 16, 1>(t * KT * 16)),
                        simd<fp16, QR * 16>(P.template select<QR * KT, 1>(t * QR * KT)));
                oc = acc;
            }
        }

        constexpr float LN2 = 0.6931471805599453f;
        auto epi_row = [&](int r) SYCL_ESIMD_FUNCTION {
            int rp = hp ? r % QLb : r, rh = hp ? kvh * G + r / QLb : h;
            size_t row = ((size_t)(b * Sp + sp) * QL + rp) * Hq + rh;
            float lr = l[r];
            float mr = m[r];
            float ir = lr > 0.0f ? 1.0f / lr : 0.0f;
            if constexpr (F8) ir *= vs[0];   // I8: V scales were applied per key to P
            simd<fp16, D> orow;
#pragma unroll
            for (int c = 0; c < NC; ++c)
#ifdef EXL3_I4_ZPFOLD
                if constexpr (I4) orow.template select<16, 1>(c * 16) = convert<fp16>((simd<float, 16>(O.template select<16, 1>((c * QR + r) * 16)) - (float)zc[r]) * ir);
                else
#endif
                orow.template select<16, 1>(c * 16) = convert<fp16>(simd<float, 16>(O.template select<16, 1>((c * QR + r) * 16)) * ir);
            block_store<fp16, D>(opart + row * D, orow);
            lpart[row] = lr > 0.0f ? (mr + sycl::log2(lr)) * LN2 : -INFINITY;
        };
#ifdef EXL3_SPEC_EPIU
        // unrolled epilogue for the plain instances: static O indices. The break loop indexed O
        // by a runtime row, and IGC then kept a memory copy of O through the key loop (int4 5.8 KB / int8 5.2 KB spill,
        // filled + spilled every 32-key block). The VARR instance never spilled and is ~2 % faster with the loop.
        if constexpr (!VA) {
#pragma unroll
            for (int r = 0; r < QR; ++r) if (r < nrows) epi_row(r);
        } else
#endif
        {
            for (int r = 0; r < QR; ++r) { if (r >= nrows) break; epi_row(r); }
        }
    }
};

// out[b*QL + r, h, :] = sum_s w_s O_s / sum_s w_s,  w_s = exp(lse_s - max_s lse_s).
// Work-group = one (b, r, h) row, one work-item per head-dim element (D = 256 = work-group size). The split
// weights are computed once per work-group (group reductions, SLM) and the split loop is unrolled so the
// independent partial loads overlap (the first version walked 64 dependent loads per item: 35 us per call).
constexpr int MERGE_MAX_S = 256;   // HEAD-PACK: up to 256 key splits (S * G partials)
struct FaSpecMerge {
    const fp16* opart; const float* lpart; fp16* out;
    int B, QL, Hq, S, D, out_stride;               // out row stride in elements (per token row)
    const int* qsl = nullptr;                      // varlen (see FaSpecKernelT): QL = partial row stride
    sycl::local_accessor<float, 1> wsm;            // MERGE_MAX_S
    sycl::local_accessor<float, 1> xsm;            // I4-ROT: one output row (D = work-group size = 256)
    const float* rsign = nullptr;                  // I4-ROT: o = s * (H o_r) / 256; nullptr = off
    void operator()(sycl::nd_item<1> it) const {
        int d = it.get_local_id(0);
        int rowid = it.get_group(0);
        int h = rowid % Hq; int rest = rowid / Hq;
        int r = rest % QL;
        int b = rest / QL;
        int orow = b * QL + r;
        if (qsl) { int q0 = qsl[b], n = qsl[b + 1] - q0; if (n <= 0 || n > 8 || r >= n) return; orow = q0 + r; }
        auto grp = it.get_group();
        size_t row0 = ((size_t)(b * S) * QL + r) * Hq + h, rstep = (size_t)QL * Hq;
        float v = d < S ? lpart[row0 + d * rstep] : -INFINITY;
        float mx = sycl::reduce_over_group(grp, v, sycl::maximum<float>());
        float w = (d < S && v != -INFINITY) ? sycl::exp(v - mx) : 0.0f;
        float wsum = sycl::reduce_over_group(grp, w, sycl::plus<float>());
        if (d < MERGE_MAX_S) wsm[d] = w;
        sycl::group_barrier(grp);
        float acc = 0.0f;
        const fp16* o = opart + row0 * D + d;
        size_t ostep = rstep * D;
#pragma unroll 16
        for (int s2 = 0; s2 < S; ++s2) acc += wsm[s2] * (float)o[s2 * ostep];
        float val = wsum > 0.0f ? acc / wsum : 0.0f;
        if (rsign) {   // I4-ROT: in-SLM Walsh-Hadamard over the row (uniform per work-group)
            xsm[d] = val;
            sycl::group_barrier(grp);
            for (int L = 1; L < 256; L <<= 1) {
                if ((d & L) == 0) { float lo = xsm[d], hi = xsm[d + L]; xsm[d] = lo + hi; xsm[d + L] = lo - hi; }
                sycl::group_barrier(grp);
            }
            val = xsm[d] * rsign[d] * (1.0f / 256.0f);
        }
        out[(size_t)orow * out_stride + (size_t)h * D + d] = (fp16)val;
    }
};

using FaSpecKernel = FaSpecKernelT<0>;
}  // namespace exl3fa
