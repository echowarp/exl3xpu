#pragma once
// Prefill (chunk) attention straight from vLLM's int8_per_token_head KV cache for Intel Xe2 in ESIMD.
// Replaces gather-dequant (int8 -> fp16 copy of the whole prefix) + fp16 FA2 + LSE merge.
//
// Why int8 math: an 8-row-per-thread FMHA on Xe2 is limited by L1 -> register bandwidth for the K/V tiles (every
// thread streams the whole K/V of its range; measured 38-48 TFLOPS for fa_esimd.h, cutlass 61-75 of ~184 fp16).
// int8 tiles are half the bytes and int8 DPAS runs at twice the fp16 rate, so the same bandwidth bound is ~2x higher.
// Same math as FaSpecKernelT<2> (spec decode): Q quantized per row (amax/127), S = int8 Q . int8 K (int32 exact)
// * q_scale * k_scale[key]; online softmax (base 2); P' = p * v_scale[key] quantized per (row, 32-key block) to
// int8; O += int8 P' . int8 V * p_scale. PV8=false keeps P in fp16 and converts V to fp16 in registers instead.
//
// Thread = 8 query rows x 1 query head (256-GRF). Work-group = NQ query tiles x the G query heads of one KV head:
// its threads walk the same keys at about the same time, so K/V come from L1/L2. Consecutive work-groups share the
// KV head (L2 locality); the latest (longest) query tiles are scheduled first (load balance).
// Causal, bottom-right aligned: query row i of Lq sees keys [0, L - Lq + i].

#include <sycl/sycl.hpp>
#include <sycl/ext/intel/esimd.hpp>
#include <sycl/ext/intel/esimd/xmx/dpas.hpp>
#include <sycl/ext/intel/experimental/esimd/memory.hpp>

namespace exl3fp {

using namespace sycl::ext::intel::esimd;
namespace xesimd = sycl::ext::intel::experimental::esimd;
using fp16 = sycl::half;

template <int NQ = 2, bool PV8 = true, int KBT = 4, bool LAZY = true>
struct FaPrefillI8 {
    static constexpr int NQ_ = NQ;
    static constexpr int D = 256, QR = 8, KT = 16, KB = KT * KBT, NC = D / 16, NH = KB / 32;   // NH 32-key halves
    const fp16* q;                 // [Lq, Hq, D] contiguous
    const uint8_t* kc;             // K view base: [nblocks][BS][Hk][520] bytes (V at +260 in each head row)
    const int* pages;              // [ceil(L / BS)] block ids of this sequence
    fp16* out;                     // row i, head h at out + i * out_stride + h * D
    int Lq, L, Hq, Hk, G, BS, out_stride, nqt;
    float scale_log2;
    size_t kbs;                    // block stride (bytes)
    int tstride, hstride;          // token / head strides (bytes)

    void operator()(sycl::nd_item<1> it) const SYCL_ESIMD_KERNEL {
        const int lid = it.get_local_id(0);
        const int g = lid % G, qsub = lid / G;
        const int gi = it.get_group(0);
        const int nqg = (nqt + NQ - 1) / NQ;
        const int qg = nqg - 1 - gi % nqg;             // longest tiles first
        const int kvh = gi / nqg;
        const int qt = qg * NQ + qsub;
        if (qt >= nqt) return;                         // no barriers in this kernel
        const int h = kvh * G + g;
        const int row0 = qt * QR;
        const int off = L - Lq;

        // Q rows -> int8 [8 chunks of 32 d][QR rows][32], per-row scales
        simd<int8_t, 8 * QR * 32> Qi8 = 0;
        simd<float, QR> qs = 1.0f;
#pragma unroll
        for (int r = 0; r < QR; ++r) {
            if (row0 + r < Lq) {
                simd<float, D> rowf = convert<float>(simd<fp16, D>(block_load<fp16, D>(q + ((size_t)(row0 + r) * Hq + h) * D)));
                float amax = hmax<float>(abs(rowf));
                float sc = amax > 0.0f ? amax / 127.0f : 1.0f;
                simd<int8_t, D> q8 = convert<int8_t>(rnde<float>(rowf * (1.0f / sc)));
#pragma unroll
                for (int c8 = 0; c8 < 8; ++c8)
                    Qi8.template select<32, 1>((c8 * QR + r) * 32) = q8.template select<32, 1>(c8 * 32);
                qs[r] = sc;
            }
        }
        const simd<float, QR * KT> qrep = qs.template replicate_vs_w_hs<QR, 1, KT, 0>(0) * scale_log2;

        simd<float, NC * QR * 16> O = 0.0f;
        simd<float, QR> m = -INFINITY, l = 0.0f;

        const unsigned surfW = (unsigned)tstride - 1, surfH = (unsigned)BS - 1, pitch = (unsigned)tstride - 1;
        const int hx = kvh * hstride / 4;              // head offset in dwords
        const int rlast = (row0 + QR < Lq ? row0 + QR : Lq) - 1;
        const int nkb = (off + rlast) / KB + 1;        // blocks with any visible key
        const int mask_from = (off + row0 + 1) / KB;   // first block where row 0 sees fewer than all 32 keys
        simd<uint32_t, KB> lane_off(0, (uint32_t)tstride);
        constexpr float LAZY_T = 8.0f;                 // log2 units: rescale O only when a row max grows by > 2^8

#ifdef EXL3_PF_DIST
        // cooperative L1 prefetch EXL3_PF_DIST blocks ahead: the head rows [K 260 | V 260] of KB keys = 9 slices of
        // 64 B x min(KB, 32) rows per 32-key half; the G threads of the work-group (same KV head) split the slices
        auto prefetch_block = [&](int kbp) SYCL_ESIMD_FUNCTION {
            const int kp0 = kbp * KB, pg = kp0 / BS, yp = kp0 - pg * BS;
            const uint8_t* pb = kc + (size_t)pages[pg] * kbs;
            for (int sl = g + qsub * G; sl < 9 * NH; sl += G * NQ)
                xesimd::lsc_prefetch_2d<uint32_t, 16, 32, 1, xesimd::cache_hint::cached, xesimd::cache_hint::cached>(
                    reinterpret_cast<const uint32_t*>(pb), surfW, surfH, pitch, hx + (sl % 9) * 16, yp + (sl / 9) * 32);
        };
#pragma unroll
        for (int d0 = 0; d0 < EXL3_PF_DIST; ++d0) if (d0 < nkb) prefetch_block(d0);
#endif
        for (int kbi = 0; kbi < nkb; ++kbi) {
            const int key0 = kbi * KB;
#ifdef EXL3_PF_DIST
            if (kbi + EXL3_PF_DIST < nkb) prefetch_block(kbi + EXL3_PF_DIST);
#endif
            const int page = key0 / BS, y0 = key0 - page * BS;
            const uint8_t* blk = kc + (size_t)pages[page] * kbs;
            const uint32_t* kb32 = reinterpret_cast<const uint32_t*>(blk);
            simd<uint32_t, KB> offs = lane_off + (uint32_t)y0 * (uint32_t)tstride;
            const uint8_t* hrow = blk + (size_t)kvh * hstride;
            simd<float, KB> ksv, vsv;
#pragma unroll
            for (int hh = 0; hh < NH; ++hh) {
                simd<uint32_t, 32> o2 = offs.template select<32, 1>(hh * 32);
                ksv.template select<32, 1>(hh * 32) = gather<float, 32>(reinterpret_cast<const float*>(hrow + D), o2);
                vsv.template select<32, 1>(hh * 32) = gather<float, 32>(reinterpret_cast<const float*>(hrow + 260 + D), o2);
            }

            // S = Q K^T (int8 x int8 -> int32), 16-key tiles
            simd<float, KBT * QR * KT> Sm;
#pragma unroll
            for (int hh = 0; hh < NH; ++hh) {
                simd<int, QR * KT> ia0 = 0, ia1 = 0;
#pragma unroll
                for (int c8 = 0; c8 < 8; ++c8) {
                    simd<uint32_t, 8 * 32> kt = xesimd::lsc_load_2d<uint32_t, 8, 32, 1, true, false,
                        xesimd::cache_hint::cached, xesimd::cache_hint::cached>(kb32, surfW, surfH, pitch, hx + c8 * 8, y0 + hh * 32);
                    auto km = kt.template bit_cast_view<uint32_t, 8, 32>();
                    simd<uint32_t, 8 * KT> k0 = km.template select<8, 1, KT, 1>(0, 0);
                    simd<uint32_t, 8 * KT> k1 = km.template select<8, 1, KT, 1>(0, KT);
                    simd<int8_t, QR * 32> a = Qi8.template select<QR * 32, 1>(c8 * QR * 32);
                    ia0 = xmx::dpas<8, 8, int, int, int8_t, int8_t>(ia0, k0.template bit_cast_view<int8_t>().read(), a);
                    ia1 = xmx::dpas<8, 8, int, int, int8_t, int8_t>(ia1, k1.template bit_cast_view<int8_t>().read(), a);
                }
                Sm.template select<QR * KT, 1>((2 * hh) * QR * KT) = convert<float>(ia0) * qrep *
                    simd<float, KT>(ksv.template select<KT, 1>(hh * 32)).template replicate<QR>();
                Sm.template select<QR * KT, 1>((2 * hh + 1) * QR * KT) = convert<float>(ia1) * qrep *
                    simd<float, KT>(ksv.template select<KT, 1>(hh * 32 + KT)).template replicate<QR>();
            }
            if (kbi >= mask_from) {
                simd<int, KT> lane(0, 1);
#pragma unroll
                for (int t = 0; t < KBT; ++t)
#pragma unroll
                    for (int r = 0; r < QR; ++r) {
                        simd<int, KT> kj = lane + (key0 + t * KT);
                        simd<float, KT> sv = Sm.template select<KT, 1>((t * QR + r) * KT);
                        sv.merge(simd<float, KT>(-INFINITY), kj > off + row0 + r);
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
            simd<float, QR> m_new = max(m, mb);
            bool resc = true;
            if constexpr (LAZY) {
                // keep the stale max while no row grows by more than LAZY_T (p <= 2^LAZY_T: fine in fp32, and P is
                // re-quantized per row and block anyway); the first block always rescales (m = -inf)
                simd<ushort, QR> grow = mb > m + LAZY_T;
                resc = grow.any();
                if (!resc) m_new = m;
            }
            simd<float, QR> m_use = m_new;
            m_use.merge(simd<float, QR>(0.0f), m_new == -INFINITY);
            simd<float, QR> alpha = exp2(m - m_use);
            simd<float, QR * KT> mrep = m_use.template replicate_vs_w_hs<QR, 1, KT, 0>(0);
            simd<float, QR> rs = 0.0f;
            simd<float, KBT * QR * KT> Pf;
#pragma unroll
            for (int t = 0; t < KBT; ++t) {
                simd<float, QR * KT> p = exp2(simd<float, QR * KT>(Sm.template select<QR * KT, 1>(t * QR * KT)) - mrep);
#pragma unroll
                for (int r = 0; r < QR; ++r) rs[r] += reduce<float>(simd<float, KT>(p.template select<KT, 1>(r * KT)), std::plus<>());
                p *= simd<float, KT>(vsv.template select<KT, 1>(t * KT)).template replicate<QR>();   // after the row sums
                Pf.template select<QR * KT, 1>(t * QR * KT) = p;
            }
            l = l * alpha + rs;
            m = m_new;
            simd<float, QR * 16> arep = alpha.template replicate_vs_w_hs<QR, 1, 16, 0>(0);

            if constexpr (PV8) {
                // P' quantized per (row, block of KB keys): A operands [QR rows][32 keys] per 32-key half
                simd<int8_t, QR * KB> P8;                 // [half][QR][32]
                simd<float, QR> pscale;
#pragma unroll
                for (int r = 0; r < QR; ++r) {
                    simd<float, KB> pr;
#pragma unroll
                    for (int t = 0; t < KBT; ++t) pr.template select<KT, 1>(t * KT) = Pf.template select<KT, 1>((t * QR + r) * KT);
                    float pm = hmax<float>(pr);
                    float sc = pm > 0.0f ? pm / 127.0f : 1.0f;
                    pscale[r] = sc;
                    simd<int8_t, KB> p8 = convert<int8_t>(rnde<float>(pr * (1.0f / sc)));
#pragma unroll
                    for (int hh = 0; hh < NH; ++hh) P8.template select<32, 1>((hh * QR + r) * 32) = p8.template select<32, 1>(hh * 32);
                }
                simd<float, QR * 16> prep = pscale.template replicate_vs_w_hs<QR, 1, 16, 0>(0);
#pragma unroll
                for (int c = 0; c < NC; ++c) {
                    auto oc = O.template select<QR * 16, 1>(c * QR * 16);
                    simd<int, QR * 16> ai = 0;
#pragma unroll
                    for (int hh = 0; hh < NH; ++hh) {
                        simd<uint8_t, 512> vq = xesimd::lsc_load_2d<uint8_t, 16, 32, 1, false, true,
                            xesimd::cache_hint::cached, xesimd::cache_hint::cached>(blk, surfW, surfH, pitch, hx * 4 + 260 + c * 16, y0 + hh * 32);
                        ai = xmx::dpas<8, 8, int, int, int8_t, int8_t>(ai, vq.template bit_cast_view<int8_t>().read(),
                                                                       simd<int8_t, QR * 32>(P8.template select<QR * 32, 1>(hh * QR * 32)));
                    }
                    if (resc) oc = simd<float, QR * 16>(oc) * arep + convert<float>(ai) * prep;
                    else oc = simd<float, QR * 16>(oc) + convert<float>(ai) * prep;
                }
            } else {
                static_assert(PV8 || KBT == 2, "fp16 PV path: 32-key blocks only");
                simd<fp16, KBT * QR * KT> P = convert<fp16>(Pf);
#pragma unroll
                for (int c = 0; c < NC; ++c) {
                    auto oc = O.template select<QR * 16, 1>(c * QR * 16);
                    simd<float, QR * 16> acc = simd<float, QR * 16>(oc) * arep;
                    // 8-bit VNNI-transform load [8 key quads][16 d][4] -> fp16 -> VNNI [16 key pairs][16 d][2]
                    simd<uint8_t, 512> vq = xesimd::lsc_load_2d<uint8_t, 16, KB, 1, false, true,
                        xesimd::cache_hint::cached, xesimd::cache_hint::cached>(blk, surfW, surfH, pitch, hx * 4 + 260 + c * 16, y0);
                    simd<fp16, 512> vh = convert<fp16>(vq.template bit_cast_view<int8_t>().read());
                    simd<uint32_t, 256> vw = vh.template bit_cast_view<uint32_t>();
                    simd<uint32_t, 256> vo;
#pragma unroll
                    for (int kq = 0; kq < KB / 4; ++kq) {
                        vo.template select<16, 1>((2 * kq) * 16) = vw.template select<16, 2>(kq * 32);
                        vo.template select<16, 1>((2 * kq + 1) * 16) = vw.template select<16, 2>(kq * 32 + 1);
                    }
                    simd<fp16, 512> vt = vo.template bit_cast_view<fp16>();
#pragma unroll
                    for (int t = 0; t < KBT; ++t)
                        acc = xmx::dpas<8, 8, float, float, fp16, fp16>(
                            acc, simd<fp16, KT * 16>(vt.template select<KT * 16, 1>(t * KT * 16)),
                            simd<fp16, QR * 16>(P.template select<QR * KT, 1>(t * QR * KT)));
                    oc = acc;
                }
            }
        }

#pragma unroll
        for (int r = 0; r < QR; ++r) {
            if (row0 + r >= Lq) break;
            float lr = l[r];
            float ir = lr > 0.0f ? 1.0f / lr : 0.0f;
            simd<fp16, D> orow;
#pragma unroll
            for (int c = 0; c < NC; ++c)
                orow.template select<16, 1>(c * 16) = convert<fp16>(simd<float, 16>(O.template select<16, 1>((c * QR + r) * 16)) * ir);
            block_store<fp16, D>(out + (size_t)(row0 + r) * out_stride + (size_t)h * D, orow);
        }
    }
};

}  // namespace exl3fp
