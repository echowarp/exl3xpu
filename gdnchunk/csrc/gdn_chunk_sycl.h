// SPDX-License-Identifier: Apache-2.0
//
// gdn_chunk_sycl.h: gated delta rule prefill (chunked WY form, 16-row pieces) for Intel Xe2 (Arc Pro B70).
//
// Port of Magnitude's `gated_delta_chunk` kernels
//   inference/engine/kernels/kernels/cuda/gated_delta_chunk.cu and lib/recurrent/recurrent.cuh
//   (https://github.com/magnitudedev/magnitude, commit 7f029060a37ed1f7884db038222dc8f36430a08e)
// Copyright 2026 Magnitude AI Inc., licensed under the Apache License, Version 2.0 (../LICENSE, ../NOTICE).
// Modifications (2026, exl3xpu Arc B70 fork):
//  * SYCL + ESIMD for Xe2 instead of CUDA: the scan runs on DPAS (8x16x16 f16 -> f32, repeat count 8 = 8 state rows
//    per thread) instead of mma.m16n8k16; the prepare kernel writes the scan's DPAS operands pre-packed (VNNI) so
//    the scan uses plain 512-byte block loads instead of cp.async staging + ldmatrix.
//  * Inputs in the Arc engine's layout (vllm_xpu_kernels gdn_attention, reorder_input): qkvz [M, q|k|v|z] f16,
//    ba [M, b|a] f16, conv weights f16 [CH, 4], A_log f32, dt_bias f16; state f32 [S, HV, V, K]; conv window =
//    rows 0..2 of vLLM's conv_state [S, >=3, CH] f16; raw core output [tokens, HV, V] f16 (+ z copied out of qkvz,
//    as the current op does). The output gate (RMSNormGated) is not part of this port.
//  * No tape, no MTP-verify row-sequential path: every row goes through WY pieces.
//  * Partial-block reloads, same contract and formats as the CUDA port (exl3cuda, csrc/gdnchunk/gdn_chunk.cu):
//    pieces aligned to ABSOLUTE positions (pos0); resume from a state slot `src` whose checkpoint may carry up to 15
//    raw tail rows (replayed without outputs); any number of checkpoints per slot, each (pos, dst, kind):
//    kind 0 "exact" (state after row pos, the piece is cut there), kind 1 "aligned + tail" (state at the last
//    16-aligned row >= the first processed row, plus the raw rows up to pos; no cut). Window/tail/tail_count are
//    published by separate gather + scatter launches at the end, so in-place slots (dst == src) are safe.
//
// Math per piece (rows t < n <= 16, zero padded), cumulative log decay G_t, gamma_t = exp(G_t), state S [V x K]:
//   A[t][s] = beta_t exp(G_t - G_s) k_t.k_s (s < t),   Tb = (I + A)^-1 diag(beta)
//   D[t][s] = exp(G_t - G_s) q_t.k_s (s <= t)
//   U = Tb (V - diag(gamma) K S^T),   O = diag(gamma) Q S^T + D U,   S <- gamma_last S + U^T diag(gamma_last / gamma) K
// q, k: conv1d (width 4) + SiLU + L2 norm (q also * K^-1/2), beta = sigmoid(b), log decay = -exp(A_log) softplus(a + dt_bias).
//
// Launches: GdnPlan (piece table) -> GdnPrepare (SYCL, per piece x key head) -> GdnScan (ESIMD, per slot x value head
// x 8 state rows; publishes checkpoint states) -> GdnPubGather -> GdnPubScatter (windows, tails, tail counts) -> GdnZCopy.
#pragma once
#include <sycl/sycl.hpp>
#include <sycl/ext/intel/esimd.hpp>
#include <sycl/ext/intel/esimd/xmx/dpas.hpp>
#include <sycl/ext/intel/experimental/grf_size_properties.hpp>

namespace gdnc {

using half = sycl::half;
using fp16 = sycl::half;
constexpr int W = 128;          // head_k_dim == head_v_dim
constexpr int PIECE = 16;       // rows per piece (= DPAS K and N)
constexpr int TAIL = PIECE - 1; // most raw rows an aligned checkpoint carries
constexpr int CONVW = 4;        // conv kernel width
constexpr int PREP_WG = 256;    // prepare work-group: 16 sub-groups of 16
constexpr int ROWS_T = 8;       // state rows per scan thread (= DPAS repeat count)
#ifndef GDNC_SCAN_SPLIT
#define GDNC_SCAN_SPLIT 1       // work-groups per (slot, value head). 2 = 96 WGs of 8 threads for one prompt,
                                // so all 32 Xe cores get work (16-thread WGs: 48 of them sit on 24 cores). Each thread keeps
                                // its own 8 state rows and arithmetic: results bit-identical to SPLIT 1.
#endif
#ifndef GDNC_PF
#define GDNC_PF 0               // 1 = cache prefetch of the NEXT piece's q/k block + record at the start of
                                // each piece (no registers; the scan step is load-latency bound). Bit-identical.
#endif
#ifndef GDNC_PF_DIST
#define GDNC_PF_DIST 1          // pieces ahead for GDNC_PF
#endif
#ifndef GDNC_PF_L1
#define GDNC_PF_L1 1            // 1: prefetch into L1+L2; 0: L2 only
#endif
#ifndef GDNC_PF_QK256
#define GDNC_PF_QK256 0         // 1: q/k block prefetched in 256-byte messages (it is a multiple of 256 B)
#endif
#ifndef GDNC_VNNI2D
#define GDNC_VNNI2D 0           // prepare writes q, k_hi, k_lo as plain row-major [16][128] tiles; the scan
                                // gets its DPAS B operands with LSC 2D block loads (transposed dword = VNNI for S.K^T / S.Q^T,
                                // VNNI-transform for the update), as fa_esimd.h. Same f16 operands: bit-identical.
#endif
constexpr int QK2_Q = 0, QK2_KH = 2048, QK2_KL = 4096, QK2_USED = 6144;   // halves, inside the QK_BLOCK slot
#ifndef GDNC_PLANPAR
#define GDNC_PLANPAR 0          // the plan kernel only sizes the slots; a parallel kernel fills the piece table
                                // of every slot whose exact checkpoints all lie on the 16-row grid (the common case).
                                // Same table: bit-identical.
#endif
#ifndef GDNC_ZFUSE
#define GDNC_ZFUSE 0            // prepare copies z for its piece's new rows and value heads (no GdnZCopy launch)
#endif
#ifndef GDNC_CONV_VEC
#define GDNC_CONV_VEC 2        // channels per work-item in the CONV2 path (2 or 4; same arithmetic)
#endif
#ifndef GDNC_VHALF
#define GDNC_VHALF 0            // prepare keeps the v columns in SLM as f16 (they are only read as f16):
                                // 46.8 -> 34.5 KB SLM per work-group, more groups per Xe-core. Bit-identical. Needs CONV2.
#endif
#if GDNC_VHALF && !GDNC_CONV2
#error "GDNC_VHALF needs GDNC_CONV2"
#endif
#ifndef GDNC_SLMPAD
#define GDNC_SLMPAD 0           // pad the prepare kernel's xs row stride (SLM bank conflicts). Layout only.
#endif
#ifndef GDNC_CONV2
#define GDNC_CONV2 0            // prepare conv with per-row source pointers computed once per item (not per
                                // element through raw_at) and channel pairs per 4-byte load. Same arithmetic: bit-identical.
#endif
#ifndef GDNC_DIAG_PREP
#define GDNC_DIAG_PREP 0        // DIAGNOSTIC ONLY (wrong results): skip prepare sections, 1 conv, 2 norms, 4 gates, 8 operand
                                // packing, 16 Gram/A/D/Tb; the cost of a section = the time it saves
#endif
#define PREP_ON(b) (!(GDNC_DIAG_PREP & (b)))
#ifndef GDNC_DIAG_NOSCAN
#define GDNC_DIAG_NOSCAN 0      // DIAGNOSTIC ONLY (wrong results): the scan returns at once, to time plan + prepare
#endif
constexpr int SCAN_SPLIT = GDNC_SCAN_SPLIT;
constexpr int SCAN_WG = W / ROWS_T / SCAN_SPLIT;   // threads per work-group (16 / SPLIT)
static_assert(W / ROWS_T % SCAN_SPLIT == 0, "split must divide the 16 row groups");
constexpr int PD = 4;           // ints per piece descriptor: slot, first absolute row, count, pad
#ifndef GDNC_UNROLL
#define GDNC_UNROLL 8           // scan key-block loops (no spills at 8 or 2 in 256-GRF mode; sweep in the GPU window)
#endif

// Per (piece, key head) block of f16 DPAS B operands, 8 key-column blocks of 256 halves each:
//   QT, KT, KLT : B[k = key col c][n = piece row t]  (S Q^T, S K_high^T, S K_low^T), VNNI index (c/2)*32 + t*2 + c%2
//   KV, KLV     : B[k = piece row t][n = key col c]  (state update),               VNNI index (t/2)*32 + c*2 + t%2
constexpr int QK_QT = 0, QK_KT = 2048, QK_KLT = 4096, QK_KV = 6144, QK_KLV = 8192, QK_BLOCK = 10240;   // halves
// Per (piece, value head) record: VT [W rows][16 t] f16 (v transposed); TbB / DB: B[k = t][n = t'] = Tb[t'][t] /
// D[t'][t] (VNNI); gates f32: gamma[16], tail[16] = gamma_last / gamma, last, 15 pad.
constexpr int REC_VT = 0, REC_TB = W * PIECE, REC_D = REC_TB + 256, REC_G = REC_D + 256, REC = REC_G + 96;   // halves
static_assert((QK_BLOCK * 2) % 64 == 0 && (REC * 2) % 64 == 0 && (REC_G * 2) % 64 == 0, "64-byte records");

inline int vnni(int k, int n) { return (k >> 1) * 32 + n * 2 + (k & 1); }   // B[k][n] of a 16x16 f16 DPAS operand

// Launch-wide parameters (device pointers; plain struct copied into each kernel).
struct Params {
    const half* qkvz; long long qkvz_stride;   // [tokens, >= CH + HV*V] row stride (elements)
    const half* ba; long long ba_stride;       // [tokens, 2 HV]: b | a
    const int* token_indx;                     // packed row -> token row (nullptr: identity)
    const half* convw; const half* convb;      // [CH, 4], [CH] or nullptr
    const float* A_log; const half* dt_bias;   // [HV]
    const int* cu; const int* pos0; const int* src;                        // [B+1], [B], [B]
    const int* ck_cu; const int* ck_pos; const int* ck_dst; const int* ck_kind;   // [B+1], [n], [n], [n]
    float* state; long long st_slot;           // [S, HV, V, K], each slot contiguous
    half* window; long long win_slot;          // conv_state [S, >= 3, CH] (rows CH apart)
    half* tail; long long tail_slot;           // [S, 15, RAW] or nullptr (then every checkpoint is exact)
    int* tail_count;                           // [S] or nullptr (= 0)
    half* out;                                 // [tokens, HV, V]
    half* zout;                                // [tokens, HV, V] z output when GDNC_ZFUSE (else nullptr)
    int* slot_start; int* piece_cu; int* pieces; half* sqk; half* srec;   // scratch
    int B, NK, NV, CH, RAW, max_pieces;
    float eps;
};

struct Slot { int lo, hi, pos0, start, end, src, ck_lo, ck_hi; };
inline Slot slot_of(const Params& p, int s) {
    Slot r;
    r.lo = p.cu[s]; r.hi = p.cu[s + 1]; r.pos0 = p.pos0[s]; r.start = p.slot_start[s];
    r.end = r.pos0 + (r.hi - r.lo); r.src = p.src[s]; r.ck_lo = p.ck_cu[s]; r.ck_hi = p.ck_cu[s + 1];
    return r;
}
inline int ck_kind_of(const Params& p, int k) { return p.tail ? p.ck_kind[k] : 0; }
// The absolute row a checkpoint's state is taken at (the CUDA port's checkpoint_base).
inline int checkpoint_base(const Slot& s, int pos, int kind) {
    if (kind == 0) return pos;
    const int a = (pos / PIECE) * PIECE;
    return a > s.start ? a : s.start;
}
// Raw input column `col` (< RAW: q|k|v channels, then a, then b) of absolute row x (>= start - 3): new rows from
// qkvz / ba, replayed rows from the source slot's tail, earlier rows from the source slot's window.
inline half raw_at(const Params& p, const Slot& s, int x, int col) {
    if (x >= s.pos0) {
        const int row = s.lo + x - s.pos0;
        const size_t tok = (size_t)(p.token_indx ? p.token_indx[row] : row);
        if (col < p.CH) return p.qkvz[tok * p.qkvz_stride + col];
        if (col < p.CH + p.NV) return p.ba[tok * p.ba_stride + p.NV + (col - p.CH)];   // a
        return p.ba[tok * p.ba_stride + (col - p.CH - p.NV)];                          // b
    }
    if (s.src < 0) return (half)0.0f;
    if (x >= s.start) return p.tail[(size_t)s.src * p.tail_slot + (size_t)(x - s.start) * p.RAW + col];
    if (col >= p.CH) return (half)0.0f;
    return p.window[(size_t)s.src * p.win_slot + (size_t)(CONVW - 1 - (s.start - x)) * p.CH + col];
}

// ---------------------------------------------------------------------------------------------------------------
// GdnPlan (one work-item): slot starts (pos0 - tail rows of src) and the piece table: cuts at every absolute
// multiple of 16 and at every exact checkpoint.
// a slot whose exact checkpoints all lie on the 16-row grid (or outside it) cuts only at multiples of 16
inline bool slot_regular(const Params& p, const Slot& s) {
    for (int k = s.ck_lo; k < s.ck_hi; ++k) {
        const int pos = p.ck_pos[k];
        if (ck_kind_of(p, k) == 0 && pos > s.start && pos < s.end && pos % PIECE != 0) return false;
    }
    return true;
}
inline int regular_pieces(const Slot& s) {
    return s.end > s.start ? (s.end - 1) / PIECE - s.start / PIECE + 1 : 0;
}
struct GdnPlan {
    Params p;
    void operator()() const {
        int n = 0;
        for (int b = 0; b < p.B; ++b) {
            const int src = p.src[b];
            p.slot_start[b] = p.pos0[b] - (src >= 0 && p.tail_count && p.tail ? p.tail_count[src] : 0);
            p.piece_cu[b] = n < p.max_pieces ? n : p.max_pieces;
            const Slot s = slot_of(p, b);
#if GDNC_PLANPAR
            if (slot_regular(p, s)) { n += regular_pieces(s); continue; }   // filled by GdnPlanFill
#endif
            for (int cur = s.start; cur < s.end;) {
                int next = (cur / PIECE + 1) * PIECE;
                if (next > s.end) next = s.end;
                for (int k = s.ck_lo; k < s.ck_hi; ++k) {
                    const int pos = p.ck_pos[k];
                    if (ck_kind_of(p, k) == 0 && pos > cur && pos < next) next = pos;
                }
                if (n < p.max_pieces) { int* d = p.pieces + (size_t)n * PD; d[0] = b; d[1] = cur; d[2] = next - cur; d[3] = 0; }
                ++n;
                cur = next;
            }
        }
        p.piece_cu[p.B] = n < p.max_pieces ? n : p.max_pieces;
    }
};
// pieces of regular slots, one work-item each (same entries the sequential loop writes)
struct GdnPlanFill {
    Params p;
    void operator()(sycl::id<1> id) const {
        const int i = id[0];
        if (i >= p.piece_cu[p.B]) return;
        int b = 0;
        while (b < p.B - 1 && p.piece_cu[b + 1] <= i) ++b;
        const Slot s = slot_of(p, b);
        if (!slot_regular(p, s)) return;
        const int j = i - p.piece_cu[b];
        const int cur = j == 0 ? s.start : (s.start / PIECE + j) * PIECE;
        int next = (cur / PIECE + 1) * PIECE;
        if (next > s.end) next = s.end;
        int* d = p.pieces + (size_t)i * PD; d[0] = b; d[1] = cur; d[2] = next - cur; d[3] = 0;
    }
};

// ---------------------------------------------------------------------------------------------------------------
template <int GROUP>
struct GdnPrepare {
    static constexpr int CHL = (2 + GROUP) * W;   // q, k and the key head's GROUP value heads
    static constexpr int XSS = (GDNC_VHALF ? 2 * W : CHL) + GDNC_SLMPAD;   // xs row stride (pad: 640 = 20 x 32 banks)
    static constexpr int VHS = GROUP * W;                                  // vh row stride (halves)
    static constexpr int SLM_FLOATS = PIECE * XSS + 2 * PIECE * (PIECE + 1) + 2 * GROUP * PIECE + GROUP * PIECE * (PIECE + 1) +
                                      (GDNC_VHALF ? PIECE * VHS / 2 : 0);
    Params p;
    sycl::local_accessor<float, 1> slm;

    [[sycl::reqd_sub_group_size(16)]] void operator()(sycl::nd_item<2> it) const {
        const int pc = it.get_group(0), kh = it.get_group(1);
        if (pc >= p.piece_cu[p.B]) return;
        const int* pd = p.pieces + (size_t)pc * PD;
        const Slot s = slot_of(p, pd[0]);
        const int first = pd[1], n = pd[2];   // absolute first row, rows
        const int NK = p.NK, NV = p.NV;
        float* xs = &slm[0];                       // [16][CHL] conv + SiLU (+ L2 norm for q, k)
        float* kk = xs + PIECE * XSS;              // [16][17] K K^T
        float* qg = kk + PIECE * (PIECE + 1);      // [16][17] Q K^T
        float* cum = qg + PIECE * (PIECE + 1);     // [GROUP][16] log decay -> cumulative
        float* bet = cum + GROUP * PIECE;          // [GROUP][16] beta
        float* sys = bet + GROUP * PIECE;          // [GROUP][16][17] A
        [[maybe_unused]] half* vh = reinterpret_cast<half*>(sys + GROUP * PIECE * (PIECE + 1));   // [16][GROUP*W] v (VHALF)
        const int lid = it.get_local_id(1);
        auto sg = it.get_sub_group();
        const int sgid = sg.get_group_linear_id(), lane = sg.get_local_linear_id();

        // conv1d + SiLU: one channel per item, sliding over the piece's rows (conv inputs before the piece: the
        // slot's earlier raw rows, its replayed tail or the source window)
#if GDNC_CONV2
        {
            // rows first-3 .. first+15: every source (new rows, replayed tail, source window) is indexed by channel the
            // same way, so one pointer per row; nullptr = zero (rows past the piece, or no source)
            const half* rp[CONVW - 1 + PIECE];
#pragma unroll
            for (int i = 0; i < CONVW - 1 + PIECE; ++i) {
                const int x = first - (CONVW - 1) + i;
                const half* q = nullptr;
                if (i < CONVW - 1 || i - (CONVW - 1) < n) {
                    if (x >= s.pos0) {
                        const int row = s.lo + x - s.pos0;
                        q = p.qkvz + (size_t)(p.token_indx ? p.token_indx[row] : row) * p.qkvz_stride;
                    } else if (s.src >= 0) {
                        q = x >= s.start ? p.tail + (size_t)s.src * p.tail_slot + (size_t)(x - s.start) * p.RAW
                                         : p.window + (size_t)s.src * p.win_slot + (size_t)(CONVW - 1 - (s.start - x)) * p.CH;
                    }
                }
                rp[i] = q;
            }
#if GDNC_CONV_VEC == 4
            for (int cc = lid; cc < (PREP_ON(1) ? CHL / 4 : 0); cc += PREP_WG) {
                const int c = 4 * cc, part = c / W, col = c % W;   // a quad never straddles a part (W % 4 == 0)
                const int ch = part == 0 ? kh * W + col : part == 1 ? (NK + kh) * W + col : (2 * NK + kh * GROUP + (part - 2)) * W + col;
                float w[4][CONVW], bq[4];
#pragma unroll
                for (int e = 0; e < 4; ++e) {
#pragma unroll
                    for (int tap = 0; tap < CONVW; ++tap) w[e][tap] = (float)p.convw[(size_t)(ch + e) * CONVW + tap];
                    bq[e] = p.convb ? (float)p.convb[ch + e] : 0.0f;
                }
                float x[4][CONVW - 1 + PIECE];
#pragma unroll
                for (int i = 0; i < CONVW - 1 + PIECE; ++i) {
                    if (rp[i]) {
                        const sycl::vec<half, 4> v = *reinterpret_cast<const sycl::vec<half, 4>*>(rp[i] + ch);
#pragma unroll
                        for (int e = 0; e < 4; ++e) x[e][i] = (float)v[e];
                    } else {
#pragma unroll
                        for (int e = 0; e < 4; ++e) x[e][i] = 0.0f;
                    }
                }
#pragma unroll
                for (int t = 0; t < PIECE; ++t) {
#pragma unroll
                    for (int e = 0; e < 4; ++e) {
                        float v = 0.0f;
#pragma unroll
                        for (int tap = 0; tap < CONVW; ++tap) v += x[e][t + tap] * w[e][tap];
                        v += bq[e];
                        v = v / (1.0f + sycl::exp(-v));
#if GDNC_VHALF
                        if (part >= 2) { vh[t * VHS + c - 2 * W + e] = (half)(t < n ? v : 0.0f); continue; }
#endif
                        xs[t * XSS + c + e] = t < n ? v : 0.0f;
                    }
                }
            }
        }
#else
            for (int cc = lid; cc < (PREP_ON(1) ? CHL / 2 : 0); cc += PREP_WG) {
                const int c = 2 * cc, part = c / W, col = c % W;   // a pair never straddles a part (W even)
                const int ch = part == 0 ? kh * W + col : part == 1 ? (NK + kh) * W + col : (2 * NK + kh * GROUP + (part - 2)) * W + col;
                float w0[CONVW], w1[CONVW];
#pragma unroll
                for (int tap = 0; tap < CONVW; ++tap) {
                    w0[tap] = (float)p.convw[(size_t)ch * CONVW + tap];
                    w1[tap] = (float)p.convw[(size_t)(ch + 1) * CONVW + tap];
                }
                const float b0 = p.convb ? (float)p.convb[ch] : 0.0f, b1 = p.convb ? (float)p.convb[ch + 1] : 0.0f;
                float x0[CONVW - 1 + PIECE], x1[CONVW - 1 + PIECE];
#pragma unroll
                for (int i = 0; i < CONVW - 1 + PIECE; ++i) {
                    if (rp[i]) {
                        const sycl::vec<half, 2> v = *reinterpret_cast<const sycl::vec<half, 2>*>(rp[i] + ch);
                        x0[i] = (float)v[0]; x1[i] = (float)v[1];
                    } else {
                        x0[i] = 0.0f; x1[i] = 0.0f;
                    }
                }
#pragma unroll
                for (int t = 0; t < PIECE; ++t) {
                    float v0 = 0.0f, v1 = 0.0f;
#pragma unroll
                    for (int tap = 0; tap < CONVW; ++tap) { v0 += x0[t + tap] * w0[tap]; v1 += x1[t + tap] * w1[tap]; }
                    v0 += b0; v1 += b1;
                    v0 = v0 / (1.0f + sycl::exp(-v0));
                    v1 = v1 / (1.0f + sycl::exp(-v1));
#if GDNC_VHALF
                    if (part >= 2) {
                        vh[t * VHS + c - 2 * W] = (half)(t < n ? v0 : 0.0f);
                        vh[t * VHS + c - 2 * W + 1] = (half)(t < n ? v1 : 0.0f);
                        continue;
                    }
#endif
                    xs[t * XSS + c] = t < n ? v0 : 0.0f;
                    xs[t * XSS + c + 1] = t < n ? v1 : 0.0f;
                }
            }
        }
#endif
#else
        for (int c = lid; c < (PREP_ON(1) ? CHL : 0); c += PREP_WG) {
            const int part = c / W, col = c % W;
            const int ch = part == 0 ? kh * W + col : part == 1 ? (NK + kh) * W + col : (2 * NK + kh * GROUP + (part - 2)) * W + col;
            float w[CONVW];
#pragma unroll
            for (int tap = 0; tap < CONVW; ++tap) w[tap] = (float)p.convw[(size_t)ch * CONVW + tap];
            const float bias = p.convb ? (float)p.convb[ch] : 0.0f;
            float x[CONVW - 1 + PIECE];
#pragma unroll
            for (int i = 0; i < CONVW - 1 + PIECE; ++i)
                x[i] = (i < CONVW - 1 || i - (CONVW - 1) < n) ? (float)raw_at(p, s, first - (CONVW - 1) + i, ch) : 0.0f;
#pragma unroll
            for (int t = 0; t < PIECE; ++t) {
                float v = 0.0f;
#pragma unroll
                for (int tap = 0; tap < CONVW; ++tap) v += x[t + tap] * w[tap];
                v += bias;
                v = v / (1.0f + sycl::exp(-v));
                xs[t * XSS + c] = t < n ? v : 0.0f;
            }
        }
#endif
        sycl::group_barrier(it.get_group());

        // L2 norms: sub-group t owns row t (8 columns per lane)
        if (PREP_ON(2)) {
            const int t = sgid;
            float qq = 0.0f, kq = 0.0f;
#pragma unroll
            for (int j = 0; j < W / 16; ++j) {
                const float qv = xs[t * XSS + lane * (W / 16) + j], kv = xs[t * XSS + W + lane * (W / 16) + j];
                qq += qv * qv; kq += kv * kv;
            }
            qq = sycl::reduce_over_group(sg, qq, sycl::plus<float>());
            kq = sycl::reduce_over_group(sg, kq, sycl::plus<float>());
            const float qi = t < n ? sycl::rsqrt(qq + p.eps) * sycl::rsqrt((float)W) : 0.0f;
            const float ki = t < n ? sycl::rsqrt(kq + p.eps) : 0.0f;
#pragma unroll
            for (int j = 0; j < W / 16; ++j) {
                xs[t * XSS + lane * (W / 16) + j] *= qi;
                xs[t * XSS + W + lane * (W / 16) + j] *= ki;
            }
        }
        // gates
        if (PREP_ON(4) && lid < GROUP * PIECE) {
            const int m = lid / PIECE, t = lid % PIECE, hv = kh * GROUP + m;
            float ld = 0.0f, be = 0.0f;
            if (t < n) {
                const float av = (float)raw_at(p, s, first + t, p.CH + hv), bv = (float)raw_at(p, s, first + t, p.CH + NV + hv);
                be = 1.0f / (1.0f + sycl::exp(-bv));
                const float sh = av + (float)p.dt_bias[hv];
                const float sp = sycl::fmax(sh, 0.0f) + sycl::log(1.0f + sycl::exp(-sycl::fabs(sh)));
                ld = -sycl::exp(p.A_log[hv]) * sp;
            }
            cum[m * PIECE + t] = ld;
            bet[m * PIECE + t] = be;
        }
        sycl::group_barrier(it.get_group());

        // f16 operands of the scan: q | k_high | k_low in both packings, v transposed per value head
        half* qk = p.sqk + (size_t)(pc * NK + kh) * QK_BLOCK;
#if GDNC_VNNI2D
        for (int idx = lid; idx < (PREP_ON(8) ? PIECE * W : 0); idx += PREP_WG) {   // row-major tiles
            const int t = idx / W, col = idx % W;
            const float kv = xs[t * XSS + W + col];
            const half k16 = (half)kv;
            qk[QK2_Q + idx] = (half)xs[t * XSS + col];
            qk[QK2_KH + idx] = k16;
            qk[QK2_KL + idx] = (half)(kv - (float)k16);
        }
#else
        for (int idx = lid; idx < (PREP_ON(8) ? 8 * 256 : 0); idx += PREP_WG) {
            const int sb = idx / 256, e = idx % 256;
            {   // T packing: k = column, n = row
                const int c = (e / 32) * 2 + (e & 1), t = (e % 32) / 2, col = 16 * sb + c;
                const float kv = xs[t * XSS + W + col];
                const half k16 = (half)kv;
                qk[QK_QT + idx] = (half)xs[t * XSS + col];
                qk[QK_KT + idx] = k16;
                qk[QK_KLT + idx] = (half)(kv - (float)k16);
            }
            {   // V packing: k = row, n = column
                const int t = (e / 32) * 2 + (e & 1), c = (e % 32) / 2, col = 16 * sb + c;
                const float kv = xs[t * XSS + W + col];
                const half k16 = (half)kv;
                qk[QK_KV + idx] = k16;
                qk[QK_KLV + idx] = (half)(kv - (float)k16);
            }
        }
#endif
#pragma unroll
        for (int m = 0; m < GROUP; ++m) {
            half* rec = p.srec + (size_t)(pc * NV + kh * GROUP + m) * REC;
            for (int idx = lid; idx < (PREP_ON(8) ? W * PIECE : 0); idx += PREP_WG) {
                const int r = idx / PIECE, t = idx % PIECE;
#if GDNC_VHALF
                rec[REC_VT + idx] = vh[t * VHS + m * W + r];
#else
                rec[REC_VT + idx] = (half)xs[t * XSS + (2 + m) * W + r];
#endif
            }
        }
#if GDNC_ZFUSE
        if (p.zout) {   // z of this piece's NEW rows (replayed tail rows have no outputs) for the key head's GROUP value heads
            const int zc = GROUP * W / 2;   // dwords per row
            for (int idx = lid; idx < PIECE * zc; idx += PREP_WG) {
                const int t = idx / zc, c2 = idx % zc, x = first + t;
                if (t >= n || x < s.pos0) continue;
                const int row = s.lo + x - s.pos0;
                const size_t tok = (size_t)(p.token_indx ? p.token_indx[row] : row);
                const int col = 2 * c2 + kh * GROUP * W;   // within [HV * V]
                *reinterpret_cast<uint32_t*>(p.zout + tok * (size_t)(NV * W) + col) =
                    *reinterpret_cast<const uint32_t*>(p.qkvz + tok * p.qkvz_stride + p.CH + col);
            }
        }
#endif
        // Gram matrices in F32, one (t, s) pair per item; cumulative decays
        {
            const int t = lid / PIECE, s2 = lid % PIECE;
            float a0 = 0.0f, a1 = 0.0f, q0 = 0.0f, q1 = 0.0f;
            for (int c = 0; c < (PREP_ON(16) ? W : 0); c += 2) {
                a0 += xs[t * XSS + W + c] * xs[s2 * XSS + W + c];
                a1 += xs[t * XSS + W + c + 1] * xs[s2 * XSS + W + c + 1];
                q0 += xs[t * XSS + c] * xs[s2 * XSS + W + c];
                q1 += xs[t * XSS + c + 1] * xs[s2 * XSS + W + c + 1];
            }
            kk[t * (PIECE + 1) + s2] = a0 + a1;
            qg[t * (PIECE + 1) + s2] = q0 + q1;
        }
        if (lid < GROUP) {
            float sum = 0.0f;
            for (int t = 0; t < PIECE; ++t) { sum += cum[lid * PIECE + t]; cum[lid * PIECE + t] = sum; }
        }
        sycl::group_barrier(it.get_group());

        // A to SLM; D (VNNI) and the gates to the records
        {
            const int t = lid / PIECE, s2 = lid % PIECE;
#pragma unroll
            for (int m = 0; m < (PREP_ON(16) ? GROUP : 0); ++m) {
                const float* G = cum + m * PIECE;
                const float decay = s2 <= t ? sycl::exp(G[t] - G[s2]) : 0.0f;
                sys[(m * PIECE + t) * (PIECE + 1) + s2] = s2 < t ? bet[m * PIECE + t] * decay * kk[t * (PIECE + 1) + s2] : 0.0f;
                half* rec = p.srec + (size_t)(pc * NV + kh * GROUP + m) * REC;
                rec[REC_D + vnni(s2, t)] = (half)(t < n ? decay * qg[t * (PIECE + 1) + s2] : 0.0f);   // B[k = s][n = t] = D[t][s]
                if (lid < PIECE) {
                    float* g = reinterpret_cast<float*>(rec + REC_G);
                    const float last = G[PIECE - 1];
                    g[s2] = sycl::exp(G[s2]);
                    g[PIECE + s2] = sycl::exp(last - G[s2]);
                    g[2 * PIECE + s2] = s2 == 0 ? sycl::exp(last) : 0.0f;
                }
            }
        }
        sycl::group_barrier(it.get_group());

        // Tb = (I + A)^-1 diag(beta): sub-group m, lane j owns column j (forward substitution)
        if (PREP_ON(16) && sgid < GROUP) {
            const int m = sgid, j = lane;
            const float* Am = sys + m * PIECE * (PIECE + 1);
            float x[PIECE];
#pragma unroll
            for (int i = 0; i < PIECE; ++i) {
                float v = i == j ? 1.0f : 0.0f;
#pragma unroll
                for (int k2 = 0; k2 < i; ++k2) v -= Am[i * (PIECE + 1) + k2] * x[k2];
                x[i] = v;
            }
            const float bj = bet[m * PIECE + j];
            half* rec = p.srec + (size_t)(pc * NV + kh * GROUP + m) * REC;
#pragma unroll
            for (int i = 0; i < PIECE; ++i) rec[REC_TB + vnni(j, i)] = (half)(x[i] * bj);   // B[k = j][n = i] = Tb[i][j]
        }
    }
};

// ---------------------------------------------------------------------------------------------------------------
// GdnScan: thread = 8 state rows (row0 .. row0 + 7) of one value head of one slot; S as 8 tiles [8 rows][16 cols].
namespace xmx = sycl::ext::intel::esimd::xmx;

struct GdnScan {
    Params p;

    // 256-GRF mode (4 threads per EU): S alone is 64 GRFs
    auto get(sycl::ext::oneapi::experimental::properties_tag) const {
        return sycl::ext::oneapi::experimental::properties{sycl::ext::intel::experimental::grf_size<256>};
    }
    static inline void store_rows(sycl::ext::intel::esimd::simd<float, ROWS_T * W>& S, float* st) SYCL_ESIMD_FUNCTION {
        using namespace sycl::ext::intel::esimd;
#pragma unroll
        for (int r = 0; r < ROWS_T; ++r) {
            simd<float, W> row;
#pragma unroll
            for (int sb = 0; sb < W / 16; ++sb) row.template select<16, 1>(sb * 16) = S.template select<16, 1>((sb * ROWS_T + r) * 16);
            block_store<float, W>(st + r * W, row);
        }
    }
    // the state to every checkpoint of the slot whose base row is `boundary`
    inline void publish(sycl::ext::intel::esimd::simd<float, ROWS_T * W>& S, const Slot& s, int boundary, size_t srow) const SYCL_ESIMD_FUNCTION {
        for (int k = s.ck_lo; k < s.ck_hi; ++k) {
            if (checkpoint_base(s, p.ck_pos[k], ck_kind_of(p, k)) != boundary) continue;
            const int dst = p.ck_dst[k];
            if (dst >= 0) store_rows(S, p.state + (size_t)dst * p.st_slot + srow);
        }
    }

    void operator()(sycl::nd_item<1> it) const SYCL_ESIMD_KERNEL {
        using namespace sycl::ext::intel::esimd;
        const int gid = it.get_group(0), tid = it.get_local_id(0);
        const int NV = p.NV, part = gid % SCAN_SPLIT, hv = (gid / SCAN_SPLIT) % NV, b = gid / (SCAN_SPLIT * NV);
        const Slot s = slot_of(p, b);
        const int GROUP = NV / p.NK, kh = hv / GROUP, row0 = (part * SCAN_WG + tid) * ROWS_T;
#if GDNC_DIAG_NOSCAN
        if (gid >= 0) return;
#endif
        const size_t srow = ((size_t)hv * W + row0) * W;   // this thread's rows within a state slot

        simd<float, ROWS_T * W> S;   // tile sb (key cols 16sb .. 16sb+15) at [sb][8 rows][16]
        if (s.src >= 0) {
            const float* st = p.state + (size_t)s.src * p.st_slot + srow;
#pragma unroll
            for (int r = 0; r < ROWS_T; ++r) {
                simd<float, W> row = block_load<float, W>(st + r * W);
#pragma unroll
                for (int sb = 0; sb < W / 16; ++sb) S.template select<16, 1>((sb * ROWS_T + r) * 16) = row.template select<16, 1>(sb * 16);
            }
        } else {
            S = 0.0f;
        }
        publish(S, s, s.start, srow);

        const int p0 = p.piece_cu[b], p1 = p.piece_cu[b + 1];
        for (int pc = p0; pc < p1; ++pc) {
            const int* pd = p.pieces + (size_t)pc * PD;
            const int first = pd[1], n = pd[2];
            const half* qk = p.sqk + (size_t)(pc * p.NK + kh) * QK_BLOCK;
            const half* rec = p.srec + (size_t)(pc * NV + hv) * REC;
            const float* gp = reinterpret_cast<const float*>(rec + REC_G);
#if GDNC_PF
            if (pc + GDNC_PF_DIST < p1) {   // piece pc+DIST: this thread's share of its q/k block and record
                namespace xe = sycl::ext::intel::experimental::esimd;
                constexpr auto L1H = GDNC_PF_L1 ? xe::cache_hint::cached : xe::cache_hint::uncached;
                const int np = pc + GDNC_PF_DIST;
                const uint32_t* nq = reinterpret_cast<const uint32_t*>(p.sqk + (size_t)(np * p.NK + kh) * QK_BLOCK);
                const uint32_t* nr = reinterpret_cast<const uint32_t*>(p.srec + (size_t)(np * NV + hv) * REC);
                constexpr int RL = REC * 2 / 64;   // the record: 64-byte lines (not a multiple of 256 B)
#if GDNC_PF_QK256
                static_assert(QK_BLOCK * 2 % 256 == 0, "q/k block in 256-byte lines");
                for (int l = tid; l < (GDNC_VNNI2D ? QK2_USED : QK_BLOCK) * 2 / 256; l += SCAN_WG)
                    xe::lsc_prefetch<uint32_t, 64, xe::lsc_data_size::default_size, L1H, xe::cache_hint::cached>(nq + 64 * l);
#else
                for (int l = tid; l < (GDNC_VNNI2D ? QK2_USED : QK_BLOCK) * 2 / 64; l += SCAN_WG)
                    xe::lsc_prefetch<uint32_t, 16, xe::lsc_data_size::default_size, L1H, xe::cache_hint::cached>(nq + 16 * l);
#endif
                for (int l = tid; l < RL; l += SCAN_WG)
                    xe::lsc_prefetch<uint32_t, 16, xe::lsc_data_size::default_size, L1H, xe::cache_hint::cached>(nr + 16 * l);
            }
#endif
            simd<float, 16> gam = block_load<float, 16>(gp);
            simd<float, 16> tail = block_load<float, 16>(gp + 16);
            simd<float, 16> lastv = block_load<float, 16>(gp + 32);
            const float last = lastv[0];

            // X^T = S K^T (S high/low x K high, S high x K low), Y^T = S Q^T: [8 rows][16 piece rows]
            simd<float, 128> rm = 0.0f, rml = 0.0f, rkl = 0.0f, o = 0.0f;
#pragma unroll GDNC_UNROLL
            for (int sb = 0; sb < W / 16; ++sb) {
                simd<float, 128> sf = S.template select<128, 1>(sb * 128);
                simd<fp16, 128> ah = convert<fp16>(sf);
                simd<fp16, 128> al = convert<fp16>(sf - convert<float>(ah));
#if GDNC_VNNI2D
                namespace xe2 = sycl::ext::intel::experimental::esimd;
                constexpr unsigned SW = W * 2 - 1, SH = PIECE - 1, SP = W * 2 - 1;   // [16][128] f16 tile: bytes-1, rows-1, pitch-1
                simd<uint32_t, 128> bk32 = xe2::lsc_load_2d<uint32_t, 8, PIECE, 1, true, false, xe2::cache_hint::cached, xe2::cache_hint::cached>(
                    reinterpret_cast<const uint32_t*>(qk + QK2_KH), SW, SH, SP, sb * 8, 0);
                simd<fp16, 256> bk = bk32.template bit_cast_view<fp16>().read();
                simd<uint32_t, 128> bq32 = xe2::lsc_load_2d<uint32_t, 8, PIECE, 1, true, false, xe2::cache_hint::cached, xe2::cache_hint::cached>(
                    reinterpret_cast<const uint32_t*>(qk + QK2_Q), SW, SH, SP, sb * 8, 0);
                simd<fp16, 256> bq = bq32.template bit_cast_view<fp16>().read();
                simd<uint32_t, 128> bl32 = xe2::lsc_load_2d<uint32_t, 8, PIECE, 1, true, false, xe2::cache_hint::cached, xe2::cache_hint::cached>(
                    reinterpret_cast<const uint32_t*>(qk + QK2_KL), SW, SH, SP, sb * 8, 0);
                simd<fp16, 256> bl = bl32.template bit_cast_view<fp16>().read();
#else
                simd<fp16, 256> bk = block_load<fp16, 256>(qk + QK_KT + sb * 256);
                simd<fp16, 256> bq = block_load<fp16, 256>(qk + QK_QT + sb * 256);
                simd<fp16, 256> bl = block_load<fp16, 256>(qk + QK_KLT + sb * 256);
#endif
                rm = xmx::dpas<8, 8, float, float, fp16, fp16>(rm, bk, ah);
                rml = xmx::dpas<8, 8, float, float, fp16, fp16>(rml, bk, al);
                rkl = xmx::dpas<8, 8, float, float, fp16, fp16>(rkl, bl, ah);
                o = xmx::dpas<8, 8, float, float, fp16, fp16>(o, bq, ah);
            }
            rm += rml + rkl;
            // Z^T = V^T - X^T diag(gamma); U^T = Z^T Tb^T; O^T = Y^T diag(gamma) + U^T D^T
            simd<float, 128> vt = convert<float>(simd<fp16, 128>(block_load<fp16, 128>(rec + REC_VT + row0 * PIECE)));
            simd<float, 128> grep = gam.template replicate<ROWS_T>();
            simd<float, 128> z = vt - rm * grep;
            o *= grep;
            simd<fp16, 128> zh = convert<fp16>(z);
            simd<fp16, 128> zl = convert<fp16>(z - convert<float>(zh));
            simd<fp16, 256> tb = block_load<fp16, 256>(rec + REC_TB);
            simd<float, 128> u = 0.0f;
            u = xmx::dpas<8, 8, float, float, fp16, fp16>(u, tb, zh);
            u = xmx::dpas<8, 8, float, float, fp16, fp16>(u, tb, zl);
            simd<fp16, 256> db = block_load<fp16, 256>(rec + REC_D);
            o = xmx::dpas<8, 8, float, float, fp16, fp16>(o, db, simd<fp16, 128>(convert<fp16>(u)));
            // outputs of new rows (replayed tail rows have none): row t -> 8 contiguous halves at (token, hv, row0)
            for (int t = 0; t < n; ++t) {
                const int x = first + t;
                if (x < s.pos0) continue;
                const int row = s.lo + x - s.pos0;
                const size_t tok = (size_t)(p.token_indx ? p.token_indx[row] : row);
                simd<fp16, ROWS_T> ov = convert<fp16>(simd<float, ROWS_T>(o.template select<ROWS_T, 16>(t)));
                block_store<fp16, ROWS_T>(p.out + (tok * NV + hv) * W + row0, ov);
            }
            // S <- last S + (U^T diag(tail)) K
            simd<float, 128> wv = u * tail.template replicate<ROWS_T>();
            simd<fp16, 128> wh = convert<fp16>(wv);
            simd<fp16, 128> wl = convert<fp16>(wv - convert<float>(wh));
#pragma unroll GDNC_UNROLL
            for (int sb = 0; sb < W / 16; ++sb) {
                simd<float, 128> acc = S.template select<128, 1>(sb * 128);
                acc *= last;
#if GDNC_VNNI2D
                namespace xe3 = sycl::ext::intel::experimental::esimd;
                constexpr unsigned SW3 = W * 2 - 1, SH3 = PIECE - 1, SP3 = W * 2 - 1;
                simd<fp16, 256> bk = xe3::lsc_load_2d<fp16, 16, PIECE, 1, false, true, xe3::cache_hint::cached, xe3::cache_hint::cached>(
                    qk + QK2_KH, SW3, SH3, SP3, sb * 16, 0);
                simd<fp16, 256> bl = xe3::lsc_load_2d<fp16, 16, PIECE, 1, false, true, xe3::cache_hint::cached, xe3::cache_hint::cached>(
                    qk + QK2_KL, SW3, SH3, SP3, sb * 16, 0);
#else
                simd<fp16, 256> bk = block_load<fp16, 256>(qk + QK_KV + sb * 256);
                simd<fp16, 256> bl = block_load<fp16, 256>(qk + QK_KLV + sb * 256);
#endif
                acc = xmx::dpas<8, 8, float, float, fp16, fp16>(acc, bk, wh);
                acc = xmx::dpas<8, 8, float, float, fp16, fp16>(acc, bk, wl);
                acc = xmx::dpas<8, 8, float, float, fp16, fp16>(acc, bl, wh);
                S.template select<128, 1>(sb * 128) = acc;
            }
            publish(S, s, first + n, srow);
        }
    }
};

// ---------------------------------------------------------------------------------------------------------------
// Windows, tails, tail counts per checkpoint: window = raw rows base-3 .. base-1 (CH channels), tail = raw rows
// base .. pos-1 (RAW columns, aligned kind only). Gathered for ALL checkpoints into tmp first, then scattered, so
// no checkpoint reads a window/tail another one (also of another slot, or in place) already overwrote.
// tmp layout per checkpoint: [3][CH] window, [15][RAW] tail.
inline int ck_slot_of(const Params& p, int k) {
    int b = 0;
    while (b < p.B - 1 && p.ck_cu[b + 1] <= k) ++b;
    return b;
}
struct GdnPubGather {
    Params p; half* tmp;
    void operator()(sycl::id<2> id) const {
        const int k = id[0], col = id[1];   // col < RAW
        const Slot s = slot_of(p, ck_slot_of(p, k));
        const int kind = ck_kind_of(p, k), pos = p.ck_pos[k], base = checkpoint_base(s, pos, kind);
        const int tail_rows = kind == 0 ? 0 : pos - base;
        half* t = tmp + (size_t)k * ((CONVW - 1) * p.CH + TAIL * p.RAW);
        if (col < p.CH) {
#pragma unroll
            for (int r = 0; r < CONVW - 1; ++r) t[(size_t)r * p.CH + col] = raw_at(p, s, base - (CONVW - 1) + r, col);
        }
        half* tt = t + (CONVW - 1) * p.CH;
        for (int r = 0; r < tail_rows && r < TAIL; ++r) tt[(size_t)r * p.RAW + col] = raw_at(p, s, base + r, col);
    }
};
struct GdnPubScatter {
    Params p; const half* tmp;
    void operator()(sycl::id<2> id) const {
        const int k = id[0], col = id[1];
        const Slot s = slot_of(p, ck_slot_of(p, k));
        const int kind = ck_kind_of(p, k), pos = p.ck_pos[k], base = checkpoint_base(s, pos, kind), dst = p.ck_dst[k];
        if (dst < 0) return;
        const int tail_rows = kind == 0 ? 0 : pos - base;
        const half* t = tmp + (size_t)k * ((CONVW - 1) * p.CH + TAIL * p.RAW);
        if (col < p.CH) {
#pragma unroll
            for (int r = 0; r < CONVW - 1; ++r) p.window[(size_t)dst * p.win_slot + (size_t)r * p.CH + col] = t[(size_t)r * p.CH + col];
        }
        if (p.tail) {
            const half* tt = t + (CONVW - 1) * p.CH;
            for (int r = 0; r < tail_rows && r < TAIL; ++r) p.tail[(size_t)dst * p.tail_slot + (size_t)r * p.RAW + col] = tt[(size_t)r * p.RAW + col];
        }
        if (col == 0 && p.tail_count) p.tail_count[dst] = tail_rows;
    }
};

struct GdnZCopy {
    const uint32_t* qkvz; uint32_t* z; const int* token_indx; long long qkvz_w2; int z_off2, z_w2;   // in uint32 (2 halves)
    void operator()(sycl::id<1> id) const {
        const size_t idx = id[0];
        const int row = idx / z_w2, c = idx % z_w2;
        const size_t tok = (size_t)(token_indx ? token_indx[row] : row);
        z[tok * z_w2 + c] = qkvz[tok * qkvz_w2 + z_off2 + c];
    }
};

}  // namespace gdnc
