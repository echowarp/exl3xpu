// GDN speculative decoding with one committed state slot per request.
// v2: per (request, v-head, tile of 16*RPS rows): normalized q/k of all T1 tokens and the pending k are staged once in
// SLM; each 16-lane sub-group owns RPS state rows (8 K-elements per lane per row) -> RPS independent reduction chains.
#pragma once
#include <sycl/sycl.hpp>
namespace exl3gdn {
using half = sycl::half;
constexpr int SG = 16, KPL = 8, NSG = 16, KD = SG * KPL, TMAX = 16;
// RC-FUSED-IN (same scheme as EXL3_RC_FUSED_IN of the CUDA port, exl3cuda): FUSED reads q/k/v straight from the conv output
// rows x [T, 2*H*K + HV*V] (row stride xs), computes g = -exp(A_log) * softplus(a + dt_bias) and beta = sigmoid(b) from
// the raw ba rows [T, 2*HV] (b | a, row stride bas), and writes o and z (copied from zsrc rows, stride zss) at row
// oidx[tok] (or tok) of the [*, HV, V] outputs (row strides os, zs): ~15 fewer launches per GDN layer per step.
template <int RPS, bool FUSED = false>
struct Recompute {
    const half* q; const half* k; const half* v; const float* g; const float* beta;
    half* o; float* h; const int* state_idx;
    float* pk; float* pv; float* pg; float* pb; const int* n_pend;   // pending pool [2, S, T1, ...] (read par, write 1-par)
    const int* slot; const int* par; const int* qsl;                 // per request: pool slot, read parity, token start
    const int* bnd_idx; const int* bnd_block;   // per request: store the state after pending token bnd_idx into bnd_block (-1 = none)
    int S, T1, H, HV, V; float scale; long long hs0, hs1, hs2;   // state strides (elements): block, head, row
    sycl::local_accessor<float, 1> slm;
    // FUSED only (after slm so the unfused aggregate initializer stays as it was)
    const half* x = nullptr; long long xs = 0; const half* ba = nullptr; long long bas = 0; const float* alog = nullptr; const half* dtb = nullptr;   // dtb: the layer's fp16 dt_bias as is
    half* zo = nullptr; const half* zsrc = nullptr; long long zss = 0; const int* oidx = nullptr; long long os = 0, zs = 0;   // [3 * TMAX * KD]: q̂, k̂ (this step), pending k̂
    [[sycl::reqd_sub_group_size(SG)]] void operator()(sycl::nd_item<1> it) const {
        auto sg = it.get_sub_group();
        const int lane = sg.get_local_id()[0], sgi = sg.get_group_id()[0], lid = it.get_local_id(0);
        const int ROWS = NSG * RPS, tiles = V / ROWS, gid = it.get_group(0);
        const int tile = gid % tiles, hv = (gid / tiles) % HV, n = gid / (tiles * HV);
        const int hk = hv / (HV / H);
        const int np = n_pend[n];
        const int t0 = qsl[n], Tn = qsl[n + 1] - qsl[n];
        const size_t rb = ((size_t)par[n] * S + slot[n]) * T1, wbase = ((size_t)(1 - par[n]) * S + slot[n]) * T1;
        float* sq = &slm[0]; float* sk = &slm[TMAX * KD]; float* sp = &slm[2 * TMAX * KD];
        // stage: sub-group t normalizes token t's q and k (T1 <= NSG); pending k̂ copied as is
        if (sgi < Tn) {
            const int tok = t0 + sgi;
            const half* qp = FUSED ? x + (size_t)tok * xs + (size_t)hk * KD + lane * KPL : q + ((size_t)tok * H + hk) * KD + lane * KPL;
            const half* kp = FUSED ? x + (size_t)tok * xs + (size_t)(H + hk) * KD + lane * KPL : k + ((size_t)tok * H + hk) * KD + lane * KPL;
            float qv[KPL], kv[KPL], qq = 0.f, kk = 0.f;
#pragma unroll
            for (int j = 0; j < KPL; ++j) { qv[j] = (float)qp[j]; kv[j] = (float)kp[j]; qq += qv[j] * qv[j]; kk += kv[j] * kv[j]; }
            qq = sycl::reduce_over_group(sg, qq, sycl::plus<float>());
            kk = sycl::reduce_over_group(sg, kk, sycl::plus<float>());
            const float qs = scale / sycl::sqrt(qq + 1e-6f), ks = 1.f / sycl::sqrt(kk + 1e-6f);
#pragma unroll
            for (int j = 0; j < KPL; ++j) { sq[sgi * KD + lane * KPL + j] = qv[j] * qs; sk[sgi * KD + lane * KPL + j] = kv[j] * ks; }
            if (tile == 0 && hv % (HV / H) == 0) {
                float* wkp = pk + ((wbase + sgi) * H + hk) * KD + lane * KPL;
#pragma unroll
                for (int j = 0; j < KPL; ++j) wkp[j] = kv[j] * ks;
            }
        }
        for (int i = lid; i < np * KD; i += NSG * SG) sp[i] = pk[((rb + i / KD) * H + hk) * KD + i % KD];
        sycl::group_barrier(it.get_group());
        float s[RPS][KPL];
        float* hp[RPS];
#pragma unroll
        for (int r = 0; r < RPS; ++r) {
            const int row = tile * ROWS + sgi * RPS + r;
            hp[r] = h + (size_t)state_idx[n] * hs0 + (size_t)hv * hs1 + (size_t)row * hs2 + lane * KPL;
#pragma unroll
            for (int j = 0; j < KPL; ++j) s[r][j] = hp[r][j];
        }
        const int row0 = tile * ROWS + sgi * RPS;
        for (int t = 0; t < np; ++t) {
            const size_t ptok = rb + t;
            float kv[KPL];
#pragma unroll
            for (int j = 0; j < KPL; ++j) kv[j] = sp[t * KD + lane * KPL + j];
            const float eg = sycl::exp(pg[ptok * HV + hv]), bb = pb[ptok * HV + hv];
            float d[RPS];
#pragma unroll
            for (int r = 0; r < RPS; ++r) {
                d[r] = 0.f;
#pragma unroll
                for (int j = 0; j < KPL; ++j) { s[r][j] *= eg; d[r] += s[r][j] * kv[j]; }
            }
#pragma unroll
            for (int r = 0; r < RPS; ++r) d[r] = sycl::reduce_over_group(sg, d[r], sycl::plus<float>());
#pragma unroll
            for (int r = 0; r < RPS; ++r) {
                const float u = (pv[(ptok * HV + hv) * V + row0 + r] - d[r]) * bb;
#pragma unroll
                for (int j = 0; j < KPL; ++j) s[r][j] += u * kv[j];
            }
            if (bnd_idx != nullptr && t == bnd_idx[n]) {   // decode-time boundary state for prefix caching
                const int bb_blk = bnd_block[n];
#pragma unroll
                for (int r = 0; r < RPS; ++r) {
                    float* bp = h + (size_t)bb_blk * hs0 + (size_t)hv * hs1 + (size_t)(row0 + r) * hs2 + lane * KPL;
#pragma unroll
                    for (int j = 0; j < KPL; ++j) bp[j] = s[r][j];
                }
            }
        }
#pragma unroll
        for (int r = 0; r < RPS; ++r)
#pragma unroll
            for (int j = 0; j < KPL; ++j) hp[r][j] = s[r][j];
        for (int t = 0; t < Tn; ++t) {
            const int tok = t0 + t;
            const size_t wtok = wbase + t;
            float kv[KPL], qv[KPL];
#pragma unroll
            for (int j = 0; j < KPL; ++j) { kv[j] = sk[t * KD + lane * KPL + j]; qv[j] = sq[t * KD + lane * KPL + j]; }
            float gg, bb;
            if constexpr (FUSED) {
                const float za = (float)ba[(size_t)tok * bas + HV + hv] + (float)dtb[hv];
                const float sp = za > 20.f ? za : sycl::log1p(sycl::exp(za));
                gg = -sycl::exp(alog[hv]) * sp;
                bb = 1.f / (1.f + sycl::exp(-(float)ba[(size_t)tok * bas + hv]));
            } else {
                gg = g[tok * HV + hv]; bb = beta[tok * HV + hv];
            }
            const float eg = sycl::exp(gg);
            const size_t orow = FUSED ? (size_t)(oidx ? oidx[tok] : tok) : (size_t)tok;
            float d[RPS], vr[RPS];
#pragma unroll
            for (int r = 0; r < RPS; ++r) {
                vr[r] = FUSED ? (float)x[(size_t)tok * xs + 2 * (size_t)H * KD + (size_t)hv * V + row0 + r]
                              : (float)v[((size_t)tok * HV + hv) * V + row0 + r];
                d[r] = 0.f;
#pragma unroll
                for (int j = 0; j < KPL; ++j) { s[r][j] *= eg; d[r] += s[r][j] * kv[j]; }
            }
#pragma unroll
            for (int r = 0; r < RPS; ++r) d[r] = sycl::reduce_over_group(sg, d[r], sycl::plus<float>());
            float out[RPS];
#pragma unroll
            for (int r = 0; r < RPS; ++r) {
                const float u = (vr[r] - d[r]) * bb;
                out[r] = 0.f;
#pragma unroll
                for (int j = 0; j < KPL; ++j) { s[r][j] += u * kv[j]; out[r] += s[r][j] * qv[j]; }
            }
#pragma unroll
            for (int r = 0; r < RPS; ++r) out[r] = sycl::reduce_over_group(sg, out[r], sycl::plus<float>());
            if (lane < RPS) {
                // lane r writes row r's output (select without dynamic register indexing)
                float mine = out[0];
#pragma unroll
                for (int r = 1; r < RPS; ++r) if (lane == r) mine = out[r];
                if constexpr (FUSED) {
                    o[orow * os + (size_t)hv * V + row0 + lane] = (half)mine;
                    zo[orow * zs + (size_t)hv * V + row0 + lane] = zsrc[(size_t)tok * zss + (size_t)hv * V + row0 + lane];
                } else {
                    o[((size_t)tok * HV + hv) * V + row0 + lane] = (half)mine;
                }
            }
            if (lane < RPS) {
                float vm = vr[0];
#pragma unroll
                for (int r = 1; r < RPS; ++r) if (lane == r) vm = vr[r];
                pv[(wtok * HV + hv) * V + row0 + lane] = vm;
            }
            if (tile == 0 && sgi == 0 && lane == 0) { pg[wtok * HV + hv] = gg; pb[wtok * HV + hv] = bb; }
        }
    }
};
}  // namespace exl3gdn
