#pragma once
// Fused top-k + top-p logits mask for sampling.
//
// Same result as vLLM's apply_top_k_top_p_pytorch (sort-based) for top_k <= TOPKP_CAP, without sorting the
// vocabulary: one work-group per row
//   1. row max, 2. candidates within [max - D, max] collected in SLM with D adapted until k <= n <= CAP
//      (usually one pass; no histogram atomics), bitonic-sorted descending; the k-th largest candidate is the
//      k-th largest logit,
//   3. top-p over the kept set: softmax over the candidates, keep a token while the probability mass ranked
//      above it is < p (== vLLM's ascending cumsum <= 1 - p test); at least one token is kept,
//   4. logits below the final threshold value are set to -inf in place.
// Ties at a threshold are kept together (the sort-based version splits exact ties arbitrarily).
// If more than TOPKP_CAP candidates tie into the top-k set, top-p is skipped for that row and flags[row] = 1.

#include <sycl/sycl.hpp>

namespace exl3tk {

constexpr int WG = 1024;
constexpr int NB = 2048;          // histogram bins (11 bits)
constexpr int TOPKP_CAP = 2048;   // candidate buffer

static inline uint32_t fkey(float x) {
    uint32_t u = sycl::bit_cast<uint32_t>(x);
    return (u & 0x80000000u) ? ~u : (u | 0x80000000u);
}
static inline float fval(uint32_t k) {
    uint32_t u = (k & 0x80000000u) ? (k & 0x7FFFFFFFu) : ~k;
    return sycl::bit_cast<float>(u);
}

struct TopKPMask {
    float* logits; int64_t row_stride; int V;
    const void* k; int k_is64;          // per-row top_k (int32 or int64), <= 0 or >= V means no top-k
    const float* p;                     // per-row top_p or nullptr
    int* flags;                         // per-row overflow flag (may be nullptr)
    sycl::local_accessor<uint32_t, 1> hist;    // NB
    sycl::local_accessor<float, 1> cand;       // TOPKP_CAP
    sycl::local_accessor<uint32_t, 1> misc;    // [0] prefix, [1] kk, [2] ncand, [3] thr key, [4] bin, [5] above

    void operator()(sycl::nd_item<1> it) const {
        int row = it.get_group(0);
        int lid = it.get_local_id(0);
        auto grp = it.get_group();
        float* x = logits + (size_t)row * row_stride;
        int kk = k_is64 ? (int)reinterpret_cast<const int64_t*>(k)[row] : reinterpret_cast<const int32_t*>(k)[row];
        if (kk <= 0 || kk > V) kk = V;
        float pp = p ? p[row] : 1.0f;

        // ---- 1. row max (float4 loads; V % 4 == 0 checked on the host)
        const sycl::float4* x4 = reinterpret_cast<const sycl::float4*>(x);
        int V4 = V / 4;
        float mx = -INFINITY;
        for (int i = lid; i < V4; i += WG) {
            sycl::float4 v = x4[i];
            mx = sycl::fmax(mx, sycl::fmax(sycl::fmax(v.x(), v.y()), sycl::fmax(v.z(), v.w())));
        }
        mx = sycl::reduce_over_group(grp, mx, sycl::maximum<float>());
        if (mx == -INFINITY) return;                  // fully masked row

        // ---- 2. candidates = logits >= mx - D, with D adapted until k <= n <= CAP (usually one pass).
        // Every element outside the window is below every candidate, so the k-th largest overall is the k-th
        // largest candidate whenever n >= k.
        float D = 16.0f, lo = 0.0f, hi = INFINITY;    // bracket on D: n(D=lo) > CAP, n(D=hi) < k
        int n = 0;
        bool whole = false;                           // window covers every finite logit
        for (int iter = 0; iter < 24; ++iter) {
            float tv = mx - D;
            if (lid == 0) misc[2] = 0;
            sycl::group_barrier(grp);
            for (int i = lid; i < V4; i += WG) {
                sycl::float4 v = x4[i];
                float e[4] = {v.x(), v.y(), v.z(), v.w()};
#pragma unroll
                for (int c = 0; c < 4; ++c)
                    if (e[c] >= tv) {
                        sycl::atomic_ref<uint32_t, sycl::memory_order::relaxed, sycl::memory_scope::work_group,
                                         sycl::access::address_space::local_space> a(misc[2]);
                        uint32_t slot = a.fetch_add(1u);
                        if (slot < (uint32_t)TOPKP_CAP) cand[slot] = e[c];
                    }
            }
            sycl::group_barrier(grp);
            n = (int)misc[2];
            sycl::group_barrier(grp);
            if (n >= kk && n <= TOPKP_CAP) break;
            if (n < kk) {
                if (D > 1e30f) { whole = true; break; }      // fewer than k finite logits: keep them all
                hi = D; D = (lo > 0.0f) ? 0.5f * (lo + hi) : D * 4.0f;
                if (D > 1e6f) D = INFINITY;
            } else {
                lo = D; D = (hi < INFINITY) ? 0.5f * (lo + hi) : D * 0.25f;
            }
        }
        uint32_t thr = 0u;                            // keep keys >= thr (0 = keep everything)
        if (whole || n < kk) {
            thr = 0u;
        } else if (n > TOPKP_CAP) {
            if (lid == 0 && flags) flags[row] = 1;    // pathological ties: leave the row unfiltered
            thr = 0u;
        } else {
            // ---- bitonic sort descending over the next power of two (pad with -inf)
            int np2 = 1;
            while (np2 < n) np2 <<= 1;
            for (int i = n + lid; i < np2; i += WG) cand[i] = -INFINITY;
            sycl::group_barrier(grp);
            for (int size = 2; size <= np2; size <<= 1) {
                for (int stride = size >> 1; stride > 0; stride >>= 1) {
                    for (int i = lid; i < np2 / 2; i += WG) {
                        int lo2 = 2 * i - (i & (stride - 1));
                        int hi2 = lo2 + stride;
                        bool desc = ((lo2 & size) == 0);
                        float a = cand[lo2], b = cand[hi2];
                        if ((a < b) == desc) { cand[lo2] = b; cand[hi2] = a; }
                    }
                    sycl::group_barrier(grp);
                }
            }
            // ---- 3. top-k value, then top-p over the kept set (single item: ~k plus ties)
            if (lid == 0) {
                float kth = cand[kk - 1];
                int nk = kk;
                while (nk < n && cand[nk] >= kth) ++nk;   // ties with the k-th value are kept (as the sort version)
                int last = nk - 1;
                if (pp < 1.0f) {
                    float m0 = cand[0], tot = 0.0f;
                    for (int i = 0; i < nk; ++i) tot += sycl::exp(cand[i] - m0);
                    float lim = pp * tot, above = 0.0f;
                    for (int i = 0; i < nk; ++i) {
                        if (i > 0 && above >= lim) break;
                        last = i;
                        above += sycl::exp(cand[i] - m0);
                    }
                }
                misc[3] = fkey(cand[last]);
            }
            sycl::group_barrier(grp);
            thr = misc[3];
        }
        // ---- 4. mask
        if (thr > 0u) {
            float tval = fval(thr);
            sycl::float4* w4 = reinterpret_cast<sycl::float4*>(x);
            for (int i = lid; i < V4; i += WG) {
                sycl::float4 v = w4[i];
                sycl::float4 o(v.x() < tval ? -INFINITY : v.x(), v.y() < tval ? -INFINITY : v.y(),
                               v.z() < tval ? -INFINITY : v.z(), v.w() < tval ? -INFINITY : v.w());
                if (v.x() < tval || v.y() < tval || v.z() < tval || v.w() < tval) w4[i] = o;
            }
        }
    }
};

}  // namespace exl3tk
