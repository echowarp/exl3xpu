#pragma once
// Fused draft-token sampling for probabilistic MTP drafting.
// Replaces vLLM's compute_probs_and_sample_next_token chain (div by temperature, softmax, exponential fill,
// clone, div, argmax: ~120 us per draft step at vocab 248320) with one kernel, one work-group per row:
//   pass 1: m = max(x / T), s = sum exp(x / T - m)
//   pass 2: p = exp(x / T - m) / s written to probs (the draft distribution q used by rejection sampling),
//           token = argmax_i p_i / E_i with E_i ~ Exp(1)   (exponential race: token ~ p exactly)
// E_i comes from a counter-based hash of (seed, offset, row, i); the caller advances the torch generator offset.

#include <sycl/sycl.hpp>

namespace exl3ds {

constexpr int WG = 1024;

static inline uint64_t mix64(uint64_t z) {         // splitmix64 finaliser
    z += 0x9E3779B97F4A7C15ull;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return z ^ (z >> 31);
}

struct DraftSample {
    const float* logits; int64_t lstride; float* probs; int64_t pstride; int64_t* token;
    const float* temperature; int V; uint64_t seed, offset;
    void operator()(sycl::nd_item<1> it) const {
        int row = it.get_group(0), lid = it.get_local_id(0);
        auto grp = it.get_group();
        const float* x = logits + (size_t)row * lstride;
        float* p = probs + (size_t)row * pstride;
        float invt = 1.0f / temperature[row];
        float m = -INFINITY;
        for (int i = lid; i < V; i += WG) m = sycl::fmax(m, x[i] * invt);
        m = sycl::reduce_over_group(grp, m, sycl::maximum<float>());
        float s = 0.0f;
        if (m != -INFINITY)
            for (int i = lid; i < V; i += WG) s += sycl::exp(x[i] * invt - m);
        s = sycl::reduce_over_group(grp, s, sycl::plus<float>());
        float invs = s > 0.0f ? 1.0f / s : 0.0f;
        float best = -1.0f; int bi = 0x7fffffff;
        uint64_t base = mix64(seed ^ mix64(offset + (uint64_t)row * 0x100000001B3ull));
        for (int i = lid; i < V; i += WG) {
            float pi = (m != -INFINITY) ? sycl::exp(x[i] * invt - m) * invs : 0.0f;
            p[i] = pi;
            if (pi > 0.0f) {
                uint64_t h = mix64(base + (uint64_t)i);
                float u = ((float)(h >> 40) + 0.5f) * (1.0f / 16777216.0f);   // (0, 1)
                float sc = pi / -sycl::log(u);
                if (sc > best) { best = sc; bi = i; }
            }
        }
        float gbest = sycl::reduce_over_group(grp, best, sycl::maximum<float>());
        int cand = (best == gbest) ? bi : 0x7fffffff;
        int gi = sycl::reduce_over_group(grp, cand, sycl::minimum<int>());
        if (lid == 0) token[row] = gi == 0x7fffffff ? 0 : gi;
    }
};


// Multi-work-group version: NCH chunks per row (rows are 1 MB; one work-group per row is ALU-bound on one Xe core).
constexpr int NCH = 64, WG2 = 256;
struct DsPartial {     // part[row][c] = (max, sum exp(x/T - max)) over chunk c
    const float* logits; int64_t lstride; const float* temperature; float* part; int V, span;
    void operator()(sycl::nd_item<1> it) const {
        int g = it.get_group(0), row = g / NCH, c = g % NCH, lid = it.get_local_id(0);
        auto grp = it.get_group();
        const float* x = logits + (size_t)row * lstride;
        float invt = 1.0f / temperature[row];
        int a = c * span, b = a + span < V ? a + span : V;
        float m = -INFINITY;
        for (int i = a + lid; i < b; i += WG2) m = sycl::fmax(m, x[i] * invt);
        m = sycl::reduce_over_group(grp, m, sycl::maximum<float>());
        float sm = 0.0f;
        if (m != -INFINITY) for (int i = a + lid; i < b; i += WG2) sm += sycl::exp(x[i] * invt - m);
        sm = sycl::reduce_over_group(grp, sm, sycl::plus<float>());
        if (lid == 0) { part[(size_t)g * 2] = m; part[(size_t)g * 2 + 1] = sm; }
    }
};
struct DsSample {      // probs for chunk c + its best exponential-race candidate -> best[row][c] = (score, idx)
    const float* logits; int64_t lstride; float* probs; int64_t pstride; const float* temperature;
    const float* part; float* best; int V, span; uint64_t seed, offset;
    void operator()(sycl::nd_item<1> it) const {
        int g = it.get_group(0), row = g / NCH, c = g % NCH, lid = it.get_local_id(0);
        auto grp = it.get_group();
        const float* pr = part + (size_t)row * NCH * 2;
        float m = -INFINITY;
        for (int j = 0; j < NCH; ++j) m = sycl::fmax(m, pr[2 * j]);
        float s = 0.0f;
        if (m != -INFINITY) for (int j = 0; j < NCH; ++j) if (pr[2 * j] != -INFINITY) s += pr[2 * j + 1] * sycl::exp(pr[2 * j] - m);
        float invs = s > 0.0f ? 1.0f / s : 0.0f, invt = 1.0f / temperature[row];
        const float* x = logits + (size_t)row * lstride;
        float* p = probs + (size_t)row * pstride;
        int a = c * span, b = a + span < V ? a + span : V;
        uint64_t base = mix64(seed ^ mix64(offset + (uint64_t)row * 0x100000001B3ull));
        float bs = -1.0f; int bi = 0x7fffffff;
        for (int i = a + lid; i < b; i += WG2) {
            float pi = (m != -INFINITY) ? sycl::exp(x[i] * invt - m) * invs : 0.0f;
            p[i] = pi;
            if (pi > 0.0f) {
                uint64_t h = mix64(base + (uint64_t)i);
                float u = ((float)(h >> 40) + 0.5f) * (1.0f / 16777216.0f);
                float sc = pi / -sycl::log(u);
                if (sc > bs) { bs = sc; bi = i; }
            }
        }
        float gb = sycl::reduce_over_group(grp, bs, sycl::maximum<float>());
        int gi = sycl::reduce_over_group(grp, bs == gb ? bi : 0x7fffffff, sycl::minimum<int>());
        if (lid == 0) { best[(size_t)g * 2] = gb; best[(size_t)g * 2 + 1] = sycl::bit_cast<float>(gi); }
    }
};
struct DsPick {        // token[row] = idx of the max score over chunks (lowest idx on ties); one sub-group per row
    const float* best; int64_t* token;
    void operator()(sycl::nd_item<1> it) const {
        int row = it.get_group(0), lid = it.get_local_id(0);
        auto grp = it.get_group();
        float sc = best[((size_t)row * NCH + lid) * 2];
        int ix = sycl::bit_cast<int>(best[((size_t)row * NCH + lid) * 2 + 1]);
        float gb = sycl::reduce_over_group(grp, sc, sycl::maximum<float>());
        int gi = sycl::reduce_over_group(grp, sc == gb ? ix : 0x7fffffff, sycl::minimum<int>());
        if (lid == 0) token[row] = gi == 0x7fffffff ? 0 : gi;
    }
};

// DS-SUB: sample directly from the pruned draft head's compact logits (n_sub <= 2048 columns, the vocab
// ids in idx[]) instead of a -inf-filled full-vocab row. Same distribution (softmax over the kept ids = the full-vocab
// softmax with -inf elsewhere). Graph-safe RNG: offset = ctr[0] (device counter, advanced by the caller on the device)
// + salt (per draft pass). Writes q into probs at the kept ids only (the caller zero-fills the full rows).
constexpr int WGS = 256;
template <typename TL>
struct DsSub {
    const TL* logits; int64_t lstride; const int64_t* idx; int n; const float* temperature;
    float* probs; int64_t pstride; int64_t* token; const int64_t* ctr; uint64_t seed, salt;
    void operator()(sycl::nd_item<1> it) const {
        int row = it.get_group(0), lid = it.get_local_id(0);
        auto grp = it.get_group();
        const TL* x = logits + (size_t)row * lstride;
        float invt = 1.0f / temperature[row];
        float m = -INFINITY;
        for (int i = lid; i < n; i += WGS) m = sycl::fmax(m, (float)x[i] * invt);
        m = sycl::reduce_over_group(grp, m, sycl::maximum<float>());
        float s = 0.0f;
        if (m != -INFINITY) for (int i = lid; i < n; i += WGS) s += sycl::exp((float)x[i] * invt - m);
        s = sycl::reduce_over_group(grp, s, sycl::plus<float>());
        float invs = s > 0.0f ? 1.0f / s : 0.0f;
        uint64_t off = (uint64_t)ctr[0] * 0x9E3779B1ull + salt;
        uint64_t base = mix64(seed ^ mix64(off + (uint64_t)row * 0x100000001B3ull));
        float best = -1.0f; int bi = 0x7fffffff;
        float* p = probs + (size_t)row * pstride;
        for (int i = lid; i < n; i += WGS) {
            float pi = (m != -INFINITY) ? sycl::exp((float)x[i] * invt - m) * invs : 0.0f;
            int64_t v = idx[i];
            p[v] = pi;
            if (pi > 0.0f) {
                uint64_t h = mix64(base + (uint64_t)v);
                float u = ((float)(h >> 40) + 0.5f) * (1.0f / 16777216.0f);
                float sc = pi / -sycl::log(u);
                if (sc > best) { best = sc; bi = i; }
            }
        }
        float gb = sycl::reduce_over_group(grp, best, sycl::maximum<float>());
        int gi = sycl::reduce_over_group(grp, best == gb ? bi : 0x7fffffff, sycl::minimum<int>());
        if (lid == 0) token[row] = gi == 0x7fffffff ? idx[0] : idx[gi];
    }
};

}  // namespace exl3ds
