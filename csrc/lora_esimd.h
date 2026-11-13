#pragma once
// LoRA terms inside the EXL3 small-M linear and fused decode MLP (llm-tuning 2026-11-13, branch lora-kernels).
//
// A LoRA linear on a decode step used to cost the base EXL3 linear (had_in -> GEMM -> had_out, 3 launches) + vLLM's
// fused punica kernel (1 more launch, ~4-10 us per call in XPU graphs; 304 calls per step = 2-3 ms). Here the LoRA
// work rides on the kernels the base linear launches anyway, no extra launch:
//   had_in  (one thread per (row, 128-wide k block), reads x anyway): + shrink partials
//           lp[j][kb][m][0:16] = A_j[slot(m), 0, 0:16, kb*128:+128] . x[m, kb*128:+128]   (raw x, distinct A_j)
//   GEMM    unchanged (and it orders the two phases for free)
//   had_out (one thread per (row, 128-wide n block), writes y anyway): h = sum_kb lp[j(s)][kb][m] (scaled),
//           y[m, n block] = had_out(base) + B_s[slot(m), 0, n block, 0:16] @ h, one fp16 rounding
// and in the fused decode MLP (mlp_small): had_in + gate/up shrink; MlpMid adds the gate/up LoRA before the activation
// and computes the down shrink from the activation it already holds; the final had_out adds the down LoRA.
// Rows with slot < 0 run exactly the base arithmetic (same code as HadOutKernel / MlpMidKernel). Rank 16 only.
// Slice boundaries must be multiples of 128 (host-checked).

namespace exl3 {

constexpr int LR = 16;   // LoRA rank handled here
constexpr int LMAXG = 4, LMAXS = 4;

struct LoraArgs {
    const fp16* A[LMAXG]; int64_t as[LMAXG];   // distinct A_j: [L, 1, 16, K] (adapter stride as[j], rank row stride K)
    const fp16* B[LMAXS]; int64_t bs[LMAXS];   // slice B_s: [L, 1, N_s, 16]
    int off[LMAXS + 1]; int ga[LMAXS];         // slice column offsets in y, A index of each slice
    int G, S;
    const int64_t* idx64; const int32_t* idx32;   // slot per row (-1 = none); one of them non-null
    float* lp; int KB;                         // shrink partials [G][KB][M][16], KB = K / 128
    float scale;
};

ESIMD_INLINE int lora_slot(const LoraArgs& la, int m) {
    return la.idx64 ? (int)la.idx64[m] : (int)la.idx32[m];
}

// lp[j][kb][m][:] for all distinct A_j from the fp32 values xf of x[m, kb*128:+128]
ESIMD_INLINE void lora_shrink_block(const LoraArgs& la, int ad, int m, int M, int kb, int Kdim, const simd<float, 128>& xf) {
    for (int j = 0; j < la.G; ++j) {
        const fp16* a = la.A[j] + (int64_t)ad * la.as[j] + kb * 128;
        simd<float, LR> s;
#pragma unroll
        for (int r = 0; r < LR; ++r) {
            simd<float, 128> p = convert<float>(block_load<fp16, 128>(a + (int64_t)r * Kdim)) * xf;
            s[r] = reduce<float>(p, std::plus<>());
        }
        block_store<float, LR>(la.lp + (((size_t)j * la.KB + kb) * M + m) * LR, s);
    }
}

// h = scale * sum_kb lp[j][kb][m][:]
ESIMD_INLINE simd<float, LR> lora_h(const LoraArgs& la, int j, int m, int M) {
    simd<float, LR> h = 0.0f;
    const float* p = la.lp + ((size_t)j * la.KB * M + m) * LR;
    int kb = 0;
    for (; kb + 4 <= la.KB; kb += 4) {
        simd<float, LR> t0 = block_load<float, LR>(p + (size_t)(kb + 0) * M * LR);
        simd<float, LR> t1 = block_load<float, LR>(p + (size_t)(kb + 1) * M * LR);
        simd<float, LR> t2 = block_load<float, LR>(p + (size_t)(kb + 2) * M * LR);
        simd<float, LR> t3 = block_load<float, LR>(p + (size_t)(kb + 3) * M * LR);
        h += (t0 + t1) + (t2 + t3);
    }
    for (; kb < la.KB; ++kb) h += block_load<float, LR>(p + (size_t)kb * M * LR);
    return h * la.scale;
}

// d[c] = B_s[ad, 0, nl + c, :] . h  for 128 columns starting at slice-local column nl
ESIMD_INLINE simd<float, 128> lora_expand_block(const LoraArgs& la, int s, int ad, int nl, const simd<float, LR>& h) {
    const fp16* b = la.B[s] + (int64_t)ad * la.bs[s] + (int64_t)nl * LR;
    simd<float, 128> d = 0.0f;
#pragma unroll
    for (int c = 0; c < 4; ++c) {   // 32 columns x 16 ranks per load
        simd<fp16, 32 * LR> bh;
#pragma unroll
        for (int q = 0; q < 4; ++q) bh.template select<128, 1>(q * 128) = block_load<fp16, 128>(b + c * 32 * LR + q * 128);
        simd<float, 32 * LR> bf = convert<float>(bh);
        simd<float, 32> acc = 0.0f;
#pragma unroll
        for (int r = 0; r < LR; ++r) acc += bf.template select<32, LR>(r) * h[r];
        d.template select<32, 1>(c * 32) = acc;
    }
    return d;
}

ESIMD_INLINE int lora_slice_of(const LoraArgs& la, int n) {
    int s = 0;
#pragma unroll
    for (int q = 1; q < LMAXS; ++q) s += (q < la.S && n >= la.off[q]) ? 1 : 0;
    return s;
}

// HadInKernel + shrink partials (shard g == 0 threads; raw x)
template <typename TIn>
struct HadInLoraKernel {
    const TIn* x; const fp16* suh; fp16* xh;
    int M, Kdim, S, x_stride, Mp;
    int row_major;
    LoraArgs la;
    void operator()(sycl::nd_item<1> it) const SYCL_ESIMD_KERNEL {
        int id = it.get_global_id(0);
        int kb_n = Kdim / 128;
        int kb = id % kb_n;
        int rows = row_major ? M : Mp;
        int m = (id / kb_n) % rows;
        int g = id / (kb_n * rows);
        if (g >= S) return;
        if (m >= M) {
            int kt = Kdim / 16;
            simd<fp16, 16> z = 0;
#pragma unroll
            for (int i = 0; i < 8; ++i)
                block_store<fp16, 16>(xh + (((size_t)g * kt + kb * 8 + i) * Mp + m) * 16, z);
            return;
        }
        simd<TIn, 128> xi = block_load<TIn, 128>(x + (size_t)m * x_stride + kb * 128);
        if (g == 0) {
            int ad = lora_slot(la, m);
            if (ad >= 0) lora_shrink_block(la, ad, m, M, kb, Kdim, convert<float>(xi));
        }
        simd<fp16, 128> su = block_load<fp16, 128>(suh + (size_t)g * Kdim + kb * 128);
        simd<float, 128> v = convert<float>(xi) * convert<float>(su);
        v = convert<float>(convert<fp16>(v));
        fwht128(v);
        v *= kRsqrt128;
        simd<fp16, 128> vh = convert<fp16>(v);
        if (row_major) {
            block_store<fp16, 128>(xh + ((size_t)g * M + m) * Kdim + kb * 128, vh);
            return;
        }
        int kt = Kdim / 16;
#pragma unroll
        for (int i = 0; i < 8; ++i)
            block_store<fp16, 16>(xh + (((size_t)g * kt + kb * 8 + i) * Mp + m) * 16, vh.template select<16, 1>(i * 16));
    }
};

// HadOutKernel (fp16 out, fp32 partials) + expand for LoRA rows. Base rows: same arithmetic as HadOutKernel.
struct HadOutLoraKernel {
    const float* part; const fp16* svh; fp16* out;
    int M, N, P, out_stride;
    LoraArgs la;
    void operator()(sycl::nd_item<1> it) const SYCL_ESIMD_KERNEL {
        int id = it.get_global_id(0);
        int nb_n = N / 128;
        int nb = id % nb_n;
        int m = id / nb_n;
        if (m >= M) return;
        simd<float, 128> v = 0.0f;
#ifdef EXL3_HADOUT_B
        constexpr int HB = EXL3_HADOUT_B;
        {
#pragma clang fp reassociate(off)
        int p = 0;
        for (; p + HB <= P; p += HB) {
            simd<float, 128> t[HB];
#pragma unroll
            for (int b = 0; b < HB; ++b) t[b] = block_load<float, 128>(part + ((size_t)(p + b) * M + m) * N + nb * 128);
#pragma unroll
            for (int b = 0; b < HB; ++b) v += t[b];
        }
        for (; p < P; ++p)
            v += block_load<float, 128>(part + ((size_t)p * M + m) * N + nb * 128);
        }
        fwht128(v);
        simd<fp16, 128> sv = block_load<fp16, 128>(svh + nb * 128);
#else
        for (int p = 0; p < P; ++p)
            v += block_load<float, 128>(part + ((size_t)p * M + m) * N + nb * 128);
        fwht128(v);
        simd<fp16, 128> sv = block_load<fp16, 128>(svh + nb * 128);
#endif
        v = v * kRsqrt128 * convert<float>(sv);
        int ad = lora_slot(la, m);
        if (ad >= 0) {
            int n = nb * 128, s = lora_slice_of(la, n);
            v += lora_expand_block(la, s, ad, n - la.off[s], lora_h(la, la.ga[s], m, M));
        }
        block_store<fp16, 128>(out + (size_t)m * out_stride + nb * 128, convert<fp16>(v));
    }
};

// MlpMidKernel + gate/up LoRA (slices 0 = gate [0, I), 1 = up [I, 2I) of la_gu) before the activation + down shrink
// partials from the activation (la_dn.lp[j][kb = column block][m]). Base rows: same arithmetic as MlpMidKernel.
struct MlpMidLoraKernel {
    const float* part; int P, M, I;
    const fp16* svh;
    const fp16* suh;
    fp16* xh;
    int Mp;
    LoraArgs la_gu, la_dn;
    void operator()(sycl::nd_item<1> it) const SYCL_ESIMD_KERNEL {
        int id = it.get_global_id(0);
        int nb_n = I / 128;
        int j = id % nb_n, m = id / nb_n;
        if (m >= Mp) return;
        if (m >= M) {
            simd<fp16, 16> z = 0;
#pragma unroll
            for (int i = 0; i < 8; ++i) block_store<fp16, 16>(xh + ((size_t)(j * 8 + i) * Mp + m) * 16, z);
            return;
        }
        simd<float, 128> g = 0.0f, u = 0.0f;
#ifdef EXL3_HADOUT_B
        {
#pragma clang fp reassociate(off)
        constexpr int MB2 = EXL3_HADOUT_B / 2 > 0 ? EXL3_HADOUT_B / 2 : 1;
        int p = 0;
        for (; p + MB2 <= P; p += MB2) {
            simd<float, 128> tg[MB2], tu[MB2];
#pragma unroll
            for (int b = 0; b < MB2; ++b) {
                tg[b] = block_load<float, 128>(part + ((size_t)(p + b) * M + m) * 2 * I + j * 128);
                tu[b] = block_load<float, 128>(part + ((size_t)(p + b) * M + m) * 2 * I + I + j * 128);
            }
#pragma unroll
            for (int b = 0; b < MB2; ++b) { g += tg[b]; u += tu[b]; }
        }
        for (; p < P; ++p) {
            g += block_load<float, 128>(part + ((size_t)p * M + m) * 2 * I + j * 128);
            u += block_load<float, 128>(part + ((size_t)p * M + m) * 2 * I + I + j * 128);
        }
        }
#else
        for (int p = 0; p < P; ++p) {
            g += block_load<float, 128>(part + ((size_t)p * M + m) * 2 * I + j * 128);
            u += block_load<float, 128>(part + ((size_t)p * M + m) * 2 * I + I + j * 128);
        }
#endif
        fwht128(g); fwht128(u);
        simd<fp16, 128> gh = convert<fp16>(g * kRsqrt128 * convert<float>(block_load<fp16, 128>(svh + j * 128)));
        simd<fp16, 128> uh = convert<fp16>(u * kRsqrt128 * convert<float>(block_load<fp16, 128>(svh + I + j * 128)));
        int ad = lora_slot(la_gu, m);
        if (ad >= 0) {   // gate_up output + LoRA, rounded to fp16 once (as the unfused path stores gate_up)
            simd<float, 128> dg = lora_expand_block(la_gu, 0, ad, j * 128, lora_h(la_gu, la_gu.ga[0], m, M));
            simd<float, 128> du = lora_expand_block(la_gu, 1, ad, j * 128, lora_h(la_gu, la_gu.ga[1], m, M));
            gh = convert<fp16>(g * kRsqrt128 * convert<float>(block_load<fp16, 128>(svh + j * 128)) + dg);
            uh = convert<fp16>(u * kRsqrt128 * convert<float>(block_load<fp16, 128>(svh + I + j * 128)) + du);
        }
        simd<float, 128> gf = convert<float>(gh);
        simd<float, 128> a = gf / (1.0f + exp(-gf)) * convert<float>(uh);
        simd<fp16, 128> ah = convert<fp16>(a);
        int add = lora_slot(la_dn, m);
        if (add >= 0) lora_shrink_block(la_dn, add, m, M, j, I, convert<float>(ah));
        simd<float, 128> v = convert<float>(ah) * convert<float>(block_load<fp16, 128>(suh + j * 128));
        v = convert<float>(convert<fp16>(v));
        fwht128(v);
        v *= kRsqrt128;
        simd<fp16, 128> vh = convert<fp16>(v);
#pragma unroll
        for (int i = 0; i < 8; ++i)
            block_store<fp16, 16>(xh + ((size_t)(j * 8 + i) * Mp + m) * 16, vh.template select<16, 1>(i * 16));
    }
};

}  // namespace exl3
