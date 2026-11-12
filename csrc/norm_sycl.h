// fused residual add + RMSNorm for small M (decode), one work-group per row.
// Semantics follow vLLM's fused_add_rms_norm: z = x + residual (fp32), residual = half(z), variance from fp32 z,
// x = half(half(z) * rsqrt(var / H + eps)) * w (fp16 multiply), weight = (w + 1) in fp16 as the ESIMD path uses.
#pragma once
#include <sycl/sycl.hpp>
namespace exl3norm {
using half = sycl::half;
constexpr int VEC = 8;
struct FusedAddRms {
    half* x; half* res; const half* w; int H; int sx; int sr; float eps;
    void operator()(sycl::nd_item<1> it) const {
        const int row = it.get_group(0), lid = it.get_local_id(0), L = it.get_local_range(0);
        half* xr = x + (size_t)row * sx; half* rr = res + (size_t)row * sr;
        float acc = 0.f;
        for (int i = lid * VEC; i < H; i += L * VEC) {
            sycl::vec<half, VEC> a = *reinterpret_cast<const sycl::vec<half, VEC>*>(xr + i);
            sycl::vec<half, VEC> b = *reinterpret_cast<const sycl::vec<half, VEC>*>(rr + i);
            sycl::vec<half, VEC> o;
#pragma unroll
            for (int j = 0; j < VEC; ++j) { float z = (float)a[j] + (float)b[j]; acc += z * z; o[j] = (half)z; }
            *reinterpret_cast<sycl::vec<half, VEC>*>(rr + i) = o;
        }
        acc = sycl::reduce_over_group(it.get_group(), acc, sycl::plus<float>());
        const float rs = sycl::rsqrt(acc / (float)H + eps);
        for (int i = lid * VEC; i < H; i += L * VEC) {
            sycl::vec<half, VEC> z = *reinterpret_cast<const sycl::vec<half, VEC>*>(rr + i);
            sycl::vec<half, VEC> wv = *reinterpret_cast<const sycl::vec<half, VEC>*>(w + i);
            sycl::vec<half, VEC> o;
#pragma unroll
            for (int j = 0; j < VEC; ++j) o[j] = (half)((float)z[j] * rs) * wv[j];
            *reinterpret_cast<sycl::vec<half, VEC>*>(xr + i) = o;
        }
    }
};
}  // namespace exl3norm
