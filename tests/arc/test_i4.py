"""int4 KV kernels: exl3_i4_gather_dequant(_hm) and exl3_spec_attn on vLLM's int4_per_token_head
cache, written by vLLM's own reshape_and_cache_int4. Synthetic tensors only.
  python test_i4.py exl3xpu/_C.so"""
import sys

import torch

torch.ops.load_library(sys.argv[1])
E = torch.ops.exl3xpu_C
from vllm.v1.attention.ops.int4_per_token_head import reshape_and_cache_int4, single_rht

dev = "xpu"
torch.manual_seed(0)
H, G, D, BS = 4, 6, 256, 3200
Hq = H * G
FAILS = []


def check(name, ok, info=""):
    print(("PASS " if ok else "FAIL ") + name + (" " + info if info else ""), flush=True)
    if not ok:
        FAILS.append(name)


def make_cache(nb):
    kv = torch.zeros(nb, BS, H, 264, dtype=torch.uint8, device=dev).permute(0, 2, 1, 3)   # logical (NB, H, BS, 264), NHD
    kc, vc = kv.transpose(1, 2).split(132, dim=-1)
    raw = torch.tensor([], dtype=torch.float32, device=dev).set_(kv.untyped_storage())
    st = kv.stride()
    ks = torch.as_strided(raw, (nb, BS, H), (st[0] // 4, st[2] // 4, st[1] // 4), 128 // 4)
    vs = torch.as_strided(raw, (nb, BS, H), (st[0] // 4, st[2] // 4, st[1] // 4), 260 // 4)
    return kv, kc, vc, ks, vs


def dequant(view, pages, n):
    """python reference: [n, H, 256] fp32 in the rotated space from a [NB, BS, H, 132] uint8 view"""
    rows = view[pages.long()].reshape(-1, H, 132)[:n]
    data = rows[..., :128]
    sb = rows[..., 128:132].contiguous().view(torch.int32)[..., 0]
    zp = (sb & 0xF).float()
    sc = (sb & ~0xF).view(torch.float32)
    lo, hi = (data & 0xF).float(), (data >> 4).float()
    q = torch.stack([lo, hi], -1).reshape(n, H, 256)
    return (q - zp[..., None]) * sc[..., None]


def ref_attn(q, k, v, L, ql, scale):
    """q [ql, Hq, D] (rotated, /D), k/v [L, H, D] -> [ql, Hq, D]"""
    qg = q.float().reshape(ql, H, G, D)
    s = torch.einsum("qhgd,khd->hgqk", qg, k.float()) * scale
    pos = torch.arange(L, device=dev)[None, :]
    lim = (L - ql + torch.arange(ql, device=dev))[:, None]
    s = s.masked_fill((pos > lim)[None, None], float("-inf"))
    return torch.einsum("hgqk,khd->qhgd", s.softmax(-1), v.float()).reshape(ql, H * G, D)


scale = D ** -0.5
for L, ql, nseq in ((500, 1, 1), (5000, 4, 2), (20000, 8, 2), (33000, 1, 3)):
    nbs = (L + BS - 1) // BS
    kv, kc, vc, ks, vs = make_cache(nbs * nseq + 1)
    bt = torch.zeros(nseq, nbs, dtype=torch.int32, device=dev)
    K, V = [], []
    for b in range(nseq):
        pages = torch.randperm(nbs * nseq + 1, device=dev)[:nbs].int() if b == 0 else \
            torch.tensor([p for p in range(nbs * nseq + 1) if p not in set(bt[:b].flatten().tolist())][:nbs], dtype=torch.int32, device=dev)
        bt[b] = pages
        k = torch.randn(L, H, D, device=dev, dtype=torch.float16)
        k[:, :, 5] += 4.0                                   # a channel outlier, as real keys have
        v = torch.randn(L, H, D, device=dev, dtype=torch.float16)
        slots = (pages.long()[torch.arange(L, device=dev) // BS] * BS + torch.arange(L, device=dev) % BS)
        reshape_and_cache_int4(k, v, kc, vc, slots, k_scale_cache=ks, v_scale_cache=vs)
        K.append(k); V.append(v)
    torch.xpu.synchronize()
    # gather-dequant (head-major and token-major) vs the python dequant of the same bytes
    kd = dequant(kc, bt[0], L); vd = dequant(vc, bt[0], L)
    Lp = (L + 255) // 256 * 256
    ko = torch.empty(H, Lp, D, dtype=torch.float16, device=dev); vo = torch.empty_like(ko)
    E.exl3_i4_gather_dequant_hm(kc, bt[0], L, ko, vo)
    e1 = max((ko[:, :L].float().permute(1, 0, 2) - kd).abs().max().item(), (vo[:, :L].float().permute(1, 0, 2) - vd).abs().max().item())
    pad0 = ko[:, L:].abs().max().item() if Lp > L else 0.0
    check(f"gather_dequant_hm L={L}", e1 <= 1e-2 * max(1.0, kd.abs().max().item()) and pad0 == 0.0, f"max|diff| {e1:.2e} pad {pad0}")
    k2 = torch.empty(L, H, D, dtype=torch.float16, device=dev); v2 = torch.empty_like(k2)
    E.exl3_i4_gather_dequant(kc, bt[0], 0, L, k2, v2)
    e2 = max((k2.float() - kd).abs().max().item(), (v2.float() - vd).abs().max().item())
    check(f"gather_dequant L={L}", e2 <= 1e-2 * max(1.0, kd.abs().max().item()), f"max|diff| {e2:.2e}")
    # decode / verify attention
    q = torch.randn(nseq * ql, Hq, D, device=dev, dtype=torch.float16)
    qr = (single_rht(q.float()) / D).half()
    seq = torch.full((nseq,), L, dtype=torch.int32, device=dev)
    out = torch.empty(nseq * ql, Hq, D, dtype=torch.float16, device=dev)
    E.exl3_spec_attn(qr, kc, vc, bt, seq, out, ql, scale, 0, None, None)
    torch.xpu.synchronize()
    worst, worst_true = 0.0, 0.0
    for b in range(nseq):
        kdb = dequant(kc, bt[b], L); vdb = dequant(vc, bt[b], L)
        ref = ref_attn(qr[b * ql:(b + 1) * ql].float(), kdb, vdb, L, ql, scale)          # rotated-space reference
        got = out[b * ql:(b + 1) * ql].float()
        rel = ((got - ref).norm() / ref.norm()).item()
        worst = max(worst, rel)
        # against unquantized attention on the original K, V (overall int4 error, rotation conventions)
        o_un = single_rht(got, inverse=True) / D
        tru = ref_attn(q[b * ql:(b + 1) * ql].float(), K[b].float(), V[b].float(), L, ql, scale)
        worst_true = max(worst_true, ((o_un - tru).norm() / tru.norm()).item())
    check(f"spec_attn L={L} ql={ql} B={nseq} vs dequant reference", worst < 2e-2, f"rel err {worst:.2e}")
    check(f"spec_attn L={L} ql={ql} B={nseq} vs unquantized (int4 error, rotation)", worst_true < 0.2, f"rel err {worst_true:.3f}")

# varlen: requests with 1, 4, 8 query rows in one launch (mixed decode / verify / drafter batches)
L = 9000; nbs = (L + BS - 1) // BS; nseq = 3
kv, kc, vc, ks, vs = make_cache(nbs * nseq)
bt = torch.arange(nbs * nseq, dtype=torch.int32, device=dev).view(nseq, nbs).flip(1).contiguous()
for b in range(nseq):
    k = torch.randn(L, H, D, device=dev, dtype=torch.float16); k[:, :, 5] += 4.0
    v = torch.randn(L, H, D, device=dev, dtype=torch.float16)
    slots = (bt[b].long()[torch.arange(L, device=dev) // BS] * BS + torch.arange(L, device=dev) % BS)
    reshape_and_cache_int4(k, v, kc, vc, slots, k_scale_cache=ks, v_scale_cache=vs)
qls = [1, 4, 8]
qsl = torch.tensor([0, 1, 5, 13], dtype=torch.int32, device=dev)
q = (single_rht(torch.randn(13, Hq, D, device=dev).float()) / D).half()   # rotated queries, as the engine sends them
seq = torch.full((nseq,), L, dtype=torch.int32, device=dev)
out = torch.zeros_like(q)
E.exl3_spec_attn_varlen(q, kc, vc, bt, seq, qsl, out, scale, 0, None, None)
torch.xpu.synchronize()
worst = 0.0
for b in range(nseq):
    a, e = qsl[b].item(), qsl[b + 1].item()
    ref = ref_attn(q[a:e].float(), dequant(kc, bt[b], L), dequant(vc, bt[b], L), L, e - a, scale)
    e_b = ((out[a:e].float() - ref).norm() / ref.norm()).item(); print(f"   varlen request {b} rows {e-a}: rel err {e_b:.2e}"); worst = max(worst, e_b)
o2 = torch.zeros(4, Hq, D, dtype=torch.float16, device=dev)
E.exl3_spec_attn(q[1:5].contiguous(), kc, vc, bt[1:2].contiguous(), seq[1:2].contiguous(), o2, 4, scale, 0, None, None)
torch.xpu.synchronize()
r1 = ref_attn(q[1:5].float(), dequant(kc, bt[1], L), dequant(vc, bt[1], L), L, 4, scale)
print(f"   same inputs, non-varlen kernel: rel err {((o2.float() - r1).norm() / r1.norm()).item():.2e}; varlen vs non-varlen {((o2.float() - out[1:5].float()).norm() / o2.float().norm()).item():.2e}")
check("spec_attn_varlen rows 1/4/8 vs dequant reference", worst < 2e-2, f"rel err {worst:.2e}")
print("FAILED: " + ", ".join(FAILS) if FAILS else "all passed")
sys.exit(1 if FAILS else 0)
