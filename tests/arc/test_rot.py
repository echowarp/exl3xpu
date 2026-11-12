"""I4-ROT: in-kernel q / output rotation vs the plugin's host matmuls, on a cache written by
vLLM's own reshape_and_cache_int4. Synthetic tensors.  python test_rot.py exl3xpu/_C.so"""
import sys, torch
torch.ops.load_library(sys.argv[1]); E = torch.ops.exl3xpu_C
from vllm.v1.attention.ops.int4_per_token_head import reshape_and_cache_int4, _get_rht_signs
dev = "xpu"; torch.manual_seed(0)
H, G, D, BS = 4, 6, 256, 3200; Hq = H * G
h = torch.ones(1, 1)
while h.shape[0] < 256: h = torch.cat([torch.cat([h, h], 1), torch.cat([h, -h], 1)], 0)
sg = _get_rht_signs(256, 0, torch.device("cpu")).float()
m = (sg[:, None] * (h / 16.0)).to(dev); fm, bm = m / 16.0, m.t().contiguous() / 16.0
signs = sg.contiguous().to(dev)
fails = []
def dequant(view, pages, n):
    rows = view[pages.long()].reshape(-1, H, 132)[:n]
    data = rows[..., :128]
    sb = rows[..., 128:132].contiguous().view(torch.int32)[..., 0]
    zp = (sb & 0xF).float(); sc = (sb & ~0xF).view(torch.float32)
    lo, hi = (data & 0xF).float(), (data >> 4).float()
    return (torch.stack([lo, hi], -1).reshape(n, H, 256) - zp[..., None]) * sc[..., None]
def exact(q, kc, vc, bt, L, ql, scale):
    """fp32 attention on the dequantized (rotated) cache with exactly rotated q; output rotated back: [B*ql, Hq, D]"""
    outs = []
    for b in range(bt.shape[0]):
        kr = dequant(kc, bt[b], L); vr = dequant(vc, bt[b], L)
        qr = q[b * ql:(b + 1) * ql].float() @ fm                  # [ql, Hq, D]
        kk = kr.repeat_interleave(G, 1); vv = vr.repeat_interleave(G, 1)
        s = torch.einsum("qhd,khd->hqk", qr, kk) * scale
        pos = torch.arange(L, device=dev)[None, :] <= (L - ql + torch.arange(ql, device=dev))[:, None]
        s = s.masked_fill(~pos[None], float("-inf"))
        o = torch.einsum("hqk,khd->qhd", s.softmax(-1), vv)
        outs.append(o @ bm)
    return torch.cat(outs)
def make(L, B):
    nbs = (L + BS - 1) // BS
    kv = torch.zeros(nbs * B, BS, H, 264, dtype=torch.uint8, device=dev).permute(0, 2, 1, 3)
    kc, vc = kv.transpose(1, 2).split(132, dim=-1)
    raw = torch.tensor([], dtype=torch.float32, device=dev).set_(kv.untyped_storage())
    st = kv.stride()
    ks = torch.as_strided(raw, (nbs * B, BS, H), (st[0] // 4, st[2] // 4, st[1] // 4), 128 // 4)
    vs = torch.as_strided(raw, (nbs * B, BS, H), (st[0] // 4, st[2] // 4, st[1] // 4), 260 // 4)
    k = torch.randn(nbs * B * BS, H, D, device=dev, dtype=torch.float16)
    v = torch.randn_like(k)
    slots = torch.arange(nbs * B * BS, device=dev)
    return kv, kc, vc, k, v, slots, nbs, ks, vs
for L, B, ql, varlen in ((5000, 1, 1, False), (20000, 2, 8, False), (33000, 3, 4, False), (40000, 3, 8, True), (9000, 1, 1, True)):
    kv, kc, vc, k, v, slots, nbs, ks, vs = make(L, B)
    try:
        reshape_and_cache_int4(k, v, kc, vc, slots, k_scale_cache=ks, v_scale_cache=vs)
    except TypeError as e:
        print("signature mismatch", e); raise
    bt = torch.arange(nbs * B, dtype=torch.int32, device=dev).view(B, nbs)
    seq = torch.full((B,), L, dtype=torch.int32, device=dev)
    q = torch.randn(B * ql, Hq, D, device=dev, dtype=torch.float16)
    ref = torch.empty_like(q); got = torch.empty_like(q)
    qr = (q.float() @ fm).half()
    if varlen:
        qsl = torch.arange(0, B * ql + 1, ql, dtype=torch.int32, device=dev)
        E.exl3_spec_attn_varlen(qr, kc, vc, bt, seq, qsl, ref, 0.0625, 32)
        E.exl3_spec_attn_varlen(q, kc, vc, bt, seq, qsl, got, 0.0625, 32, None, None, signs)
    else:
        E.exl3_spec_attn(qr, kc, vc, bt, seq, ref, ql, 0.0625, 32)
        E.exl3_spec_attn(q, kc, vc, bt, seq, got, ql, 0.0625, 32, None, None, signs)
    ref = (ref.float() @ bm)
    ex = exact(q, kc, vc, bt, L, ql, 0.0625)
    e_host = ((ref - ex).norm() / ex.norm()).item(); e_rot = ((got.float() - ex).norm() / ex.norm()).item()
    rel = ((got.float() - ref).norm() / ref.norm()).item()
    ok = e_rot <= e_host * 1.05
    print(("PASS" if ok else "FAIL"), f"L={L} B={B} ql={ql} varlen={varlen} vs exact: host-rotation {e_host:.3e}, "
          f"in-kernel {e_rot:.3e}; host vs in-kernel {rel:.2e}", flush=True)
    if not ok: fails.append(L)
# HEAD-PACK path (ql=1, B=1, hp env) is the same kernel member; the merge handles both
print("all passed" if not fails else f"FAILED {fails}")
