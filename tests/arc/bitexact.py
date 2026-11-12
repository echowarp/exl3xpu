"""bit-exactness of exl3_spec_attn between two builds (separate processes): bitexact.py <so> <out.pt> | --cmp a.pt b.pt"""
import sys, torch
if sys.argv[1] == "--cmp":
    a, b = torch.load(sys.argv[2]), torch.load(sys.argv[3])
    for k in a:
        if k not in b: continue
        d = (a[k].float() - b[k].float()).abs().max().item()
        print(f"{k}: {'IDENTICAL' if torch.equal(a[k], b[k]) else f'DIFF max {d:.3e}'}")
    sys.exit(0)
torch.ops.load_library(sys.argv[1]); E = torch.ops.exl3xpu_C
dev = "xpu"; H, G, D = 4, 6, 256; Hq = H * G; res = {}
for kv, BS, ROW in (("int8", 1664, 260), ("int4", 3200, 132)):
    for L, B, ql, S in ((5000, 1, 8, 64), (40000, 2, 8, 32), (100000, 3, 8, 40), (30000, 1, 1, 64), (130000, 4, 4, 40)):
        g = torch.Generator(device="cpu").manual_seed(L + B)
        nbs = (L + BS - 1) // BS
        raw = torch.randint(0, 256, (nbs * B, BS, H, 2 * ROW), dtype=torch.uint8, generator=g)
        v = raw.view(-1, 2 * ROW); f = (torch.rand(v.shape[0], 1, generator=g) * 0.02 + 0.001).view(torch.uint8)
        v[:, ROW - 4:ROW] = f; v[:, 2 * ROW - 4:] = f
        raw = raw.to(dev); kvt = raw if kv == "int4" else raw.view(torch.int8)
        kc, vc = kvt[..., :ROW], kvt[..., ROW:]
        bt = torch.arange(nbs * B, dtype=torch.int32, device=dev).view(B, nbs)
        seq = torch.full((B,), L, dtype=torch.int32, device=dev)
        q = torch.randn(B * ql, Hq, D, generator=g).half().to(dev); out = torch.empty_like(q)
        try:
            E.exl3_spec_attn(q, kc, vc, bt, seq, out, ql, 0.0625, S)
        except RuntimeError:
            continue   # (build without int4 KV)
        res[f"{kv} L={L} B={B} ql={ql} S={S}"] = out.cpu()
torch.save(res, sys.argv[2])
