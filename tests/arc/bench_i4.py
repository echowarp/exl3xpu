import sys, time, torch
torch.ops.load_library(sys.argv[1]); E = torch.ops.exl3xpu_C
ONLY4 = len(sys.argv) > 2
dev = "xpu"; H, G, D = 4, 6, 256; Hq = H * G
def timeit(f, n=50):
    for _ in range(5): f()
    torch.xpu.synchronize(); t = time.perf_counter()
    for _ in range(n): f()
    torch.xpu.synchronize(); return (time.perf_counter() - t) / n * 1e3
for L in (64000, 120000):
    for B, ql in ((1, 4), (2, 4), (3, 4), (4, 4)):
        q = torch.randn(B * ql, Hq, D, device=dev, dtype=torch.float16)
        out = torch.empty_like(q); seq = torch.full((B,), L, dtype=torch.int32, device=dev)
        res = {}
        for name, bs, row in ((("int4", 3200, 132),) if ONLY4 else (("int8", 1664, 260), ("int4", 3200, 132))):
            nbs = (L + bs - 1) // bs
            raw = torch.randint(0, 256, (nbs * B, bs, H, 2 * row), dtype=torch.uint8, device=dev)
            sc = raw.view(-1, 2 * row)[:, row - 4:row].contiguous()
            f = torch.full((raw.numel() // (2 * row), 2), 0.01, dtype=torch.float32, device=dev)
            v = raw.view(-1, 2 * row); v[:, row - 4:row] = f[:, :1].contiguous().view(torch.uint8); v[:, 2 * row - 4:] = f[:, 1:].contiguous().view(torch.uint8)
            kv = raw if name == "int4" else raw.view(torch.int8)
            kc, vc = kv[..., :row], kv[..., row:]
            bt = torch.arange(nbs * B, dtype=torch.int32, device=dev).view(B, nbs)
            res[name] = timeit(lambda: E.exl3_spec_attn(q, kc, vc, bt, seq, out, ql, 0.0625, 0, None, None))
        m = torch.randn(256, 256, device=dev)
        rot = timeit(lambda: (q.float() @ m).half())
        print(f"L={L} B={B} ql={ql}: " + ", ".join(f"{k} {v:.3f} ms" for k, v in res.items()), flush=True)
