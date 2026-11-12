import sys, time, torch
torch.ops.load_library(sys.argv[1]); E = torch.ops.exl3xpu_C
dev = "xpu"; H, G, D = 4, 6, 256; Hq = H * G
def timeit(f, n=40):
    for _ in range(5): f()
    torch.xpu.synchronize(); t = time.perf_counter()
    for _ in range(n): f()
    torch.xpu.synchronize(); return (time.perf_counter() - t) / n * 1e3
for L in (32000, 120000):
    for B in (1, 2, 3, 4):
        ql = 4
        q = torch.randn(B * ql, Hq, D, device=dev, dtype=torch.float16); out = torch.empty_like(q)
        seq = torch.full((B,), L, dtype=torch.int32, device=dev)
        row = []
        for name, bs, rw in (("int8", 1664, 260), ("int4", 3200, 132)):
            nbs = (L + bs - 1) // bs
            raw = torch.randint(0, 16, (nbs * B, bs, H, 2 * rw), dtype=torch.uint8, device=dev)
            v = raw.view(-1, 2 * rw); f = torch.full((v.shape[0], 1), 0.01, dtype=torch.float32, device=dev).view(torch.uint8)
            v[:, rw - 4:rw] = f; v[:, 2 * rw - 4:] = f
            kv = raw if name == "int4" else raw.view(torch.int8); kc, vc = kv[..., :rw], kv[..., rw:]
            bt = torch.arange(nbs * B, dtype=torch.int32, device=dev).view(B, nbs)
            for S in (32, 40, 64, 96, 128, 192, 256):
                try:
                    row.append(f"{name}/S{S} {timeit(lambda: E.exl3_spec_attn(q, kc, vc, bt, seq, out, ql, 0.0625, S, None, None)):.3f}")
                except Exception as e:
                    row.append(f"{name}/S{S} err")
        print(f"L={L} B={B}: " + "  ".join(row), flush=True)
