"""int4 decode attention: kernel time vs split count / variant. Kernel only (rot in-kernel when
the .so supports it), random int4 cache, min over REPS of N iterations.
  sweep.py <so> [--lens 32000,120000,190000] [--bs 1,2,3,4] [--rows 8] [--splits 16,24,32,40,42,48,64,96,128]"""
import argparse, os, time, torch
ap = argparse.ArgumentParser(); ap.add_argument("so")
ap.add_argument("--lens", default="32000,120000,190000"); ap.add_argument("--bs", default="1,2,3,4")
ap.add_argument("--rows", type=int, default=8); ap.add_argument("--splits", default="16,24,32,40,42,48,64,96,128")
ap.add_argument("--kv", default="int4"); ap.add_argument("--n", type=int, default=60); ap.add_argument("--reps", type=int, default=3)
a = ap.parse_args()
torch.ops.load_library(a.so); E = torch.ops.exl3xpu_C
dev = "xpu"; H, G, D = 4, 6, 256; Hq = H * G
BS, ROW = (3200, 132) if a.kv == "int4" else (1664, 260)
rot = "rot" in [x.name for x in E.exl3_spec_attn.default._schema.arguments]
signs = (torch.randint(0, 2, (256,)) * 2 - 1).float().to(dev) if rot else None
def timeit(f):
    best = 1e9
    for _ in range(a.reps):
        for _ in range(3): f()
        torch.xpu.synchronize(); t = time.perf_counter()
        for _ in range(a.n): f()
        torch.xpu.synchronize(); best = min(best, (time.perf_counter() - t) / a.n * 1e3)
    return best
for L in map(int, a.lens.split(",")):
    for B in map(int, a.bs.split(",")):
        nbs = (L + BS - 1) // BS
        raw = torch.randint(0, 256, (nbs * B, BS, H, 2 * ROW), dtype=torch.uint8, device=dev)
        v = raw.view(-1, 2 * ROW); f = torch.full((v.shape[0], 1), 0.01, dtype=torch.float32, device=dev).view(torch.uint8)
        v[:, ROW - 4:ROW] = f; v[:, 2 * ROW - 4:] = f
        kv = raw if a.kv == "int4" else raw.view(torch.int8)
        kc, vc = kv[..., :ROW], kv[..., ROW:]
        bt = torch.arange(nbs * B, dtype=torch.int32, device=dev).view(B, nbs)
        seq = torch.full((B,), L, dtype=torch.int32, device=dev)
        q = torch.randn(B * a.rows, Hq, D, device=dev, dtype=torch.float16); out = torch.empty_like(q)
        row = []
        for S in map(int, a.splits.split(",")):
            args = (q, kc, vc, bt, seq, out, a.rows, 0.0625, S) + ((None, None, signs) if rot and a.kv == "int4" else ())
            t = timeit(lambda: E.exl3_spec_attn(*args))
            row.append(f"S{S}={t:.3f}")
        gbs = L * B * H * 2 * ROW / 1e9
        print(f"L={L} B={B} rows={a.rows}: " + " ".join(row) + f"  (KV {gbs * 1e3:.0f} MB)", flush=True)
        del raw
