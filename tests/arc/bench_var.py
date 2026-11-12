"""int4 decode-attention variants: whole attention call as the plugin runs it, production split
table (EXL3_I8_SPLITS 64/32/40: B=1 64, B=2 32, B>=3 40), verify rows ql=8 (k=7) and drafter ql=1 (HEAD-PACK at B=1).
  bench_var.py base_C.so variant_C.so [--rot-in-kernel for the variant]
'host' = q.float() @ fm -> half, kernel, o.float() @ bm -> half -> copy (the i4 plugin); 'kern' = kernel with rot=signs."""
import os, sys, time, torch
dev = "xpu"; H, G, D, BS = 4, 6, 256, 3200; Hq = H * G
SPLITS = {1: 64, 2: 32, 3: 40, 4: 40}
os.environ.setdefault("EXL3_SPEC_HP", "1"); os.environ.setdefault("EXL3_SPEC_HP_BMAX", "8")
libs = sys.argv[1:3]
from vllm.v1.attention.ops.int4_per_token_head import _get_rht_signs
h = torch.ones(1, 1)
while h.shape[0] < 256: h = torch.cat([torch.cat([h, h], 1), torch.cat([h, -h], 1)], 0)
sg = _get_rht_signs(256, 0, torch.device("cpu")).float()
m = (sg[:, None] * (h / 16.0)).to(dev); fm, bm = m / 16.0, m.t().contiguous() / 16.0
signs = sg.contiguous().to(dev)


def timeit(f, n=int(os.environ.get("BENCH_N", "40"))):
    for _ in range(5): f()
    torch.xpu.synchronize(); t = time.perf_counter()
    for _ in range(n): f()
    torch.xpu.synchronize(); return (time.perf_counter() - t) / n * 1e3


def run(lib_idx, rot):
    torch.ops.load_library(libs[lib_idx])
    E = torch.ops.exl3xpu_C
    return E


# one process per library (ops of the same name can't be loaded twice): re-exec per lib
if len(sys.argv) > 3 and sys.argv[3] == "--child":
    lib, mode = sys.argv[1], sys.argv[2]
    torch.ops.load_library(lib); E = torch.ops.exl3xpu_C
    torch.manual_seed(0)
    for L in (32000, 64000, 120000, 190000):
        for ql, Bs in ((8, (1, 2, 3, 4)), (1, (1, 3))):
            for B in Bs:
                nbs = (L + BS - 1) // BS
                raw = torch.randint(0, 256, (nbs * B, BS, H, 264), dtype=torch.uint8, device=dev)
                v = raw.view(-1, 264)
                f = torch.full((v.shape[0], 1), 0.01, dtype=torch.float32, device=dev).view(torch.uint8)
                v[:, 128:132] = f; v[:, 260:264] = f
                kc, vc = raw[..., :132], raw[..., 132:]
                bt = torch.arange(nbs * B, dtype=torch.int32, device=dev).view(B, nbs)
                seq = torch.full((B,), L, dtype=torch.int32, device=dev)
                q = torch.randn(B * ql, Hq, D, device=dev, dtype=torch.float16)
                out = torch.empty_like(q); o_r = torch.empty_like(q)
                S = SPLITS[min(B, 4)]
                if mode == "host":
                    def f_():
                        q_r = (q.float() @ fm).half()
                        E.exl3_spec_attn(q_r, kc, vc, bt, seq, o_r, ql, 0.0625, S)
                        out.copy_((o_r.float() @ bm).half())
                else:
                    def f_():
                        E.exl3_spec_attn(q, kc, vc, bt, seq, out, ql, 0.0625, S, None, None, signs)
                print(f"{L} {ql} {B} {timeit(f_):.4f}", flush=True)
                del raw
    sys.exit(0)

import subprocess
res = {}
REPS = int(os.environ.get("BENCH_REPS", "1"))
for lib, mode in [(libs[0], "host"), (libs[1], sys.argv[3] if len(sys.argv) > 3 else "kern")] * REPS:   # ABAB..., min per cell
    p = subprocess.run([sys.executable, __file__, lib, mode, "--child"], capture_output=True, text=True)
    for line in p.stdout.splitlines():
        L, ql, B, t = line.split()
        res.setdefault((int(L), int(ql), int(B)), {}).setdefault(mode, []).append(float(t))
    if p.returncode: print(p.stderr[-2000:])
print(f"{'L':>7} {'rows':>4} {'B':>2} {'base ms':>8} {'var ms':>8} {'delta':>7}")
for k, v in sorted(res.items()):
    if len(v) == 2:
        b0, b1 = min(v["host"]), min(v["kern"])
        print(f"{k[0]:7d} {k[1]:4d} {k[2]:2d} {b0:8.3f} {b1:8.3f} {100 * (b1 / b0 - 1):+6.1f}%   (runs {len(v['host'])}+{len(v['kern'])}, min)")
