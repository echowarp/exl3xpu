#!/usr/bin/env python3
"""test_lorak.py (2026-11-13): correctness + microbenchmark of llmt_lk.lora_delta vs vLLM's XPU punica path
(bgmv_linear -> torch.ops._xpu_C.lora_linear) for the Qwen3.8-27B LoRA shapes (rank 16, fp16 slots).
Synthetic random weights and inputs only. Runs alone on the Arc (no engine), < 2 GiB device memory.

  test_lorak.py check        correctness vs fp32 reference: 7 layer types x rows 1..64 x mappings (1 adapter, LoRA rows
                             mixed with base rows (-1), 3 adapters), max abs error of lk and of punica
  test_lorak.py bench        per call: eager wall time (host incl.) and graph-replay time of 50 calls, rows 1..32
  test_lorak.py step         one decode step's 304 LoRA linears (distinct weights per layer, L2 cold-ish) captured in one
                             XPU graph, ms per step, punica vs lk, rows 1/2/4/6/8/16/24/32, mapping variants
"""
import sys
import time

import torch

sys.path.insert(0, __import__("os").path.dirname(__file__))
import lk_delta as lk  # noqa: E402

dev = torch.device("xpu")
dt = torch.float16
R = 16
L = 4          # adapter slots
TYPES = {   # name: (in, [slice outs], A sharing per slice)
    "gdn.in_proj_qkvz": (5120, [2048, 2048, 6144, 6144], [0, 0, 0, 1]),
    "gdn.in_proj_ba": (5120, [48, 48], [0, 1]),
    "gdn.out_proj": (6144, [5120], [0]),
    "attn.qkv_proj": (5120, [12288, 1024, 1024], [0, 1, 2]),
    "attn.o_proj": (6144, [5120], [0]),
    "mlp.gate_up_proj": (5120, [17408, 17408], [0, 1]),
    "mlp.down_proj": (17408, [5120], [0]),
}
COUNT = {"gdn.in_proj_qkvz": 48, "gdn.in_proj_ba": 48, "gdn.out_proj": 48, "attn.qkv_proj": 16, "attn.o_proj": 16,
         "mlp.gate_up_proj": 64, "mlp.down_proj": 64}
ORDER = []   # layer order of a decode step: 16 x (3 GDN layers + 1 attention layer), MLP after each
for _ in range(16):
    for _ in range(3):
        ORDER += ["gdn.in_proj_qkvz", "gdn.in_proj_ba", "gdn.out_proj", "mlp.gate_up_proj", "mlp.down_proj"]
    ORDER += ["attn.qkv_proj", "attn.o_proj", "mlp.gate_up_proj", "mlp.down_proj"]
assert len(ORDER) == 304


def make_layer(name, slots=L, scale_w=0.02):
    k, outs, share = TYPES[name]
    ua = [torch.randn(slots, 1, R, k, device=dev, dtype=dt) * scale_w for _ in range(max(share) + 1)]
    a = tuple(ua[s] for s in share)
    b = tuple(torch.randn(slots, 1, o, R, device=dev, dtype=dt) * scale_w for o in outs)
    return a, b, outs


def mapping(kind, rows):
    if kind == "one":
        m = [0] * rows
    elif kind == "mix":       # 1 LoRA request among base requests: first quarter LoRA (>= 1 row)
        n = max(1, rows // 4)
        m = [0] * n + [-1] * (rows - n)
    elif kind == "three":     # 3 adapters + base
        m = [(i * 4 // rows) if (i * 4 // rows) < 3 else -1 for i in range(rows)]
        m = [x if x != 2 else 3 for x in m]   # slots 0, 1, 3
    else:
        raise ValueError(kind)
    return torch.tensor(m, device=dev, dtype=torch.int32)


def ref_delta(y0, x, a, b, outs, idx, scale):
    ref = y0.float().clone()
    off = 0
    for ai, bi, o in zip(a, b, outs):
        for m in range(x.size(0)):
            s = int(idx[m])
            if s < 0:
                continue
            ref[m, off:off + o] += (x[m].float() @ ai[s, 0].float().t() @ bi[s, 0].float().t()) * scale
        off += o
    return ref


def punica(y, x, a, b, idx, outs, scale):
    import vllm.lora.ops.xpu_ops.lora_ops as X
    X.bgmv_linear(x, list(a), list(b), y, idx, list(outs), R, scaling=scale, add_inputs=True)


def lkcall(y, x, a, b, idx, outs, scale):
    assert lk.add_lora_linear(y, x, a, b, idx, outs, scale)


def check():
    torch.manual_seed(0)
    worst = {"lk": 0.0, "punica": 0.0}
    bad = 0
    for name in TYPES:
        a, b, outs = make_layer(name)
        k = TYPES[name][0]
        for rows in (1, 2, 3, 4, 6, 8, 16, 17, 32, 64, 80):
            x = torch.randn(rows, k, device=dev, dtype=dt)
            for kind in ("one", "mix", "three"):
                idx = mapping(kind, rows)
                y0 = torch.randn(rows, sum(outs), device=dev, dtype=dt)
                ref = ref_delta(y0, x, a, b, outs, idx.cpu(), 2.0)
                e, out = {}, {}
                for nm, f in (("lk", lkcall), ("punica", punica)):
                    y = y0.clone()
                    f(y, x, a, b, idx, outs, 2.0)
                    torch.xpu.synchronize()
                    out[nm] = y
                    e[nm] = (y.float() - ref).abs().max().item()
                    worst[nm] = max(worst[nm], e[nm])
                neg = idx < 0
                untouched = bool((out["lk"][neg] == y0[neg]).all())
                flag = "" if (e["lk"] <= max(4e-3, 2 * e["punica"]) and untouched) else "  <-- FAIL"
                bad += bool(flag)
                if flag or rows in (1, 8, 32):
                    print(f"{name:18s} rows {rows:3d} {kind:5s}  max err lk {e['lk']:.2e}  punica {e['punica']:.2e}"
                          f"  base rows untouched {untouched}{flag}")
    print(f"worst: lk {worst['lk']:.2e}  punica {worst['punica']:.2e}; failures {bad}")
    return bad


def timeit(fn, n=200):
    for _ in range(10):
        fn()
    torch.xpu.synchronize()
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    torch.xpu.synchronize()
    return (time.perf_counter() - t0) / n * 1e6


def graph_time(fn, reps=20):
    """us per fn() call, fn captured in one XPU graph (device time incl. kernel gaps, no host launch cost)."""
    for _ in range(3):
        fn()
    torch.xpu.synchronize()
    g = torch.xpu.XPUGraph()
    with torch.xpu.graph(g):
        fn()
    for _ in range(3):
        g.replay()
    torch.xpu.synchronize()
    t0 = time.perf_counter()
    for _ in range(reps):
        g.replay()
    torch.xpu.synchronize()
    return (time.perf_counter() - t0) / reps * 1e6


def bench():
    torch.manual_seed(0)
    lk.reserve(dev)
    print("type               rows | punica eager us | lk eager us | punica graph us/call | lk graph us/call")
    for name in TYPES:
        a, b, outs = make_layer(name, slots=1)
        k = TYPES[name][0]
        for rows in (1, 2, 4, 8, 16, 32):
            x = torch.randn(rows, k, device=dev, dtype=dt)
            y = torch.zeros(rows, sum(outs), device=dev, dtype=dt)
            idx = mapping("one", rows)
            res = []
            for f in (punica, lkcall):
                res.append(timeit(lambda: f(y, x, a, b, idx, outs, 1.0)))
            for f in (punica, lkcall):
                res.append(graph_time(lambda: [f(y, x, a, b, idx, outs, 1.0) for _ in range(50)]) / 50)
            print(f"{name:18s} {rows:4d} | {res[0]:8.1f} | {res[1]:8.1f} | {res[2]:8.1f} | {res[3]:8.1f}")


def step():
    torch.manual_seed(0)
    lk.reserve(dev)
    layers = {n: [make_layer(n, slots=2) for _ in range(COUNT[n])] for n in TYPES}
    mb = sum(t.numel() * 2 for n in layers for (a, b, _) in layers[n] for t in set(a) | set(b)) / 2**20
    print(f"step: 304 LoRA linears, distinct weights per layer, 2 slots ({mb:.0f} MiB)")
    print("rows kind  | punica ms/step | lk ms/step | saved ms")
    for rows in (1, 2, 4, 6, 8, 16, 24, 32):
        xs = {n: torch.randn(rows, TYPES[n][0], device=dev, dtype=dt) for n in TYPES}
        ys = {n: torch.zeros(rows, sum(TYPES[n][1]), device=dev, dtype=dt) for n in TYPES}
        for kind in ("one", "mix", "three"):
            idx = mapping(kind, rows).clamp(max=1)    # 2 slots here
            res = []
            for f in (punica, lkcall):
                ctr = {n: 0 for n in TYPES}

                def run():
                    for n in ctr:
                        ctr[n] = 0
                    for n in ORDER:
                        a, b, outs = layers[n][ctr[n]]
                        ctr[n] += 1
                        f(ys[n], xs[n], a, b, idx, outs, 1.0)
                res.append(graph_time(run, reps=10) / 1e3)
            print(f"{rows:4d} {kind:5s} | {res[0]:8.3f} | {res[1]:8.3f} | {res[0] - res[1]:8.3f}")
            if rows == 1:
                break


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "check"
    print("lk available:", lk.available())
    if what == "check":
        sys.exit(1 if check() else 0)
    elif what == "bench":
        bench()
    elif what == "step":
        step()
