#!/usr/bin/env python3
"""test_lora_c.py (2026-11-13): the in-kernel LoRA ops of branch lora-kernels (exl3xpu_C.linear_lora,
exl3xpu_C.mlp_small_lora) on REAL Qwen3.8-27B EXL3 layers with synthetic rank-16 adapters. Standalone on the Arc.
Loads lorak/_C_lora.so (not the production _C.so). No request data.

  check : base rows (slot -1) bit-identical to exl3xpu_C.linear / mlp_auto; LoRA rows vs a reference built from the
          base op + fp32 LoRA (the punica arithmetic: fp16 base + delta), max abs error of both vs fp32
  bench : XPU-graph us per call: base linear | base + punica (today's LoRA path) | linear_lora; MLP: mlp_auto |
          plain (linear + punica + act + linear + punica) | mlp_small_lora
"""
import json
import os
import sys
import time

import torch
from safetensors import safe_open

HERE = os.path.dirname(os.path.abspath(__file__))
torch.ops.load_library(os.environ.get("LK_C", os.path.join(HERE, "_C_lora.so")))
C = torch.ops.exl3xpu_C
MODEL = os.environ.get("MODEL", "/models/turboderp-Qwen3.8-27B-exl3-4.00bpw")
dev = torch.device("xpu")
dt = torch.float16
R = 16
L = 4
SMALL_M_MAX, SLICE_N = 128, 16384
widx = json.load(open(f"{MODEL}/model.safetensors.index.json"))["weight_map"]
P = "model.language_model.layers"


def load(key):
    t = {}
    for sub in ("trellis", "suh", "svh", "mul1", "mcg"):
        name = f"{key}.{sub}"
        if name in widx:
            with safe_open(f"{MODEL}/{widx[name]}", "pt", device="cpu") as f:
                t[sub] = f.get_tensor(name)
    return t


class Lin:
    """packed exl3 linear as vllm_plugin builds it: trellis cat along n, suh [groups, k], svh cat, shard_of_nb, bounds"""
    def __init__(self, keys, groups=None):
        ts = [load(k) for k in keys]
        self.K = ts[0]["trellis"].shape[-1] // 16
        self.cb = 2 if "mul1" in ts[0] else (1 if "mcg" in ts[0] else 0)
        self.sizes = [t["svh"].shape[0] for t in ts]
        self.trellis = torch.cat([t["trellis"] for t in ts], 1).contiguous().to(dev)
        self.svh = torch.cat([t["svh"] for t in ts]).to(dev)
        groups = groups or [[i] for i in range(len(ts))]
        offs = [0]
        for s in self.sizes:
            offs.append(offs[-1] + s)
        self.suh = torch.stack([ts[g[0]]["suh"] for g in groups]).contiguous().to(dev)
        self.bounds = [offs[g[0]] for g in groups] + [offs[-1]]
        sh = torch.empty(offs[-1] // 128, dtype=torch.int32)
        for gi in range(len(groups)):
            sh[self.bounds[gi] // 128: self.bounds[gi + 1] // 128] = gi
        self.shard = sh.to(dev)
        self.k = self.suh.shape[1]
        self.n = offs[-1]

    def __call__(self, x):
        return C.linear(x, self.trellis, self.suh, self.svh, self.shard, self.bounds, self.K, self.cb, SMALL_M_MAX, SLICE_N)


def adapter(k, slices, scale_w=0.02):
    a = [torch.randn(L, 1, R, k, device=dev, dtype=dt) * scale_w for _ in slices]
    b = [torch.randn(L, 1, n, R, device=dev, dtype=dt) * scale_w for n in slices]
    return a, b


def mapping(kind, rows):
    if kind == "one":
        m = [0] * rows
    elif kind == "mix":
        n = max(1, rows // 4)
        m = [0] * n + [-1] * (rows - n)
    elif kind == "three":
        m = [[0, 1, 3, -1][min(3, i * 4 // rows)] for i in range(rows)]
    else:   # none
        m = [-1] * rows
    return torch.tensor(m, device=dev, dtype=torch.int64)


def delta(x, a, b, idx):
    """fp32 LoRA term [M, sum n]"""
    outs = []
    xf = x.float()
    for ai, bi in zip(a, b):
        d = torch.zeros(x.size(0), bi.size(2), device=dev)
        for m in range(x.size(0)):
            s = int(idx[m])
            if s >= 0:
                d[m] = xf[m] @ ai[s, 0].float().t() @ bi[s, 0].float().t()
        outs.append(d)
    return torch.cat(outs, 1)


def offs(slices):
    o = [0]
    for s in slices:
        o.append(o[-1] + s)
    return o


def lin_lora(lin, x, a, b, idx):
    return C.linear_lora(x, lin.trellis, lin.suh, lin.svh, lin.shard, lin.K, lin.cb, a, b, idx,
                         list(range(len(a))), offs([t.size(2) for t in b]), 1.0)


def layers():
    return {
        "gdn.in_proj_qkvz": (Lin([f"{P}.0.linear_attn.in_proj_qkv", f"{P}.0.linear_attn.in_proj_z"]), [2048, 2048, 6144, 6144]),
        "gdn.out_proj": (Lin([f"{P}.0.linear_attn.out_proj"]), [5120]),
        "attn.qkv_proj": (Lin([f"{P}.3.self_attn.q_proj", f"{P}.3.self_attn.k_proj", f"{P}.3.self_attn.v_proj"]), None),
        "attn.o_proj": (Lin([f"{P}.3.self_attn.o_proj"]), [5120]),
        "mlp.gate_up_proj": (Lin([f"{P}.0.mlp.gate_proj", f"{P}.0.mlp.up_proj"]), None),
        "mlp.down_proj": (Lin([f"{P}.0.mlp.down_proj"]), [5120]),
    }


def check():
    torch.manual_seed(0)
    bad = 0
    for name, (lin, slices) in layers().items():
        slices = slices or lin.sizes
        assert sum(slices) == lin.n, (name, slices, lin.n)
        a, b = adapter(lin.k, slices)
        if name == "gdn.in_proj_qkvz":    # vLLM: in_proj_qkv's A copied into the q, k, v slices
            a = [a[0], a[0].clone(), a[0].clone(), a[3]]
        for rows in (1, 2, 3, 4, 6, 8, 16, 32):
            x = torch.randn(rows, lin.k, device=dev, dtype=dt)
            base = lin(x)
            if name == "gdn.in_proj_qkvz":
                idx = mapping("three", rows)
                y4 = lin_lora(lin, x, a, b, idx)
                y2 = C.linear_lora(x, lin.trellis, lin.suh, lin.svh, lin.shard, lin.K, lin.cb, [a[0], a[3]], b, idx,
                                   [0, 0, 0, 1], offs(slices), 1.0)
                same = bool((y4 == y2).all())
                bad += not same
                if not same or rows in (1, 8):
                    print(f"{name:17s} rows {rows:2d} deduped A (2 of 4) bit-identical to 4 A: {same}")
            for kind in ("none", "mix", "one", "three"):
                idx = mapping(kind, rows)
                y = lin_lora(lin, x, a, b, idx)
                neg = idx < 0
                ident = bool((y[neg] == base[neg]).all())
                d = delta(x, a, b, idx)
                ref32 = base.float() + d
                pun = (base.float() + d).half()   # punica: fp16 base + delta, one rounding
                e = (y.float() - ref32).abs().max().item()
                ep = (pun.float() - ref32).abs().max().item()
                ok = ident and e <= max(2 * ep, 2e-3)
                bad += not ok
                if not ok or rows in (1, 8):
                    print(f"{name:17s} rows {rows:2d} {kind:5s} base rows bit-identical {ident}  max err {e:.2e} "
                          f"(punica-style {ep:.2e}){'' if ok else '  <-- FAIL'}")
    # MLP
    gu, dnl = Lin([f"{P}.0.mlp.gate_proj", f"{P}.0.mlp.up_proj"]), Lin([f"{P}.0.mlp.down_proj"])
    I = gu.n // 2
    ga, gb = adapter(gu.k, [I, I])
    da, db = adapter(I, [dnl.n])
    for rows in (3, 4, 6, 8):
        x = torch.randn(rows, gu.k, device=dev, dtype=dt)
        base = C.mlp_auto(x, gu.trellis, gu.suh, gu.svh, gu.shard, gu.bounds, dnl.trellis, dnl.suh, dnl.svh, dnl.shard,
                          dnl.bounds, gu.K, dnl.K, gu.cb, SMALL_M_MAX, SLICE_N)
        for kind in ("none", "mix", "one", "three"):
            idx = mapping(kind, rows)
            y = C.mlp_small_lora(x, gu.trellis, gu.suh, gu.svh, gu.shard, dnl.trellis, dnl.suh, dnl.svh, dnl.shard,
                                 gu.K, dnl.K, ga, gb, [0, 1], da, db, idx, 1.0)
            neg = idx < 0
            ident = bool((y[neg] == base[neg]).all())
            # plain path reference (what the LoRA step runs today): fp16 gate_up + delta, act fp16, down + delta
            g = (gu(x).float() + delta(x, ga, gb, idx)).half().float()
            act = (torch.nn.functional.silu(g[:, :I]) * g[:, I:]).half()
            ref = dnl(act).float() + delta(act, da, db, idx)
            e = (y.float() - ref).abs().max().item()
            eb = (base.float() - (dnl((torch.nn.functional.silu(gu(x).float()[:, :I]) * gu(x).float()[:, I:]).half()).float())).abs().max().item()
            rel = e / max(ref.abs().max().item(), 1e-6)
            ok = ident and e <= max(4 * eb, 4e-3)
            bad += not ok
            print(f"mlp_small_lora    rows {rows:2d} {kind:5s} base rows bit-identical {ident}  max err vs plain path {e:.2e}"
                  f" (rel {rel:.1e}; fused-vs-plain base noise {eb:.2e}){'' if ok else '  <-- FAIL'}")
    print("failures:", bad)
    return bad


def graph_us(fn, n=20, reps=10):
    for _ in range(3):
        fn()
    torch.xpu.synchronize()
    g = torch.xpu.XPUGraph()
    with torch.xpu.graph(g):
        for _ in range(n):
            fn()
    g.replay()
    torch.xpu.synchronize()
    t0 = time.perf_counter()
    for _ in range(reps):
        g.replay()
    torch.xpu.synchronize()
    return (time.perf_counter() - t0) / reps / n * 1e6


def bench():
    import vllm.lora.ops.xpu_ops.lora_ops as X
    torch.manual_seed(0)
    print("layer             rows | base us | base+punica us | linear_lora us | saved us/call")
    for name, (lin, slices) in layers().items():
        slices = slices or lin.sizes
        a, b = adapter(lin.k, slices)
        a1 = [t[:1].contiguous() for t in a]
        b1 = [t[:1].contiguous() for t in b]
        for rows in (1, 4, 8, 16):
            x = torch.randn(rows, lin.k, device=dev, dtype=dt)
            idx = mapping("one", rows)
            idx32 = idx.to(torch.int32)

            def pun():
                y = lin(x)
                X.bgmv_linear(x, a1, b1, y, idx32, list(slices), R, scaling=1.0, add_inputs=True)
            t = [graph_us(lambda: lin(x)), graph_us(pun), graph_us(lambda: lin_lora(lin, x, a1, b1, idx))]
            print(f"{name:17s} {rows:4d} | {t[0]:7.1f} | {t[1]:7.1f} | {t[2]:7.1f} | {t[1] - t[2]:6.1f}")
    gu, dnl = Lin([f"{P}.0.mlp.gate_proj", f"{P}.0.mlp.up_proj"]), Lin([f"{P}.0.mlp.down_proj"])
    I = gu.n // 2
    ga, gb = adapter(gu.k, [I, I])
    da, db = adapter(I, [dnl.n])
    ga, gb, da, db = ([t[:1].contiguous() for t in v] for v in (ga, gb, da, db))
    print("MLP rows | mlp_auto (prod, no LoRA) | plain + punica (today's LoRA step) | mlp_small_lora | saved us")
    for rows in (3, 4, 6, 8):
        x = torch.randn(rows, gu.k, device=dev, dtype=dt)
        idx = mapping("one", rows)
        idx32 = idx.to(torch.int32)

        def auto():
            return C.mlp_auto(x, gu.trellis, gu.suh, gu.svh, gu.shard, gu.bounds, dnl.trellis, dnl.suh, dnl.svh,
                              dnl.shard, dnl.bounds, gu.K, dnl.K, gu.cb, SMALL_M_MAX, SLICE_N)

        def plain():
            g = gu(x)
            X.bgmv_linear(x, ga, gb, g, idx32, [I, I], R, scaling=1.0, add_inputs=True)
            act = (torch.nn.functional.silu(g[:, :I].float()) * g[:, I:].float()).half()
            y = dnl(act)
            X.bgmv_linear(act, da, db, y, idx32, [dnl.n], R, scaling=1.0, add_inputs=True)

        def lk():
            return C.mlp_small_lora(x, gu.trellis, gu.suh, gu.svh, gu.shard, dnl.trellis, dnl.suh, dnl.svh, dnl.shard,
                                    gu.K, dnl.K, ga, gb, [0, 1], da, db, idx, 1.0)
        t = [graph_us(auto), graph_us(plain), graph_us(lk)]
        print(f"MLP {rows:4d} | {t[0]:7.1f} | {t[1]:7.1f} | {t[2]:7.1f} | {t[1] - t[2]:6.1f}")


def sums():
    """hashes of BASE outputs (C.linear, C.mlp_auto) for fixed inputs: run with LK_C=<production _C.so> and with
    _C_lora.so; equal hashes = this build reproduces production's base kernels bit for bit"""
    import hashlib
    torch.manual_seed(1)
    L_ = layers()
    for name, (lin, _) in L_.items():
        for rows in (1, 4, 8, 16, 64):
            x = torch.randn(rows, lin.k, device=dev, dtype=dt)
            print(name, rows, hashlib.sha1(lin(x).cpu().numpy().tobytes()).hexdigest()[:16])
    gu, dnl = L_["mlp.gate_up_proj"][0], L_["mlp.down_proj"][0]
    for rows in (3, 4, 6, 8):
        x = torch.randn(rows, gu.k, device=dev, dtype=dt)
        y = C.mlp_auto(x, gu.trellis, gu.suh, gu.svh, gu.shard, gu.bounds, dnl.trellis, dnl.suh, dnl.svh, dnl.shard,
                       dnl.bounds, gu.K, dnl.K, gu.cb, SMALL_M_MAX, SLICE_N)
        print("mlp_auto", rows, hashlib.sha1(y.cpu().numpy().tobytes()).hexdigest()[:16])


if __name__ == "__main__":
    sys.stdout.reconfigure(line_buffering=True)
    w = sys.argv[1] if len(sys.argv) > 1 else "check"
    if w == "check":
        sys.exit(1 if check() else 0)
    if w == "sums":
        sums()
        sys.exit(0)
    bench()
