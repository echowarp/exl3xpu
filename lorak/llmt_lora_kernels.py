"""LoRA decode kernels for the Arc B70: hooks for the AllBroker lend manager (EXL3_LORA_KERNELS=llmt_lora_kernels).

The LoRA terms run inside the kernels the EXL3 base linear launches anyway (exl3xpu_C ops of branch lora-kernels,
csrc/lora_esimd.h), so a LoRA linear costs the base linear's 3 launches instead of 3 + punica's 1 (4-10 us each in
graphs, 304 per step):
  linear_full(layer, x)  -> exl3xpu_C.linear_lora: base EXL3 small-M linear + LoRA for the rows with a slot
  mlp(module, x)         -> exl3xpu_C.mlp_small_lora: the fused exl3 decode MLP (3..8 rows) + gate_up LoRA before the
                            activation + down LoRA, so LoRA steps keep the fused decode MLP
Both return None (llmt_lora's current path runs) whenever the call is outside what the kernels do: engine _C.so
without the ops, rank != 16, fp16 only, M above the small-M path, slices not multiples of 128 (in_proj_ba), bias.
Graph-safe: no host syncs; allocations depend on shapes only. Base rows (slot -1) run the base kernels' arithmetic.
Knobs: LLMT_LK_LINEAR (1), LLMT_LK_MLP (1).
"""
import os

import torch

_ENV = os.environ.get
LINEAR = _ENV("LLMT_LK_LINEAR", "1") == "1"
MLP = _ENV("LLMT_LK_MLP", "1") == "1"
_N = {"linear": 0, "linear_skip": 0, "mlp": 0, "mlp_skip": 0}
_plan = {}


def _ops():
    C = getattr(torch.ops, "exl3xpu_C", None)
    return C if (C is not None and hasattr(C, "linear_lora") and hasattr(C, "mlp_small_lora")) else None


def _small_m_max():
    try:
        from exl3xpu import ops
        return int(ops.SMALL_M_MAX)
    except Exception:  # noqa: BLE001
        return 128


def _layer_plan(layer):
    """Static per-layer facts (cached on the layer): exl3 base attributes, slice offsets, eligibility."""
    p = getattr(layer, "_llmt_lk_plan", None)
    if p is not None:
        return p
    b = getattr(layer, "base_layer", None)
    ok = (b is not None and getattr(b, "exl3_cpp", False) and getattr(b, "bias", None) is None)
    off = [0]
    for n in getattr(layer, "output_slices", ()):
        off.append(off[-1] + int(n))
    if ok:
        ok = (len(off) >= 2 and len(off) <= 5 and all(o % 128 == 0 for o in off) and off[-1] == int(b.svh.shape[0])
              and int(b.trellis.shape[0]) > 0)
        try:
            ok = ok and bool(torch.ops.exl3xpu_C.exl3_supported(int(b.exl3_K), int(b.exl3_cb)))
        except Exception:  # noqa: BLE001
            ok = False
    p = (ok, off)
    layer._llmt_lk_plan = p
    return p


def _slots_ok(a, b):
    t = a[0]
    return (t.device.type == "xpu" and t.dtype == torch.float16 and t.dim() == 4 and t.size(2) == 16
            and all(u.dtype == torch.float16 for u in b))


def linear_full(layer, x):
    if not LINEAR or x.dim() != 2 or x.dtype != torch.float16:
        return None
    ok, off = _layer_plan(layer)
    M = x.size(0)
    if not ok or M == 0 or M > _small_m_max():
        _N["linear_skip"] += 1
        return None
    a, bb = layer.lora_a_stacked, layer.lora_b_stacked
    if not _slots_ok(a, bb) or len(a) != len(off) - 1:
        _N["linear_skip"] += 1
        return None
    base = layer.base_layer
    idx = layer.punica_wrapper._token_lora_indices[:M]
    uniq, a_of = _a_groups(layer, a)
    _N["linear"] += 1
    return torch.ops.exl3xpu_C.linear_lora(x, base.trellis, base.suh, base.svh, base.exl3_shard_of_nb,
                                           int(base.exl3_K), int(base.exl3_cb), uniq, list(bb), idx, a_of, off, 1.0)


def _capturing():
    f = getattr(torch.xpu, "is_current_stream_capturing", None)
    try:
        return True if f is None else bool(f())
    except Exception:  # noqa: BLE001
        return True


def _a_groups(layer, a):
    """Distinct A tensors + slice -> A map. vLLM's expand_packed_lora copies one checkpoint A into every slice its
    group covers (in_proj_qkv -> the q, k, v slices of in_proj_qkvz), so equal slices stay equal for any adapter: decided
    once per layer by comparing the slot buffers (first call outside graph capture; before that: no dedup, still exact)."""
    g = getattr(layer, "_llmt_lk_ag", None)
    if g is not None and g[0] == tuple(t.data_ptr() for t in a):
        return [a[i] for i in g[1]], g[2]
    if len(a) == 1:
        res = ([0], [0])
    elif _capturing():
        return list(a), list(range(len(a)))
    else:
        # an all-zero slice matches anything: without data in every slice, keep the exact no-dedup plan (cached)
        reps, a_of = [], []
        if not all(bool(t.any()) for t in a):
            reps, a_of = list(range(len(a))), list(range(len(a)))
        for i, t in enumerate(a if not reps else []):
            for j, r in enumerate(reps):
                if a[r].shape == t.shape and torch.equal(a[r], t):
                    a_of.append(j)
                    break
            else:
                reps.append(i)
                a_of.append(len(reps) - 1)
        res = (reps, a_of)
    layer._llmt_lk_ag = (tuple(t.data_ptr() for t in a), res[0], res[1])
    return [a[i] for i in res[0]], res[1]


def mlp(module, x):
    if not MLP or x.dim() != 2 or x.dtype != torch.float16 or getattr(module, "expert_gate", None) is not None:
        return None
    M = x.size(0)
    if M < 3 or M > 8:
        return None
    gu, dn = module.gate_up_proj, module.down_proj
    gb, db = getattr(gu, "base_layer", None), getattr(dn, "base_layer", None)
    if gb is None or db is None:
        return None
    key = id(module)
    p = _plan.get(key)
    if p is None:
        p = (getattr(gb, "exl3_cpp", False) and getattr(db, "exl3_cpp", False) and gb.exl3_cb == db.exl3_cb == 2
             and int(gb.exl3_K) in (4, 6) and int(db.exl3_K) in (4, 6) and gb.suh.shape[0] <= 2 and db.suh.shape[0] == 1
             and gb.svh.shape[0] % 256 == 0 and getattr(gb, "bias", None) is None and getattr(db, "bias", None) is None
             and len(gu.lora_a_stacked) == 2 and len(dn.lora_a_stacked) == 1)
        _plan[key] = p
    if not p:
        _N["mlp_skip"] += 1
        return None
    ga, gbb, da, dbb = gu.lora_a_stacked, gu.lora_b_stacked, dn.lora_a_stacked, dn.lora_b_stacked
    if not (_slots_ok(ga, gbb) and _slots_ok(da, dbb)):
        _N["mlp_skip"] += 1
        return None
    idx = gu.punica_wrapper._token_lora_indices[:M]
    _N["mlp"] += 1
    return torch.ops.exl3xpu_C.mlp_small_lora(x, gb.trellis, gb.suh, gb.svh, gb.exl3_shard_of_nb,
                                              db.trellis, db.suh, db.svh, db.exl3_shard_of_nb,
                                              int(gb.exl3_K), int(db.exl3_K), list(ga), list(gbb), [0, 1],
                                              list(da), list(dbb), idx, 1.0)


def counts():
    return dict(_N)


def install(llmt_lora):
    if _ops() is None:
        raise RuntimeError("exl3xpu_C has no linear_lora / mlp_small_lora (engine runs a _C.so without the LoRA ops)")
    llmt_lora.set_hooks(linear=None, mlp=mlp if MLP else None, linear_full=linear_full if LINEAR else None)
