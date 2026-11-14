"""Standalone LoRA delta op (superseded by the in-kernel path of llmt_lora_kernels.py, kept
for reference: 0.35-0.8 ms per decode step SLOWER than punica at 1-8 rows in graphs). API over llmt_lk.so.

    import lk_delta as lk
    if lk.available():
        lk.add_lora_linear(y, x, lora_a_stacked, lora_b_stacked, token_lora_mapping, output_slices, scale=1.0)

Semantics = PunicaWrapperXPU.add_lora_linear (add_inputs=True): for every row m with mapping[m] >= 0
    y[m, off_s:off_s + N_s] += scale * (x[m] @ A_s[mapping[m], 0].T) @ B_s[mapping[m], 0].T
with the slot buffers vLLM keeps (lora_a_stacked: tuple of [L, 1, R, K], lora_b_stacked: tuple of [L, 1, N_s, R], fp16).
Rows with mapping < 0 are not touched. Slices whose A tensors share storage (same data_ptr) are shrunk once.
Two device kernels per call, no host sync, no allocation once the workspace exists (call reserve() before capturing
graphs). Returns False (and does nothing) when the call is outside what the kernels support (dtype, rank not in
8/16/32/64, > 4 slices or distinct A tensors); the caller then uses punica.
"""
import os

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_SO = os.environ.get("LLMT_LK_SO", os.path.join(_HERE, "llmt_lk.so"))
_ok = None
MAX_ROWS = 64          # rows per kernel launch (larger M is split)


def available():
    global _ok
    if _ok is None:
        try:
            torch.ops.load_library(_SO)
            _ok = hasattr(torch.ops, "llmt_lk") and hasattr(torch.ops.llmt_lk, "lora_delta")
        except Exception:  # noqa: BLE001
            _ok = False
    return _ok


_plan = {}


def _plan_for(lora_a_stacked, lora_b_stacked, output_slices):
    key = (tuple(t.data_ptr() for t in lora_a_stacked), tuple(t.data_ptr() for t in lora_b_stacked), tuple(output_slices))
    p = _plan.get(key)
    if p is None:
        uniq, a_of, seen = [], [], {}
        for t in lora_a_stacked:
            # same storage AND same view = same A
            k = (t.data_ptr(), tuple(t.shape), tuple(t.stride()))
            if k not in seen:
                seen[k] = len(uniq)
                uniq.append(t)
            a_of.append(seen[k])
        off = [0]
        for n in output_slices:
            off.append(off[-1] + int(n))
        ok = (len(uniq) <= 4 and len(lora_b_stacked) <= 4 and all(t.dtype == torch.float16 for t in lora_a_stacked)
              and all(t.dtype == torch.float16 for t in lora_b_stacked) and lora_a_stacked[0].size(2) in (8, 16, 32, 64))
        # SLM per work-group: 2 x G x rows x R floats <= 48 KiB
        rows = max(1, min(MAX_ROWS, 6144 // (len(uniq) * int(lora_a_stacked[0].size(2)))))
        p = (ok, uniq, a_of, off, rows)
        _plan[key] = p
    return p


def add_lora_linear(y, x, lora_a_stacked, lora_b_stacked, token_lora_mapping, output_slices, scale=1.0):
    """In-place y += LoRA(x) for the rows with a LoRA (see the module doc). True if done, False = use punica."""
    if not available():
        return False
    ok, uniq, a_of, off, step = _plan_for(lora_a_stacked, lora_b_stacked, output_slices)
    if not ok or x.dtype != torch.float16 or y.dtype != torch.float16:
        return False
    x = x.view(-1, x.shape[-1])
    y = y.view(-1, y.shape[-1])
    M = x.size(0)
    idx = token_lora_mapping
    op = torch.ops.llmt_lk.lora_delta
    for m0 in range(0, M, step):
        m1 = min(M, m0 + step)
        op(y[m0:m1], x[m0:m1], uniq, list(lora_b_stacked), idx[m0:m1], a_of, off, float(scale))
    return True


def reserve(device, max_rows=MAX_ROWS, rank=16, k_max=17408, groups=4):
    """Size the per-device workspace (partials + h) for the largest call before graph capture."""
    if not available():
        return
    dev = torch.device(device)
    R = rank
    max_rows = min(max_rows, 6144 // (groups * rank))
    x = torch.zeros(max_rows, k_max, device=dev, dtype=torch.float16)
    y = torch.zeros(max_rows, 8, device=dev, dtype=torch.float16)
    a = [torch.zeros(1, 1, R, k_max, device=dev, dtype=torch.float16) for _ in range(groups)]
    b = [torch.zeros(1, 1, 2, R, device=dev, dtype=torch.float16) for _ in range(groups)]
    idx = torch.full((max_rows,), -1, device=dev, dtype=torch.int32)
    torch.ops.llmt_lk.lora_delta(y, x, a, b, idx, list(range(groups)), [0, 2, 4, 6, 8], 1.0)
    torch.xpu.synchronize(dev)
