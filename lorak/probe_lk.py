#!/usr/bin/env python3
"""probe_lk.py: lab probe of the llmt_lk phases (shrink / expand / both) and K chunk, graph replay, synthetic data."""
import sys
import torch
sys.path.insert(0, __import__("os").path.dirname(__file__))
import test_lorak as T  # noqa: E402
lk = T.lk
dev = T.dev
lk.available(); lk.reserve(dev)
z = torch.zeros(16, device=dev)
sys.stdout.reconfigure(line_buffering=True)
print("empty-ish kernel (16-elem add_) in graph: %.2f us" % (T.graph_time(lambda: [z.add_(1) for _ in range(100)]) / 100))
for name in ("gdn.in_proj_ba", "gdn.out_proj", "mlp.gate_up_proj", "mlp.down_proj"):
    a, b, outs = T.make_layer(name, slots=1)
    for rows in (1, 4, 8):
        x = torch.randn(rows, T.TYPES[name][0], device=dev, dtype=T.dt)
        y = torch.zeros(rows, sum(outs), device=dev, dtype=T.dt)
        idx = T.mapping("one", rows)
        line = []
        for mask, ck in ((17, 256), (18, 256), (19, 256), (19, 512), (3, 256)):
            torch.ops.llmt_lk.lk_set(mask, ck)
            line.append("m%d/ck%d %5.1f" % (mask, ck, T.graph_time(lambda: [T.lkcall(y, x, a, b, idx, outs, 1.0) for _ in range(50)]) / 50))
        print(f"{name:17s} rows {rows}: " + "  ".join(line))
torch.ops.llmt_lk.lk_set(3, 256)
