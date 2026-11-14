"""CPU-only checks of the exl3xpu plugin pieces that need no Arc GPU: the no-mmap loader reads safetensors exactly,
the plugin prefers AllBroker's llmt_kv copy of that loader, the vLLM entry point is declared, the LoRA-kernel hook
module degrades to None without its ops, and env-switched features default off.

    python -m pytest -q tests/cpu/test_cpu_plugin.py      (from the repository root)
"""
import ast
import importlib.util
import json
import os
import struct
import sys
import tomllib

import torch

try:
    import pytest
except ImportError:                      # the Arc image has no pytest: run this file directly (see __main__)
    class _Skip(Exception):
        pass

    class pytest:                        # noqa: N801
        @staticmethod
        def skip(msg):
            raise _Skip(msg)

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _write_st(path, tensors):
    hdr, blobs, off = {}, [], 0
    names = {torch.float16: "F16", torch.bfloat16: "BF16", torch.float32: "F32", torch.int32: "I32", torch.uint8: "U8"}
    for k, t in tensors.items():
        b = t.contiguous().view(torch.uint8).numpy().tobytes()
        hdr[k] = {"dtype": names[t.dtype], "shape": list(t.shape), "data_offsets": [off, off + len(b)]}
        blobs.append(b)
        off += len(b)
    h = json.dumps(hdr).encode()
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(h)) + h + b"".join(blobs))


def test_nommap_roundtrip(tmp_path):
    nm = _load("exl3xpu_nommap_t", os.path.join(ROOT, "exl3xpu", "nommap_loader.py"))
    g = torch.Generator().manual_seed(0)
    want = {"a.weight": torch.randn(7, 5, generator=g).half(), "b": torch.arange(11, dtype=torch.int32),
            "c": torch.randn(3, 4, generator=g).bfloat16(), "d": torch.randn(2, generator=g)}
    p = tmp_path / "m.safetensors"
    _write_st(p, want)
    got = dict(nm.iter_file(str(p)))
    assert set(got) == set(want)
    for k, t in want.items():
        assert got[k].dtype == t.dtype and got[k].shape == t.shape and torch.equal(got[k], t)


def test_nommap_same_code_as_allbroker():
    """The bundled loader is the same code as llmt_kv.nommap_loader (docstrings may differ)."""
    src = os.environ.get("LLMT_KV_NOMMAP")            # path to AllBroker's llmt_kv/nommap_loader.py, or installed
    if not src:
        try:
            import llmt_kv.nommap_loader as ab
            src = ab.__file__
        except ImportError:
            pytest.skip("llmt_kv not installed (set LLMT_KV_NOMMAP=<AllBroker>/crates/py/python/llmt_kv/nommap_loader.py)")

    def body(src):
        t = ast.parse(src)
        for n in ast.walk(t):
            if isinstance(n, (ast.FunctionDef, ast.ClassDef, ast.Module, ast.AsyncFunctionDef)) and n.body \
                    and isinstance(n.body[0], ast.Expr) and isinstance(getattr(n.body[0], "value", None), ast.Constant):
                n.body = n.body[1:] or [ast.Pass()]
        return ast.dump(t)
    with open(src) as f:
        a = f.read()
    with open(os.path.join(ROOT, "exl3xpu", "nommap_loader.py")) as f:
        b = f.read()
    if body(a) != body(b):
        pytest.skip("installed llmt_kv has a different nommap_loader version (re-sync the bundled copy)")


def test_plugin_prefers_llmt_kv_loader():
    """register() imports llmt_kv.nommap_loader first and the bundled copy only on ImportError (source check: the
    plugin module needs the XPU build of vLLM to import)."""
    with open(os.path.join(ROOT, "exl3xpu", "vllm_plugin.py")) as f:
        src = f.read()
    reg = next(n for n in ast.parse(src).body if isinstance(n, ast.FunctionDef) and n.name == "register")
    seg = ast.get_source_segment(src, reg)
    i, j = seg.find("from llmt_kv import nommap_loader"), seg.find("from . import nommap_loader")
    assert 0 <= i < j
    assert "except ImportError" in seg[i:j]


def test_entry_point_declared():
    with open(os.path.join(ROOT, "pyproject.toml"), "rb") as f:
        cfg = tomllib.load(f)
    eps = cfg["project"]["entry-points"]["vllm.general_plugins"]
    assert any(v.startswith("exl3xpu.vllm_plugin:") for v in eps.values())


def test_lora_kernel_hooks_without_ops():
    lk = _load("llmt_lora_kernels_t", os.path.join(ROOT, "lorak", "llmt_lora_kernels.py"))
    assert lk.LINEAR is True and lk.MLP is True
    assert lk._ops() is None or hasattr(lk._ops(), "linear_lora")


def test_lazyvis_default_off():
    old = os.environ.pop("EXL3_LAZY_VISION", None)
    try:
        lv = _load("exl3xpu_lazyvis_t", os.path.join(ROOT, "exl3xpu", "lazyvis.py"))
    finally:
        if old is not None:
            os.environ["EXL3_LAZY_VISION"] = old
    assert lv.ON is False


if __name__ == "__main__":
    import inspect
    import tempfile
    import pathlib
    ok = skip = fail = 0
    for name, fn in list(globals().items()):
        if not (name.startswith("test_") and callable(fn)):
            continue
        try:
            with tempfile.TemporaryDirectory() as d:
                fn(*([pathlib.Path(d)] if "tmp_path" in inspect.signature(fn).parameters else []))
            ok += 1
            print("PASS", name)
        except Exception as e:  # noqa: BLE001
            if type(e).__name__ in ("_Skip", "Skipped"):
                skip += 1
                print("SKIP", name, e)
            else:
                fail += 1
                print("FAIL", name, repr(e))
    print(f"{ok} passed, {skip} skipped, {fail} failed")
    sys.exit(1 if fail else 0)
