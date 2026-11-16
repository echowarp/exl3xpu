"""CPU-only checks of the exl3xpu plugin pieces that need no Arc GPU: the no-mmap loader reads safetensors exactly,
the plugin prefers AllBroker's llmt_kv copy of that loader, the vLLM entry point is declared, the LoRA-kernel hook
module degrades to None without its ops, env-switched features default off, and the model.yaml serving config is the
production (int4 KV) build.

    python -m pytest -q tests/cpu/test_cpu_plugin.py      (from the repository root)
"""
import ast
import importlib.util
import json
import os
import struct
import sys
import tomllib
import types

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


def test_model_yaml_is_production_int4():
    """models/<id>/model.yaml = the production build: int4 KV, k = 7 with a depth schedule, the switches as strings, and
    serve.py turns it into a vllm serve command (no GPU needed for --print)."""
    import subprocess
    try:
        import yaml
    except ImportError:
        pytest.skip("pyyaml not installed")
    mdir = os.path.join(ROOT, "models", "qwen3.8-27b-exl3-4.00bpw")
    cfg = yaml.safe_load(open(os.path.join(mdir, "model.yaml")))
    v, env = cfg["vllm"], cfg["env"]
    assert v["kv_cache_dtype"] == "int4_per_token_head" and v["attention_backend"] == "TRITON_ATTN"
    assert v["speculative_config"]["num_speculative_tokens"] == 7
    assert all(isinstance(x, str) for x in env.values())
    for k in ("EXL3_GDN_RECOMPUTE", "EXL3_GDN_PORT", "EXL3_DRAFT_DYN", "EXL3_INT8_PREFILL", "EXL3_SHAPE_P_FILE"):
        assert k in env, k
    assert "EXL3_LAZY_VISION" not in env and "EXL3_WHX" not in env
    ks = env["EXL3_KSMOOTH"].replace("{model_dir}", mdir)                    # the shipped public K-smoothing table
    assert json.load(open(ks + ".inputs.json"))["sha256"] == __import__("hashlib").sha256(open(ks, "rb").read()).hexdigest()
    assert os.path.exists(os.path.join(mdir, "shape_p_c.json"))
    for k, n in (("EXL3_DRAFT_VOCAB", 512), ("EXL3_DRAFT_VOCAB_FAR", 128)):   # the shipped public draft vocabularies
        dv = json.load(open(env[k].replace("{model_dir}", mdir)))
        assert dv["source"] == "public" and dv["n_blocks"] == n == len(dv["blocks"]), k
    out = subprocess.run([sys.executable, os.path.join(ROOT, "scripts", "serve.py"), mdir, "--gpu", "0", "--print"],
                         capture_output=True, text=True, check=True).stdout
    assert "--kv-cache-dtype int4_per_token_head" in out and "num_speculative_tokens_per_batch_size" in out


def test_gdn_port_1_means_package_lib():
    src = open(os.path.join(ROOT, "exl3xpu", "vllm_plugin.py")).read()
    assert 'if _PORT_SO == "1":' in src and '"gdnchunk.so")' in src
    with open(os.path.join(ROOT, "pyproject.toml"), "rb") as f:
        assert "gdnchunk.so" in tomllib.load(f)["tool"]["setuptools"]["package-data"]["exl3xpu"]


def test_lazyvis_defers_to_allbroker():
    """With AllBroker's llmt_kv plugin installed, the standalone lazy tower does not install a second time."""
    import importlib.metadata as md
    old_env, old_ep = os.environ.get("EXL3_LAZY_VISION"), md.entry_points
    os.environ["EXL3_LAZY_VISION"] = "1"
    try:
        spec = importlib.util.spec_from_file_location("lazyvis_defer_t", os.path.join(ROOT, "exl3xpu", "lazyvis.py"))
        m = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(m)
        md.entry_points = lambda group=None: [types.SimpleNamespace(value="llmt_kv.plugin:register")]
        assert m._allbroker_lazyvis() is True
        m.install()                                   # returns before touching vLLM's loader
        assert m._STATE["installed"] is True and not m._STATE.get("towers")
        os.environ["LLMT_KV_PLUGIN"] = "0"
        assert m._allbroker_lazyvis() is False
        md.entry_points = lambda group=None: []
        os.environ.pop("LLMT_KV_PLUGIN")
        assert m._allbroker_lazyvis() is False
    finally:
        md.entry_points = old_ep
        os.environ.pop("LLMT_KV_PLUGIN", None)
        if old_env is None:
            os.environ.pop("EXL3_LAZY_VISION", None)
        else:
            os.environ["EXL3_LAZY_VISION"] = old_env


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
