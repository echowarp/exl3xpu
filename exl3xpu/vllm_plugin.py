"""
vLLM out-of-tree quantization plugin: EXL3 (exllamav3 trellis) weights on Intel XPU.

Registered via the `vllm.general_plugins` entry point, so every vLLM process (API server,
engine core, workers) imports it before model construction.
"""
from __future__ import annotations
import copy
import json
import os
import re
from typing import Any

import torch
from torch.nn import Parameter

from vllm.logger import init_logger
from vllm.model_executor.layers.linear import LinearBase, LinearMethodBase, UnquantizedLinearMethod
from vllm.model_executor.layers.quantization import register_quantization_config
from vllm.model_executor.layers.quantization.base_config import QuantizationConfig
from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead

logger = init_logger("vllm.exl3xpu")

# vLLM fused module -> checkpoint constituents
FUSED = {
    "gate_up_proj": ["gate_proj", "up_proj"],
    "qkv_proj": ["q_proj", "k_proj", "v_proj"],
    "in_proj_qkvz": ["in_proj_qkv", "in_proj_z"],
    "in_proj_ba": ["in_proj_b", "in_proj_a"],
}
QKV_IDS = {"q": 0, "k": 1, "v": 2}
CB_IDS = {"3inst": 0, "mcg": 1, "mul1": 2}


def _norm_key(k: str) -> str:
    """Canonical module name shared by checkpoint keys and vLLM prefixes:
       'model.language_model.layers.3.mlp.gate_proj' / 'language_model.model.layers.3.mlp.gate_proj'
           -> 'layers.3.mlp.gate_proj'
       'mtp.layers.0.mlp.gate_proj' / 'model.mtp.layers.0...' -> 'mtp.layers.0.mlp.gate_proj'
       '...lm_head' -> 'lm_head'"""
    parts = k.split(".")
    if "mtp" in parts:
        return "mtp." + ".".join(parts[parts.index("mtp") + 1:])
    m = re.search(r"(layers\.\d+\..*)$", k)
    if m:
        return m.group(1)
    return parts[-1]


@register_quantization_config("exl3")
class Exl3Config(QuantizationConfig):

    def __init__(self, bits: float, head_bits: int, codebook: str, storage: dict | None = None,
                 mtp_bits: int | None = None):
        super().__init__()
        self.bits = bits
        self.mtp_bits = int(mtp_bits or bits)
        self.head_bits = head_bits
        self.codebook = codebook
        self.storage = storage or {}

    def __repr__(self):
        return f"Exl3Config(bits={self.bits}, head_bits={self.head_bits}, codebook={self.codebook})"

    def get_name(self):
        return "exl3"

    def get_supported_act_dtypes(self):
        return [torch.float16, torch.bfloat16]

    @classmethod
    def get_min_capability(cls) -> int:
        return 0

    @staticmethod
    def get_config_filenames() -> list[str]:
        return []

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "Exl3Config":
        return cls(bits=config.get("bits", 4), head_bits=config.get("head_bits", 6),
                   codebook=config.get("codebook", "mul1"), mtp_bits=config.get("mtp_bits"))

    def _model_file(self, model_name, fname, revision):
        path = os.path.join(model_name, fname)
        if os.path.isfile(path):
            return path
        try:
            from huggingface_hub import hf_hub_download
            return hf_hub_download(model_name, fname, revision=revision)
        except Exception:
            return None

    def maybe_update_config(self, model_name: str, hf_config=None, revision=None):
        # Which modules are EXL3 is decided by the checkpoint itself: a module is EXL3 iff the weight
        # index has '<module>.trellis'. (quantization_config.json's tensor_storage omits the MTP head.)
        idx_path = self._model_file(model_name, "model.safetensors.index.json", revision)
        qc_path = self._model_file(model_name, "quantization_config.json", revision)
        stored = {}
        if qc_path:
            with open(qc_path) as f:
                for key, v in json.load(f).get("tensor_storage", {}).items():
                    if v.get("quant_format") == "exl3":
                        stored[_norm_key(key)] = int(v.get("bits_per_weight", self.bits))
        if idx_path:
            with open(idx_path) as f:
                wmap = json.load(f)["weight_map"]
            for name in wmap:
                if name.endswith(".trellis"):
                    key = _norm_key(name[: -len(".trellis")])
                    default = self.mtp_bits if key.startswith("mtp.") else int(self.bits)
                    self.storage[key] = stored.get(key, default)
        else:
            self.storage.update(stored)
        logger.info("exl3: %d EXL3 modules (%d in MTP head)", len(self.storage),
                    sum(1 for k in self.storage if k.startswith("mtp.")))

    def _bits_for(self, prefix: str) -> int | None:
        parts = prefix.split(".")
        if "visual" in parts or "vision_tower" in parts:
            return None
        key = _norm_key(prefix)
        if not self.storage:
            if key.endswith("in_proj_ba"):
                return None
            return self.head_bits if key == "lm_head" else int(self.bits)
        base, _, leaf = key.rpartition(".")
        members = FUSED.get(leaf, [leaf])
        names = [f"{base}.{p}" if base else p for p in members]
        bits = [self.storage.get(n) for n in names]
        if all(b is None for b in bits):
            return None
        if any(b != bits[0] for b in bits):
            # mixed-precision checkpoints (e.g. turboderp SC_*_H5): one EXL3 call per equal-K run of members
            assert all(b is not None for b in bits), f"exl3: partly quantized fused module {prefix}: {bits}"
            return tuple(bits)
        return bits[0]

    def get_quant_method(self, layer: torch.nn.Module, prefix: str):
        if isinstance(layer, LinearBase):
            bits = self._bits_for(prefix)
            if bits is None:
                return UnquantizedLinearMethod()
            if isinstance(bits, tuple):
                return Exl3MixedLinearMethod(self, bits)
            return Exl3LinearMethod(self, bits)
        if isinstance(layer, ParallelLMHead):
            bits = self._bits_for(prefix)
            if bits is None:
                return None
            return Exl3LinearMethod(self, bits)
        return None


class Exl3LinearMethod(LinearMethodBase):

    def __init__(self, config: Exl3Config, bits: int):
        self.config = config
        self.K = int(bits)
        self.cb = CB_IDS[config.codebook]

    def create_weights(self, layer, input_size_per_partition, output_partition_sizes, input_size,
                       output_size, params_dtype, **extra_weight_attrs):
        k = input_size_per_partition
        n = sum(output_partition_sizes)
        if k != input_size or n != (output_size if not isinstance(layer, ParallelLMHead) else n):
            raise NotImplementedError("exl3: tensor parallel sharding not supported yet (use DP)")
        assert k % 128 == 0 and all(s % 128 == 0 for s in output_partition_sizes), \
            f"exl3: dims must be multiples of 128 (k={k}, n={output_partition_sizes})"
        K = self.K
        layer.exl3_sizes = list(output_partition_sizes)
        layer.exl3_offsets = [sum(output_partition_sizes[:i]) for i in range(len(output_partition_sizes) + 1)]
        layer.exl3_groups = set()
        layer.exl3_K = K
        layer.exl3_cb = self.cb

        def reg(name, tensor, loader):
            p = Parameter(tensor, requires_grad=False)
            p.weight_loader = loader
            layer.register_parameter(name, p)

        reg("trellis", torch.empty((k // 16, n // 16, 16 * K), dtype=torch.int16), self._load_trellis(layer))
        reg("suh", torch.empty((len(output_partition_sizes), k), dtype=torch.float16), self._load_suh(layer))
        reg("svh", torch.empty((n,), dtype=torch.float16), self._load_svh(layer))
        if self.config.codebook != "3inst":
            reg(self.config.codebook, torch.empty((), dtype=torch.int32), lambda p, w, *a, **kw: p.data.copy_(w))

    @staticmethod
    def _shards(layer, loaded, sid=None) -> list[int]:
        if sid is None:
            sid = getattr(loaded, "shard_id", None)
        if sid is None:
            return list(range(len(layer.exl3_sizes)))
        if isinstance(sid, str):
            return [QKV_IDS[sid]]
        if isinstance(sid, int):
            return [sid]
        return list(sid)

    def _range(self, layer, loaded, sid=None):
        s = self._shards(layer, loaded, sid)
        assert s == list(range(s[0], s[-1] + 1)), s
        return s, layer.exl3_offsets[s[0]], layer.exl3_offsets[s[-1] + 1]

    def _load_trellis(self, layer):
        def f(param, w, sid=None):
            s, a, b = self._range(layer, w, sid)
            assert w.dtype == torch.int16 and w.shape[-1] == param.shape[-1], \
                f"exl3: trellis bits mismatch {tuple(w.shape)} vs {tuple(param.shape)}"
            assert w.shape[1] * 16 == b - a and w.shape[0] == param.shape[0], (w.shape, a, b)
            param.data[:, a // 16: b // 16].copy_(w)
            layer.exl3_groups.add(tuple(s))
        return f

    def _load_suh(self, layer):
        def f(param, w, sid=None):
            s, _, _ = self._range(layer, w, sid)
            for i in s:
                param.data[i].copy_(w)
        return f

    def _load_svh(self, layer):
        def f(param, w, sid=None):
            _, a, b = self._range(layer, w, sid)
            param.data[a:b].copy_(w)
        return f

    def process_weights_after_loading(self, layer):
        groups = sorted(layer.exl3_groups) or [tuple(range(len(layer.exl3_sizes)))]
        covered = [i for g in groups for i in g]
        assert covered == list(range(len(layer.exl3_sizes))), f"exl3: incomplete shards {groups}"
        dev = layer.trellis.device
        suh = torch.stack([layer.suh.data[g[0]] for g in groups]).contiguous()
        bounds = [layer.exl3_offsets[g[0]] for g in groups] + [layer.exl3_offsets[-1]]
        n = bounds[-1]
        shard_of_nb = torch.empty(n // 128, dtype=torch.int32)
        for gi in range(len(groups)):
            shard_of_nb[bounds[gi] // 128: bounds[gi + 1] // 128] = gi
        layer.suh = Parameter(suh, requires_grad=False)
        layer.exl3_shard_of_nb = shard_of_nb.to(dev)
        layer.exl3_bounds = bounds
        if isinstance(layer, ParallelLMHead) and os.environ.get("EXL3_DRAFT_VOCAB") and _spec_is_mtp():
            _build_draft_head(layer, os.environ["EXL3_DRAFT_VOCAB"])
            # GRADED-DRAFT: a smaller draft vocab for the deeper draft positions
            # (EXL3_DRAFT_VOCAB_FAR, from position EXL3_DRAFT_FAR_FROM, 0-based): those passes read fewer head blocks.
            if os.environ.get("EXL3_DRAFT_VOCAB_FAR"):
                _build_draft_head(layer, os.environ["EXL3_DRAFT_VOCAB_FAR"], attr="exl3_draft_far")
        cb = self.cb
        if hasattr(layer, "mcg"):
            cb = 1
        elif hasattr(layer, "mul1"):
            cb = 2
        layer.exl3_cb = cb
        if isinstance(layer, ParallelLMHead) and os.environ.get("EXL3_DRAFT_Q4"):   # DRAFT-Q4
            # EXL3_DRAFT_Q4_SCOPE=far (keep the high accuracy close, low accuracy far out): int4 only for
            # the far head (passes >= EXL3_DRAFT_FAR_FROM); default "all" = both heads
            attrs = ("exl3_draft_far",) if os.environ.get("EXL3_DRAFT_Q4_SCOPE", "all") == "far" else ("exl3_draft", "exl3_draft_far")
            for attr in attrs:
                if getattr(layer, attr, None) is not None:
                    _build_draft_q4(layer, getattr(layer, attr), int(os.environ["EXL3_DRAFT_Q4"]))
        if isinstance(layer, ParallelLMHead) and getattr(layer, "exl3_draft", None) is not None and (
                os.environ.get("EXL3_DRAFT_DYN", "0") != "0" or os.environ.get("EXL3_DRAFT_BIAS", "0") not in ("0", "")):
            _build_draft_dyn(layer)   # DRAFT-DYN + context bias, probabilistic-exact
        # decide the backend once, outside any traced code: the all-C++ op when the ESIMD library has it
        from . import ops
        E = ops._get_esimd()
        layer.exl3_cpp = bool(E) and hasattr(E, "linear") and bool(E.exl3_supported(layer.exl3_K, cb))

    def apply(self, layer, x, bias=None):
        from . import ops
        if layer.exl3_cpp:
            y = torch.ops.exl3xpu_C.linear(x, layer.trellis, layer.suh, layer.svh, layer.exl3_shard_of_nb,
                                           layer.exl3_bounds, layer.exl3_K, layer.exl3_cb,
                                           ops.SMALL_M_MAX, ops.RECON_SLICE_N)
        else:
            y = ops._exl3_linear_py(x, layer.trellis, layer.suh, layer.svh, layer.exl3_shard_of_nb,
                                    layer.exl3_bounds, layer.exl3_K, layer.exl3_cb)
        if bias is not None:
            y = y + bias
        return y

    # lm_head (ParallelLMHead) path
    def embedding(self, layer, input_):
        raise NotImplementedError("exl3: quantized input embeddings are not supported")


class Exl3MixedLinearMethod(LinearMethodBase):
    """Fused module whose checkpoint members have different bit widths. Each loaded
    checkpoint tensor is kept as its own group (own trellis K, suh, svh); consecutive groups with equal K are
    concatenated into one EXL3 call, and apply() concatenates the call outputs along the output dim."""

    def __init__(self, config: Exl3Config, bits: tuple):
        self.config = config
        self.bits = bits
        self.cb = CB_IDS[config.codebook]

    def create_weights(self, layer, input_size_per_partition, output_partition_sizes, input_size,
                       output_size, params_dtype, **extra_weight_attrs):
        k = input_size_per_partition
        if k != input_size:
            raise NotImplementedError("exl3: tensor parallel sharding not supported yet (use DP)")
        layer.exl3_sizes = list(output_partition_sizes)
        layer.exl3_offsets = [sum(output_partition_sizes[:i]) for i in range(len(output_partition_sizes) + 1)]
        layer.exl3_mx = {}                 # group (tuple of partition ids) -> {"tr", "suh", "svh"}
        layer.exl3_cb = self.cb
        layer.exl3_k_in = k

        def reg(name, loader):
            p = Parameter(torch.empty(1, dtype=torch.int16 if name == "trellis" else torch.float16),
                          requires_grad=False)
            p.weight_loader = loader
            layer.register_parameter(name, p)

        def store(key):
            def f(param, w, sid=None):
                s = tuple(Exl3LinearMethod._shards(layer, w, sid))
                layer.exl3_mx.setdefault(s, {})[key] = w.to(param.device)
            return f
        reg("trellis", store("tr"))
        reg("suh", store("suh"))
        reg("svh", store("svh"))
        if self.config.codebook != "3inst":
            p = Parameter(torch.empty((), dtype=torch.int32), requires_grad=False)
            p.weight_loader = lambda p, w, *a, **kw: p.data.copy_(w)
            layer.register_parameter(self.config.codebook, p)

    def process_weights_after_loading(self, layer):
        groups = sorted(layer.exl3_mx)
        covered = [i for g in groups for i in g]
        assert covered == list(range(len(layer.exl3_sizes))), f"exl3: incomplete shards {groups}"
        cb = self.cb
        if hasattr(layer, "mcg"):
            cb = 1
        elif hasattr(layer, "mul1"):
            cb = 2
        from . import ops
        E = ops._get_esimd()
        calls, run = [], []

        def flush():
            if not run:
                return
            K = run[0]["tr"].shape[-1] // 16
            tr = torch.cat([g["tr"] for g in run], dim=1).contiguous() if len(run) > 1 else run[0]["tr"].contiguous()
            suh = torch.stack([g["suh"] for g in run]).contiguous()
            svh = torch.cat([g["svh"] for g in run]).contiguous()
            widths = [g["tr"].shape[1] * 16 for g in run]
            bounds = [sum(widths[:i]) for i in range(len(widths) + 1)]
            shard = torch.cat([torch.full((w // 128,), i, dtype=torch.int32) for i, w in enumerate(widths)])
            assert bool(E) and E.exl3_supported(K, cb), f"exl3: K={K} cb={cb} needs an EXL3_ALL_CODEBOOKS build"
            calls.append(dict(tr=tr, suh=suh, svh=svh, shard=shard.to(tr.device), bounds=bounds, K=K))
            run.clear()
        for g in groups:
            d = layer.exl3_mx[g]
            if run and run[0]["tr"].shape[-1] != d["tr"].shape[-1]:
                flush()
            run.append(d)
        flush()
        layer.exl3_calls = calls
        layer.exl3_cb = cb
        layer.exl3_mx = None
        for name in ("trellis", "suh", "svh"):
            delattr(layer, name)
        logger.debug("exl3: mixed fused module, K per call %s", [c["K"] for c in calls])

    def apply(self, layer, x, bias=None):
        from . import ops
        ys = [torch.ops.exl3xpu_C.linear(x, c["tr"], c["suh"], c["svh"], c["shard"], c["bounds"], c["K"],
                                         layer.exl3_cb, ops.SMALL_M_MAX, ops.RECON_SLICE_N)
              for c in layer.exl3_calls]
        y = ys[0] if len(ys) == 1 else torch.cat(ys, dim=-1)
        if bias is not None:
            y = y + bias
        return y

    def embedding(self, layer, input_):
        raise NotImplementedError("exl3: quantized input embeddings are not supported")


# ------------------------------------------------------------------------------------------------
# Pruned-vocabulary draft head for MTP speculative decoding.
# The drafter proposes tokens from a subset of 128-token blocks (EXL3 lm_head columns can only be
# sliced by whole 128-column Hadamard blocks); the target still verifies with the full lm_head, so
# generated text is unchanged. Enabled by EXL3_DRAFT_VOCAB=<draft_vocab.json> (see scripts/draft_vocab.py).

_DRAFT_FAR_FROM = int(os.environ.get("EXL3_DRAFT_FAR_FROM", "2"))


def _spec_is_mtp():
    """The pruned draft head only serves the MTP drafter; with another drafter (dspark, dflash) or none it is
    dead weight in VRAM, so skip it."""
    try:
        from vllm.config import get_current_vllm_config
        sc = get_current_vllm_config().speculative_config
    except Exception:
        return True
    return sc is not None and "mtp" in str(getattr(sc, "method", "")).lower()


def _build_draft_head(layer, path, attr="exl3_draft"):
    with open(path) as f:
        spec = json.load(f)
    blocks = torch.tensor(spec["blocks"], dtype=torch.long)
    n_total_blocks = layer.svh.shape[0] // 128
    blocks = blocks[blocks < n_total_blocks]
    dev = layer.trellis.device
    tiles = (blocks[:, None] * 8 + torch.arange(8)[None, :]).flatten().to(dev)
    trellis = layer.trellis.data.index_select(1, tiles).contiguous()
    svh = layer.svh.data.view(-1, 128).index_select(0, blocks.to(dev)).flatten().contiguous()
    idx = (blocks[:, None] * 128 + torch.arange(128)[None, :]).flatten().to(dev)
    setattr(layer, attr, dict(trellis=trellis, svh=svh, idx=idx,
                              shard=torch.zeros(len(blocks), dtype=torch.int32, device=dev),
                              bounds=[0, len(blocks) * 128]))
    logger.info("exl3: MTP draft head (%s) uses %d of %d vocab blocks (%.1f%% of lm_head)",
                attr, len(blocks), n_total_blocks, 100.0 * len(blocks) / n_total_blocks)


def _build_draft_q4(layer, d, gs):
    """DRAFT-Q4: the drafter's compact head as plain int4 (asymmetric, group gs along K)
    for torch's XPU weight-only int4 GEMV (M=1 65536 x 2048: 126 us at gs 64 vs 228 us for the 6-bit EXL3 slice: the
    trellis decode is integer-pipe bound). Built from the EXL3 slice itself (identity rows through the same kernel),
    so it quantizes exactly what the drafter used. Drafter only: the target's lm_head and verify are untouched, so the
    output is unchanged; only acceptance can move. The EXL3 slice is dropped to give its VRAM back."""
    from . import ops
    K = layer.suh.shape[-1]
    N = d["bounds"][-1]
    dev = layer.trellis.device
    wt = torch.empty(K, N, dtype=torch.float16, device=dev)            # W^T: row i = W @ e_i
    for a in range(0, K, 256):
        eye = torch.zeros(min(256, K - a), K, dtype=torch.float16, device=dev)
        eye[torch.arange(eye.shape[0]), torch.arange(a, a + eye.shape[0])] = 1
        wt[a:a + eye.shape[0]] = torch.ops.exl3xpu_C.linear(eye, d["trellis"], layer.suh, d["svh"], d["shard"],
                                                             d["bounds"], layer.exl3_K, layer.exl3_cb,
                                                             ops.SMALL_M_MAX, ops.RECON_SLICE_N)
    G = K // gs
    packed = torch.empty(N, K // 8, dtype=torch.int32, device=dev)
    scales = torch.empty(G, N, dtype=torch.float16, device=dev)
    zeros = torch.empty(G, N, dtype=torch.int8, device=dev)
    err = 0.0
    for a in range(0, N, 8192):
        w = wt[:, a:a + 8192].t().float().reshape(-1, G, gs)          # [n, G, gs]
        mn, mx = w.amin(-1, keepdim=True), w.amax(-1, keepdim=True)
        s = ((mx - mn) / 15).clamp_min(1e-8)
        z = (-mn / s).round().clamp(0, 15)
        q = (w / s + z).round().clamp(0, 15)
        err = max(err, ((q - z) * s - w).abs().amax().item())
        q = q.reshape(w.shape[0], K).to(torch.uint8)
        packed[a:a + w.shape[0]] = (q[:, 0::2] | (q[:, 1::2] << 4)).contiguous().view(torch.int32)
        scales[:, a:a + w.shape[0]] = s.squeeze(-1).t().half()
        zeros[:, a:a + w.shape[0]] = z.squeeze(-1).t().to(torch.int8)
    del wt
    d["q4"] = (packed, gs, scales, zeros)
    if os.environ.get("EXL3_DRAFT_Q4_KEEP", "0") != "1":
        for key in ("trellis", "svh"):
            d.pop(key, None)
    torch.xpu.empty_cache()
    logger.info("exl3: DRAFT-Q4 head: %d x %d int4, group %d (%.1f MB; max abs dequant err %.2e)",
                N, K, gs, (packed.numel() * 4 + scales.numel() * 3) / 1e6, err)


_DYN_HOST: dict = {}


def _dyn_host_table(layer, gs):
    """DRAFT-DYN: the whole lm_head as per-row int4 g{gs} rows on the host (packed [V, K/8] int32, scales [V, G]
    fp16, zeros [V, G] int8; ~0.7 GB RAM for the 27B), built in 128-block chunks through the EXL3 kernel (identity rows,
    as _build_draft_q4) so it quantizes exactly what DRAFT-Q4 quantizes. Built once per process: the MTP drafter shares
    the target's lm_head (vLLM "Sharing target model lm_head weights"), a second ParallelLMHead with the same svh reuses it."""
    from . import ops
    from .draftdyn import quantize_rows_int4
    key = (tuple(layer.svh.shape), gs)
    hit = _DYN_HOST.get(key)
    if hit is not None and torch.equal(hit["svh"], layer.svh.data[:4096].cpu()):
        return hit["table"]
    K = layer.suh.shape[-1]
    n_blocks = layer.svh.shape[0] // 128
    dev = layer.trellis.device
    parts = ([], [], [])
    for b0 in range(0, n_blocks, 128):
        blocks = torch.arange(b0, min(n_blocks, b0 + 128), dtype=torch.long)
        tiles = (blocks[:, None] * 8 + torch.arange(8)[None, :]).flatten().to(dev)
        trellis = layer.trellis.data.index_select(1, tiles).contiguous()
        svh = layer.svh.data.view(-1, 128).index_select(0, blocks.to(dev)).flatten().contiguous()
        N = len(blocks) * 128
        shard = torch.zeros(len(blocks), dtype=torch.int32, device=dev)
        wt = torch.empty(K, N, dtype=torch.float16, device=dev)
        for a in range(0, K, 256):
            eye = torch.zeros(min(256, K - a), K, dtype=torch.float16, device=dev)
            eye[torch.arange(eye.shape[0]), torch.arange(a, a + eye.shape[0])] = 1
            wt[a:a + eye.shape[0]] = torch.ops.exl3xpu_C.linear(eye, trellis, layer.suh, svh, shard, [0, N],
                                                                 layer.exl3_K, layer.exl3_cb, ops.SMALL_M_MAX,
                                                                 ops.RECON_SLICE_N)
        for r0 in range(0, N, 4096):
            q, s_, z = quantize_rows_int4(wt[:, r0:r0 + 4096].t(), gs)
            parts[0].append(q.cpu()); parts[1].append(s_.cpu()); parts[2].append(z.cpu())
        del wt, trellis, svh
    torch.xpu.empty_cache()
    table = tuple(torch.cat(x).numpy() for x in parts)
    _DYN_HOST[key] = dict(svh=layer.svh.data[:4096].cpu(), table=table)
    logger.info("exl3: DRAFT-DYN host int4 lm_head: %d x %d, group %d (%.0f MB host)", table[0].shape[0], K, gs,
                sum(x.nbytes for x in table) / 1e6)
    return table


def _build_draft_dyn(layer):
    """Persistent per-batch-row buffers for DRAFT-DYN / context bias on both draft heads, allocated
    at load (before vLLM sizes the KV cache). See draftdyn.py and draftgraph._dyn_set_batch / _sub_sample."""
    E_far = int(os.environ.get("EXL3_DRAFT_DYN", "0"))
    E_near = int(os.environ.get("EXL3_DRAFT_DYN_NEAR", str(E_far)))
    beta = float(os.environ.get("EXL3_DRAFT_BIAS", "0") or 0)
    gs = int(os.environ.get("EXL3_DRAFT_Q4", "64") or 64)
    try:
        from vllm.config import get_current_vllm_config
        S = int(get_current_vllm_config().scheduler_config.max_num_seqs)
    except Exception:  # noqa
        S = 16
    dev = layer.suh.device
    K = layer.suh.shape[-1]
    G = K // gs
    host = _dyn_host_table(layer, gs) if max(E_far, E_near) > 0 else None
    mb = 0.0
    for attr, E in (("exl3_draft", E_near), ("exl3_draft_far", E_far)):
        d = getattr(layer, attr, None)
        if d is None:
            continue
        Ns = d["idx"].numel()
        dy = dict(E=E, Ns=Ns, S=S, beta=beta, gs=gs, G=G, B=0, host=host if E > 0 else None)
        if beta:
            dy["SB"] = torch.zeros(S, Ns, dtype=torch.float16, device=dev)
        if E > 0:
            dy["X"] = torch.zeros(S, E, dtype=torch.int64, device=dev)
            dy["XB"] = torch.full((S, E), float("-inf"), dtype=torch.float16, device=dev)
            dy["L"] = torch.empty(S, Ns + E, dtype=torch.float16, device=dev)
            dy["Wq"] = torch.zeros(S * E, K // 8, dtype=torch.int32, device=dev)
            dy["Ws"] = torch.zeros(G * S * E, dtype=torch.float16, device=dev)
            dy["Wz"] = torch.zeros(G * S * E, dtype=torch.int8, device=dev)
        mb += sum(v.numel() * v.element_size() for v in dy.values() if isinstance(v, torch.Tensor)) / 1e6
        d["dyn"] = dy
    logger.info("exl3: DRAFT-DYN buffers: near E=%d, far E=%d, bias %.2f, %d rows (%.1f MB device)",
                E_near, E_far, beta, S, mb)


def draft_sub_logits(lm, d, h):
    """Compact draft-head logits [M, n_kept]: the int4 head when DRAFT-Q4 built one, else the EXL3 slice."""
    q = d.get("q4")
    if q is not None:
        return torch.ops.aten._weight_int4pack_mm_with_scales_and_zeros(h if h.dtype == torch.float16 else h.half(),
                                                                         q[0], q[1], q[2], q[3])
    from . import ops
    return torch.ops.exl3xpu_C.linear(h, d["trellis"], lm.suh, d["svh"], d["shard"], d["bounds"],
                                      lm.exl3_K, lm.exl3_cb, ops.SMALL_M_MAX, ops.RECON_SLICE_N)


def _patch_mtp_draft_logits():
    try:
        from vllm.model_executor.models import qwen3_5_mtp
    except Exception:
        return
    cls = qwen3_5_mtp.Qwen3_5MTP
    if getattr(cls, "_exl3_patched", False):
        return
    orig = cls.compute_logits

    def compute_logits(self, hidden_states, spec_step_idx: int = 0):
        lm = self.lm_head
        d = getattr(lm, "exl3_draft", None)
        if d is None:
            return orig(self, hidden_states, spec_step_idx)
        far = getattr(lm, "exl3_draft_far", None)
        if far is not None:
            k = getattr(self, "_exl3_pass", 0)
            self._exl3_pass = k + 1
            if k >= _DRAFT_FAR_FROM:
                d = far
        sub = draft_sub_logits(lm, d, hidden_states)   # DRAFT-Q4 int4 head when built
        logits = hidden_states.new_full((hidden_states.shape[0], lm.svh.shape[0]), float("-inf"))
        logits.index_copy_(1, d["idx"], sub)
        return logits[:, : self.config.vocab_size]

    cls.compute_logits = compute_logits
    cls._exl3_patched = True
    if os.environ.get("EXL3_DRAFT_VOCAB_FAR"):
        try:
            from vllm.v1.spec_decode.llm_base_proposer import SpecDecodeBaseProposer as _P
            _orig_propose = _P.propose

            def propose(self, *args, **kwargs):
                try:
                    self.model._exl3_pass = 0
                except Exception:
                    pass
                return _orig_propose(self, *args, **kwargs)

            _P.propose = propose
            logger.info("exl3: graded draft vocab: far head from draft position %d", _DRAFT_FAR_FROM)
        except Exception as e:  # noqa
            logger.warning("exl3: graded draft vocab not installed (%s)", e)


def _patch_fused_mlp():
    """Decode-batch MLP (3..8 rows) as one exl3xpu op: had_in -> gate_up DPAS -> fused
    had_out(gate, up) + silu*mul + had_in(down) -> down DPAS -> had_out (3 fewer kernels per layer; -9% MLP time in
    a microbenchmark; same math, fp16-rounding-level differences). EXL3_FUSED_MLP=0 disables."""
    if os.environ.get("EXL3_FUSED_MLP", "1") != "1":
        return
    try:
        from vllm.model_executor.models.qwen2_moe import Qwen2MoeMLP
    except Exception:
        return
    if getattr(Qwen2MoeMLP, "_exl3_mlp_patched", False):
        return
    try:
        from . import ops
        E = ops._get_esimd()
        if not E or not hasattr(torch.ops.exl3xpu_C, "mlp_auto"):
            return
    except Exception as e:  # noqa
        logger.info("exl3: fused MLP not available here (%s)", e)
        return
    try:
        @torch.library.register_fake("exl3xpu_C::mlp_auto")
        def _(x, gu_tr, gu_suh, gu_svh, gu_shard, gu_bounds, dn_tr, dn_suh, dn_svh, dn_shard, dn_bounds, K_gu, K_dn, cb,
              small_m_max, slice_n):
            return x.new_empty((*x.shape[:-1], dn_svh.shape[0]))
    except Exception as e:  # noqa
        logger.warning("exl3: fused MLP fake impl not registered (%s); fused MLP off", e)
        return
    orig = Qwen2MoeMLP.forward

    def forward(self, x):
        gu, dn = self.gate_up_proj, self.down_proj
        # static (per-layer) conditions only: the batch-size decision is made inside the op at run time
        if (getattr(gu, "exl3_cpp", False) and getattr(dn, "exl3_cpp", False) and gu.exl3_cb == dn.exl3_cb
                and len(gu.exl3_bounds) == 3 and len(dn.exl3_bounds) == 2 and getattr(gu, "bias", None) is None):
            from . import ops as _ops
            return torch.ops.exl3xpu_C.mlp_auto(x, gu.trellis, gu.suh, gu.svh, gu.exl3_shard_of_nb, gu.exl3_bounds,
                                                dn.trellis, dn.suh, dn.svh, dn.exl3_shard_of_nb, dn.exl3_bounds,
                                                gu.exl3_K, dn.exl3_K, gu.exl3_cb, _ops.SMALL_M_MAX, _ops.RECON_SLICE_N)
        return orig(self, x)

    Qwen2MoeMLP.forward = forward
    Qwen2MoeMLP._exl3_mlp_patched = True
    logger.info("exl3: fused decode MLP enabled (3..8 rows)")


def _patch_fast_norm():
    """Residual add + RMSNorm for decode batches via exl3xpu_C::exl3_fused_add_rms_norm
    (one work-group per row, 256 threads) instead of custom_esimd_kernels_vllm.esimd_fused_add_rms_norm_batched:
    2.8 vs 8.4 us device time at M=6, 3.6 vs 10.9 at M=18 (microbenchmark), 138 calls per decode step; exact vLLM
    fused_add_rms_norm semantics (the ESIMD op is 1-2 ulp off the fp32 reference). layernorm.py looks the ESIMD op up
    by attribute on every call, so replacing the attribute reroutes it. EXL3_FAST_NORM=0 disables."""
    if os.environ.get("EXL3_FAST_NORM", "1") != "1":
        return
    try:
        import importlib
        from . import ops
        E = ops._get_esimd()
        if not E or not hasattr(torch.ops.exl3xpu_C, "exl3_fused_add_rms_norm"):
            return
        mod = importlib.import_module("custom_esimd_kernels_vllm")
    except Exception as e:  # noqa
        logger.info("exl3: fast RMSNorm not available here (%s)", e)
        return
    if getattr(mod, "_exl3_norm_patched", False):
        return
    try:
        torch.ops.exl3xpu_C.exl3_set_norm_wg(int(os.environ.get("EXL3_FAST_NORM_WG", "256")))
    except Exception:
        pass
    _op = torch.ops.exl3xpu_C.exl3_fused_add_rms_norm

    def esimd_fused_add_rms_norm_batched(x, residual, weight, eps):
        _op(x, residual, weight, float(eps))

    mod.esimd_fused_add_rms_norm_batched = esimd_fused_add_rms_norm_batched
    mod._exl3_norm_patched = True
    logger.info("exl3: fast fused add+RMSNorm enabled (decode batches)")


# ---------------------------------------------------------------------------------------------------------------
# GDN-RECOMPUTE: speculative decoding for the GDN layers with one
# committed SSM state slot per request + the previous step's accepted tokens applied lazily (exl3_gdn_recompute),
# so MambaSpec.num_speculative_blocks = 0. EXL3_GDN_RECOMPUTE=1 enables.
GDN_RC = {"on": os.environ.get("EXL3_GDN_RECOMPUTE", "0") == "1", "S": int(os.environ.get("EXL3_GDN_RC_SLOTS", "16"))}


def _patch_gdn_recompute():
    if not GDN_RC["on"]:
        return
    import dataclasses
    try:
        from vllm.model_executor.layers.mamba import abstract as mabs
        from vllm.v1.kv_cache_interface import MambaSpec
        from vllm.model_executor.layers.mamba import mamba_utils as mmu
        from vllm.v1.worker import mamba_utils as wmu
        from vllm.model_executor.layers.mamba.gdn import qwen_gdn_linear_attn as qg
    except Exception as e:  # noqa
        logger.warning("exl3: GDN recompute not installed (%s)", e)
        return
    if getattr(mabs.MambaBase, "_exl3_rc", False):
        return
    # 1. no speculative state blocks
    orig_spec = mabs.MambaBase.get_kv_cache_spec

    def get_kv_cache_spec(self, vllm_config):
        spec = orig_spec(self, vllm_config)
        if isinstance(spec, MambaSpec) and spec.num_speculative_blocks:
            GDN_RC["T1"] = spec.num_speculative_blocks + 1
            spec = dataclasses.replace(spec, num_speculative_blocks=0)
        return spec
    mabs.MambaBase.get_kv_cache_spec = get_kv_cache_spec
    mabs.MambaBase._exl3_rc = True

    # 2. block-crossing SSM copy takes the committed state (no accept bias); same function object (identity checks)
    def _temporal(state, block_ids, cur_block_idx, num_accepted_tokens):
        src_state = state[block_ids[cur_block_idx]]
        return MambaCopySpec(start_addr=src_state.data_ptr(), num_elements=src_state.numel())
    mmu.get_temporal_copy_spec.__code__ = _temporal.__code__

    # 3. fused postprocess: no SSM copies (the kernel stores decode-time boundary states itself)
    orig_init = wmu.MambaSpecDecodeGPUContext.initialize_from_forward_context

    def initialize_from_forward_context(self, *a, **kw):
        was = self.is_initialized
        orig_init(self, *a, **kw)
        if not was:
            self.state_inner_sizes.masked_fill_(self.state_conv_widths == 0, 0)
            logger.info("exl3: GDN recompute: postprocess SSM copies disabled")
    wmu.MambaSpecDecodeGPUContext.initialize_from_forward_context = initialize_from_forward_context

    # 4. GDN core: route inside the opaque custom op (torch.compile freezes Python branches in forward_xpu at trace
    #    time): swap the op implementation's code for a dispatcher (same function object -> registration intact).
    import types
    from vllm import _xpu_ops as xo
    impl = xo._gdn_attention_core_xpu_impl
    xo._exl3_rc_orig_impl = types.FunctionType(impl.__code__, impl.__globals__, "_orig", impl.__defaults__, impl.__closure__)
    xo._exl3_rc_dispatch = _rc_dispatch

    def _wrapper(core_attn_out, z, projected_states_qkvz, projected_states_ba, layer_name):
        return _exl3_rc_dispatch(core_attn_out, z, projected_states_qkvz, projected_states_ba, layer_name)
    impl.__code__ = _wrapper.__code__
    logger.info("exl3: GDN recompute enabled (num_speculative_blocks = 0, %d pending slots)", GDN_RC["S"])


def _rc_alloc(layer, dev):
    T1 = GDN_RC.get("T1", 6); S = GDN_RC["S"]
    H = layer.num_k_heads // layer.tp_size; HV = layer.num_v_heads // layer.tp_size; K = layer.head_k_dim; V = layer.head_v_dim
    layer._exl3_pool = (torch.zeros(2, S, T1, H, K, device=dev), torch.zeros(2, S, T1, HV, V, device=dev),
                        torch.zeros(2, S, T1, HV, device=dev), torch.zeros(2, S, T1, HV, device=dev))


_RC_FUSED_IN = os.environ.get("EXL3_RC_FUSED_IN", "0") == "1"
_RCF_N = [0, 0, 0]
_RCF_FB = {False: 0, True: 0}   # unfused-fallback layer calls (eager, captured)
_RCF_DBG = os.environ.get("EXL3_RC_FUSED_DBG", "0") == "1"


def _capturing_rc():
    f = getattr(torch.xpu, "is_current_stream_capturing", None)
    return bool(f and f())


def _rcf_ok(x, sb, sq, out, z):
    E = getattr(torch.ops, "exl3xpu_C", None)
    return (E is not None and hasattr(E, "exl3_gdn_recompute_fused") and x.dtype == torch.float16
            and sb.dtype == torch.float16 and x.stride(-1) == 1 and sb.stride(-1) == 1 and sq.stride(-1) == 1
            and out.dim() == 3 and out.stride(-1) == 1 and out.stride(-2) == out.size(-1)
            and z.dim() == 3 and z.stride(-1) == 1 and z.stride(-2) == z.size(-1))


def _rc_dispatch(core_attn_out, z, qkvz, ba, layer_name):
    import torch.nn.functional as F
    from vllm.forward_context import get_forward_context
    from vllm import _xpu_ops as xo
    from vllm.model_executor.layers.mamba.ops.causal_conv1d import causal_conv1d_update
    fc = get_forward_context()
    self = fc.no_compile_layers[layer_name]
    if not hasattr(self, "_exl3_pool"):
        _rc_alloc(self, core_attn_out.device)
    md_all = fc.attn_metadata
    md = md_all.get(self.prefix) if isinstance(md_all, dict) else None
    if md is None or int(getattr(md, "num_spec_decodes", 0) or 0) == 0 or "buf" not in GDN_RC:
        return xo._exl3_rc_orig_impl(core_attn_out, z, qkvz, ba, layer_name)
    H = self.num_k_heads // self.tp_size; HV = self.num_v_heads // self.tp_size; K = self.head_k_dim; V = self.head_v_dim
    KD, VD = H * K, HV * V; CONV = 2 * KD + VD
    ns = int(md.num_spec_decodes)
    conv_state, ssm_state = self.kv_cache[0], self.kv_cache[1]
    conv_w = self.conv1d.weight.view(self.conv1d.weight.size(0), self.conv1d.weight.size(2))
    pure = int(md.num_prefills) == 0 and int(md.num_decodes) == 0
    if not pure:
        nsi = md.non_spec_token_indx
        nq = qkvz.index_select(0, nsi); nb_ = ba.index_select(0, nsi)
        nT = nq.size(0)
        c2 = torch.zeros(nT, HV, V, dtype=core_attn_out.dtype, device=core_attn_out.device); z2 = torch.empty_like(c2)
        torch.ops._xpu_C.gdn_attention(
            c2, z2, nq, nb_, self.num_k_heads, self.num_v_heads, K, V, conv_state=conv_state, ssm_state=ssm_state,
            conv_weights=conv_w, conv_bias=self.conv1d.bias, activation=self.activation, A_log=self.A_log,
            dt_bias=self.dt_bias, num_prefills=int(md.num_prefills), num_decodes=int(md.num_decodes), num_spec_decodes=0,
            has_initial_state=md.has_initial_state,
            non_spec_query_start_loc=md.non_spec_query_start_loc.to(torch.int32).contiguous(),
            non_spec_token_indx=torch.arange(nT, device=c2.device, dtype=torch.int32),
            non_spec_state_indices_tensor=md.non_spec_state_indices_tensor.to(torch.int32).contiguous(),
            spec_query_start_loc=None, spec_token_indx=None, spec_state_indices_tensor=None, num_accepted_tokens=None,
            num_actual_tokens=nT, tp_size=self.tp_size, reorder_input=not self.gqa_interleaved_layout)
        core_attn_out[:qkvz.size(0)].index_copy_(0, nsi, c2); z[:qkvz.size(0)].index_copy_(0, nsi, z2)
        si = md.spec_token_indx
        sq = qkvz.index_select(0, si); sb = ba.index_select(0, si)
    else:
        nact = int(md.num_actual_tokens)
        sq, sb = qkvz[:nact], ba[:nact]
    T = sq.size(0)
    T1 = GDN_RC.get("T1", 6)
    sidx = md.spec_state_indices_tensor[:ns, 0].to(torch.int32).contiguous()
    qsl = md.spec_query_start_loc[:ns + 1].to(torch.int32).contiguous()
    x = causal_conv1d_update(sq[:, :CONV], conv_state.transpose(-1, -2), conv_w, self.conv1d.bias, self.activation,
                             conv_state_indices=sidx, num_accepted_tokens=md.num_accepted_tokens[:ns],
                             query_start_loc=md.spec_query_start_loc[:ns + 1], max_query_len=T1, validate_data=False)
    st = GDN_RC["buf"]
    gi = GDN_RC["layer_group"].get(self.prefix, 0)
    if _RC_FUSED_IN and _RCF_N[0] < 4 and not pure:
        _RCF_N[0] += 1
        logger.info("exl3: RC-FUSED-IN mixed step check: ok=%s x %s/%s sb %s sq %s out %s/%s z %s/%s", _rcf_ok(x, sb, sq, core_attn_out, z),
                    tuple(x.shape), x.stride(), sb.stride(), sq.stride(), tuple(core_attn_out.shape), core_attn_out.stride(),
                    tuple(z.shape), z.stride())
    if _RC_FUSED_IN and _rcf_ok(x, sb, sq, core_attn_out, z):
        # RC-FUSED-IN (as EXL3_RC_FUSED_IN of the CUDA port): q/k/v split, g/beta and the o/z
        # writes (at the spec rows in mixed steps) inside the recompute kernel: ~15 fewer launches per GDN layer
        # the live parameters every call (fp32 A_log / fp16 dt_bias are used as is, no copy): a converted copy cached at
        # the first call held pre-load values (the first call runs before the weights are in place)
        alog_f = self.A_log if self.A_log.dtype == torch.float32 and self.A_log.is_contiguous() else self.A_log.float().contiguous()
        dtb_h = self.dt_bias if self.dt_bias.dtype == torch.float16 and self.dt_bias.is_contiguous() else self.dt_bias.half().contiguous()
        if not _RCF_N[2] and self.prefix.endswith("layers.0.linear_attn"):
            _RCF_N[2] = 1
            logger.info("exl3: RC-FUSED-IN active")
        oidx = None if pure else si.to(torch.int32)
        dbg = None
        if _RCF_DBG and not pure and _RCF_N[1] < 12 and not _capturing_rc():
            # debug check: the unfused path on cloned state rows / pending pool, compared after the fused call
            _RCF_N[1] += 1
            Tq = x.size(0)
            q_ = x[:, :KD].reshape(Tq, H, K).contiguous(); k_ = x[:, KD:2 * KD].reshape(Tq, H, K).contiguous()
            v_ = x[:, 2 * KD:].reshape(Tq, HV, V).contiguous()
            g_ = (-torch.exp(self.A_log.float()) * F.softplus(sb[:, HV:2 * HV].float() + self.dt_bias.float())).contiguous()
            b_ = torch.sigmoid(sb[:, :HV].float()).contiguous()
            o_ = torch.empty(Tq, HV, V, dtype=core_attn_out.dtype, device=x.device)
            hs = ssm_state.index_select(0, sidx.long()).clone()
            pool_ = [t.clone() for t in self._exl3_pool]
            torch.ops.exl3xpu_C.exl3_gdn_recompute(q_, k_, v_, g_, b_, o_, hs, torch.arange(ns, device=x.device, dtype=torch.int32),
                                                   *pool_, st["npend"][:ns], st["slot"][:ns], st["par"][:ns], qsl, T1, K ** -0.5,
                                                   None, None)
            dbg = (o_, hs, pool_, sq[:, CONV:].reshape(Tq, HV, V).clone())
            if _RCF_N[1] <= 2:
                logger.info("exl3: RC-FUSED-DBG inputs A_log %s %s %s dt_bias %s %s %s rcf %s %s | g_[0,:4] %s a[0,:4] %s alog[:4] %s dtb[:4] %s",
                            self.A_log.dtype, tuple(self.A_log.shape), self.A_log.stride(), self.dt_bias.dtype,
                            tuple(self.dt_bias.shape), self.dt_bias.stride(), tuple(alog_f.shape),
                            tuple(dtb_h.shape), g_[0, :4].tolist(), sb[0, HV:HV + 4].tolist(),
                            alog_f[:4].tolist(), dtb_h[:4].tolist())
        torch.ops.exl3xpu_C.exl3_gdn_recompute_fused(x, sb, alog_f, dtb_h, core_attn_out, z,
                                                     sq[:, CONV:], oidx, H, ssm_state, sidx, *self._exl3_pool,
                                                     st["npend"][:ns], st["slot"][:ns], st["par"][:ns], qsl, T1, K ** -0.5,
                                                     st["bidx"][:ns], st["bblk"][gi, :ns])
        if dbg is not None:
            o_, hs, pool_, z_ = dbg
            d = lambda u, w: (u.float() - w.float()).abs().max().item()
            wb = int(1 - st["par"][0].item()) ; sl = int(st["slot"][0].item())
            logger.info("exl3: RC-FUSED-DBG pg fused[:4] %s unfused[:4] %s", self._exl3_pool[2][wb, sl, 0, :4].tolist(),
                        pool_[2][wb, sl, 0, :4].tolist())
            logger.info("exl3: RC-FUSED-DBG %s T=%d ns=%d: o %.2e z %.2e state %.2e pk %.2e pv %.2e pg %.2e pb %.2e",
                        self.prefix.split(".")[-2], x.size(0), ns, d(o_, core_attn_out.index_select(0, si)),
                        d(z_, z.index_select(0, si)), d(hs, ssm_state.index_select(0, sidx.long())),
                        *[d(u, w) for u, w in zip(pool_, self._exl3_pool)])
        return
    if _RC_FUSED_IN:   # graph audit: which steps miss the fused path (count all; log the first 3 eager + 3 captured)
        _cap = _capturing_rc(); _RCF_FB[_cap] += 1
        if _RCF_FB[_cap] <= 3 * 48 and _RCF_FB[_cap] % 48 == 1:
          logger.info("exl3: RC-FUSED-IN fallback (unfused recompute) #%d capturing=%s: pure=%s ns=%d x %s/%s %s sb %s/%s sq %s out %s/%s z %s/%s",
                    _RCF_FB[_cap], _cap, pure, ns, tuple(x.shape), x.stride(), x.dtype, tuple(sb.shape), sb.stride(), sq.stride(),
                    tuple(core_attn_out.shape), core_attn_out.stride(), tuple(z.shape), z.stride())
    q = x[:, :KD].reshape(T, H, K).contiguous(); k = x[:, KD:2 * KD].reshape(T, H, K).contiguous()
    v = x[:, 2 * KD:].reshape(T, HV, V).contiguous()
    b, a = sb[:, :HV].float(), sb[:, HV:2 * HV].float()
    g = (-torch.exp(self.A_log.float()) * F.softplus(a + self.dt_bias.float())).contiguous()
    beta = torch.sigmoid(b).contiguous()
    o = torch.empty(T, HV, V, dtype=core_attn_out.dtype, device=core_attn_out.device)
    st = GDN_RC["buf"]
    gi = GDN_RC["layer_group"].get(self.prefix, 0)
    torch.ops.exl3xpu_C.exl3_gdn_recompute(q, k, v, g, beta, o, ssm_state, sidx, *self._exl3_pool,
                                           st["npend"][:ns], st["slot"][:ns], st["par"][:ns], qsl, T1, K ** -0.5,
                                           st["bidx"][:ns], st["bblk"][gi, :ns])
    zs = sq[:, CONV:].reshape(T, HV, V)
    if pure:
        core_attn_out[:T].copy_(o); z[:T].copy_(zs)
    else:
        core_attn_out.index_copy_(0, si, o); z.index_copy_(0, si, zs)


# ---- int8_per_token_head KV cache at fp16-or-better speed (vLLM's Triton backend layout and
# write path; attention compute replaced). Uniform decode / spec-verify batches (q_len <= 8, incl. the MTP drafter's
# q_len 1) -> exl3_spec_attn int8 kernel (int8 Q.K and P.V on XMX, per-token scales inline in the cache rows).
# Prefill / mixed batches -> per request: q_len <= 8 -> the same kernel; longer -> block-dequantized fp16 FA2 with a
# log-sum-exp merge (as exl3xpu.fp8kv_prefill). Env EXL3_INT8KV_FAST=0 keeps vLLM's Triton kernels.
_I8_KB = int(os.environ.get("EXL3_INT8KV_KB", "32768"))


def _i8_gather_dequant(cache, pages, bs, start, end):
    """cache: vLLM per-token-head view [blocks, BS, H, 260] int8 (data 0..255, f32 scale at 256..259)
    -> fp16 [end-start, H, 256]."""
    p0, p1 = start // bs, (end + bs - 1) // bs
    blk = cache.index_select(0, pages[p0:p1]).flatten(0, 1)
    off = start - p0 * bs
    x = blk[off: off + (end - start)]
    sc = x[..., 256:260].contiguous().view(torch.float32)              # [n, H, 1]
    return (x[..., :256].to(torch.float32) * sc).to(torch.float16)


def _i8_fa(q, k, v, causal, scale):
    from vllm_xpu_kernels.flash_attn_interface import flash_attn_varlen_func
    Q, L = q.shape[0], k.shape[0]
    cq = torch.tensor([0, Q], dtype=torch.int32, device=q.device)
    ck = torch.tensor([0, L], dtype=torch.int32, device=q.device)
    return flash_attn_varlen_func(q, k, v, Q, cq, L, cu_seqlens_k=ck, causal=causal,
                                  softmax_scale=scale, return_softmax_lse=True)


def _i8_kv(key_cache, value_cache, pages, bs, s0, e0):
    """fp16 K, V [e0-s0, H, 256] for tokens [s0, e0): one fused SYCL gather+dequant (torch chain as fallback)."""
    E = getattr(torch.ops, "exl3xpu_C", None)
    if E is not None and key_cache.dtype == torch.uint8:   # int4_per_token_head (rotated space)
        H = key_cache.shape[2]
        k = torch.empty((e0 - s0, H, 256), dtype=torch.float16, device=key_cache.device)
        v = torch.empty_like(k)
        E.exl3_i4_gather_dequant(key_cache, pages, s0, e0, k, v)
        return k, v
    if E is not None and hasattr(E, "exl3_i8_gather_dequant"):
        H = key_cache.shape[2]
        k = torch.empty((e0 - s0, H, 256), dtype=torch.float16, device=key_cache.device)
        v = torch.empty_like(k)
        E.exl3_i8_gather_dequant(key_cache, pages, s0, e0, k, v)
        return k, v
    return _i8_gather_dequant(key_cache, pages, bs, s0, e0), _i8_gather_dequant(value_cache, pages, bs, s0, e0)


_I8_ONEDNN = os.environ.get("EXL3_INT8KV_ONEDNN", "1") == "1"
_I8_LBUCKET = int(os.environ.get("EXL3_INT8KV_LBUCKET", "256"))   # key length bucket (compiled shapes)
_I8_QBUCKET = int(os.environ.get("EXL3_INT8KV_QBUCKET", "256"))


def _i8_prefill_onednn(E, q, key_cache, pages, seq_len, scale, out):
    """One oneDNN Graph fused SDPA call per layer (upstream EXL3_ONEDNN_ATTN idea, here for the
    int8 cache): K/V gathered + dequantized head-major [Hk, Lp, 256] (keys padded to a bucket, zeros), queries
    head-major [Hq, Qp, 256] padded at the top; causal bottom-right with the real lengths (len_k = seq_len,
    len_q = Qp) so the padded keys are masked and the padded query rows are discarded. GQA natively (Hq = G*Hk):
    K/V read once per group, 1.18-1.26x cutlass FA2 at 3k-60k (microbenchmark)."""
    Q, Hq, D = q.shape
    Hk = key_cache.shape[2]
    # The fused kernel aligns the causal mask by the tensor shapes (row r sees cols <= r + Lp - Qp), not by the
    # len scalars. Keys padded at the end by b = Lp - seq_len are therefore balanced by b padding query rows at the
    # BOTTOM: real query j sits at row a + j and sees keys <= seq_len - Q + j exactly; the padding keys are visible
    # only to the discarded bottom rows. Bucketing (Lp, Qp) keeps the set of compiled shapes small (~0.7 s each, once).
    Lp = (seq_len + _I8_LBUCKET - 1) // _I8_LBUCKET * _I8_LBUCKET
    b = Lp - seq_len
    Qp = (Q + b + _I8_QBUCKET - 1) // _I8_QBUCKET * _I8_QBUCKET
    a = Qp - Q - b
    kb = torch.empty((Hk, Lp, D), dtype=torch.float16, device=q.device)
    vb = torch.empty_like(kb)
    (E.exl3_i4_gather_dequant_hm if key_cache.dtype == torch.uint8 else E.exl3_i8_gather_dequant_hm)(key_cache, pages, seq_len, kb, vb)
    qg = torch.zeros((Hq, Qp, D), dtype=torch.float16, device=q.device)
    qg[:, a:a + Q].copy_(q.permute(1, 0, 2))
    og = torch.empty_like(qg)
    E.exl3_sdpa_len(qg, kb, vb, og, _I8_NONE.setdefault(q.device, torch.empty(0, device=q.device)), True, scale,
                    Lp, Qp)
    out.copy_(og[:, a:a + Q].permute(1, 0, 2))
    return out


_I8_NONE = {}


# 262k context: the oneDNN path dequantizes the WHOLE K/V history to fp16 per layer call (~4 KB/token:
# ~530 MB at 129k), which overran the VRAM headroom at gpu_memory_utilization 0.965 (a test engine died at ~129k with
# UR_RESULT_ERROR_OUT_OF_RESOURCES). Above EXL3_I8_ONEDNN_MAXLEN keys (0 = no limit) prefill attention takes the chunked
# FA path below: _I8_KB-key slices (128 MB at 32k), LSE merge, bounded memory.
_I8_ONEDNN_MAXLEN = int(os.environ.get("EXL3_I8_ONEDNN_MAXLEN", "0"))
_I8_LONG_SEEN = []


def _i8_prefill(q, key_cache, value_cache, pages, seq_len, scale, out):
    if _I8_ONEDNN_MAXLEN and seq_len > _I8_ONEDNN_MAXLEN and not _I8_LONG_SEEN:
        _I8_LONG_SEEN.append(1); logger.info("exl3: I8-LONG-PREFILL chunked FA path above %d keys (first: %d)", _I8_ONEDNN_MAXLEN, seq_len)
    if _I8_ONEDNN and seq_len > q.shape[0] and not (_I8_ONEDNN_MAXLEN and seq_len > _I8_ONEDNN_MAXLEN):   # chunks with a prefix (a lone first chunk is faster on FA2: 0.81x)
        E = getattr(torch.ops, "exl3xpu_C", None)
        if E is not None and hasattr(E, "exl3_sdpa_len") and hasattr(E, "exl3_i8_gather_dequant_hm"):
            return _i8_prefill_onednn(E, q, key_cache, pages.to(torch.int32), seq_len, scale, out)
    Q, bs = q.shape[0], key_cache.shape[1]
    prefix = seq_len - Q
    parts = []
    pages = pages.to(torch.int32)
    if _I8_ONEDNN_MAXLEN and seq_len > _I8_ONEDNN_MAXLEN:
        # long context: running LSE merge (one fp32 accumulator), memory independent of the number of key slices
        acc = m = den = None
        spans = [(s0, min(s0 + _I8_KB, prefix), False) for s0 in range(0, prefix, _I8_KB)] + [(prefix, seq_len, True)]
        for s0, e0, causal in spans:
            o, lse = _i8_fa(q, *_i8_kv(key_cache, value_cache, pages, bs, s0, e0), causal, scale)[:2]
            lse = lse.float()                                      # [Q, Hq] (as the stacked merge below uses it)
            if acc is None:
                acc, m, den = o.float(), lse, torch.ones_like(lse)
                continue
            m2 = torch.maximum(m, lse)
            a1, a2 = torch.exp(m - m2), torch.exp(lse - m2)
            acc.mul_(a1.unsqueeze(-1)).add_(o.float() * a2.unsqueeze(-1))
            den = den * a1 + a2
            m = m2
            del o
        out.copy_((acc / den.unsqueeze(-1)).to(out.dtype))
        return out
    for s0 in range(0, prefix, _I8_KB):
        e0 = min(s0 + _I8_KB, prefix)
        parts.append(_i8_fa(q, *_i8_kv(key_cache, value_cache, pages, bs, s0, e0), False, scale))
    parts.append(_i8_fa(q, *_i8_kv(key_cache, value_cache, pages, bs, prefix, seq_len), True, scale))
    if len(parts) == 1:
        out.copy_(parts[0][0])
        return out
    lses = torch.stack([pp[1] for pp in parts])
    m = lses.amax(0)
    w = torch.exp(lses - m)
    num = sum(pp[0].float() * wi.unsqueeze(-1) for pp, wi in zip(parts, w))
    out.copy_((num / w.sum(0).unsqueeze(-1)).to(out.dtype))
    return out



# graph-replay sweep: int8 decode-attention split count per batch size.
# EXL3_I8_SPLITS="s1,s2,...,sN": request count r uses entry min(r, N) - 1. Default = the old fixed rule (64 at 1, 32 at >= 2).
_I8_SPLITS = [int(x) for x in os.environ.get("EXL3_I8_SPLITS", "64,32").split(",") if x.strip()]


def _i8_splits(nr):
    return _I8_SPLITS[min(nr, len(_I8_SPLITS)) - 1]


# int4 KV: vLLM's int4_per_token_head stores K, V rotated by its unnormalised RHT (H D x, H = 256x256
# Sylvester Hadamard, D = its fixed signs): k_r = 16 R k with R = Hn D orthonormal. q' = q R^T / 16 = q M / 16 gives
# q'.k_r = q.k; the output o_r = 16 R o -> o = o_r M^T / 16 (M = D Hn). One small matmul each (graph-safe).
_I4_M = {}


# optional split table for the MTP drafter's 1-row calls (EXL3_I8_SPLITS_DRAFT;
# unset = the verify table)
_I8_SPLITS_DRAFT = [int(x) for x in os.environ.get("EXL3_I8_SPLITS_DRAFT", "").split(",") if x.strip()]


def _i8_splits_for(nr, draft):
    if draft and _I8_SPLITS_DRAFT:
        return _I8_SPLITS_DRAFT[min(nr, len(_I8_SPLITS_DRAFT)) - 1]
    return _i8_splits(nr)


_I4_S = {}
# I4-ROT: the .so takes the RHT signs (rot=) and rotates in-kernel; EXL3_I4_ROT=0: host matmuls
_I4_ROT_OK = []
_I4_KVW_OK = []


def _i4_kvw():
    if not _I4_KVW_OK:
        ok = os.environ.get("EXL3_I4_KVW", "1") == "1" and hasattr(getattr(torch.ops, "exl3xpu_C", None), "exl3_i4_kv_write")
        _I4_KVW_OK.append(ok)
        logger.info("exl3: I4-KVW (fused int4 KV write) %s", "on" if ok else "off")
    return _I4_KVW_OK[0]


def _i4_rot():
    if not _I4_ROT_OK:
        ok = False
        if os.environ.get("EXL3_I4_ROT", "1") == "1":
            try:
                ok = "rot" in [x.name for x in torch.ops.exl3xpu_C.exl3_spec_attn.default._schema.arguments]
            except Exception:  # noqa: BLE001
                ok = False
        _I4_ROT_OK.append(ok)
        logger.info("exl3: I4-ROT (int4 q/output rotation in the decode kernels) %s", "on" if ok else "off")
    return _I4_ROT_OK[0]


def _i4_signs(dev):
    s = _I4_S.get(dev)
    if s is None:
        from vllm.v1.attention.ops.int4_per_token_head import _get_rht_signs
        s = _I4_S[dev] = _get_rht_signs(256, 0, torch.device("cpu")).float().contiguous().to(dev)
    return s


def _i4_m(dev):
    m = _I4_M.get(dev)
    if m is None:
        from vllm.v1.attention.ops.int4_per_token_head import _get_rht_signs
        h = torch.ones(1, 1, dtype=torch.float32)
        while h.shape[0] < 256:
            h = torch.cat([torch.cat([h, h], 1), torch.cat([h, -h], 1)], 0)
        sg = _get_rht_signs(256, 0, torch.device("cpu")).float()
        m = (sg[:, None] * (h / 16.0)).to(dev)              # M = D Hn  (rows scaled by the signs)
        _I4_M[dev] = m = (m / 16.0, m.t().contiguous() / 16.0)
    return m


def _patch_int8kv_fast():
    if os.environ.get("EXL3_INT8KV_FAST", "1") != "1":
        return
    try:
        from vllm.v1.attention.backends.triton_attn import TritonAttentionImpl
        from vllm.v1.kv_cache_interface import KVQuantMode
    except Exception as e:  # noqa
        logger.warning("exl3: int8 KV fast path not installed (%s)", e)
        return
    if getattr(TritonAttentionImpl, "_exl3_i8_patched", False):
        return
    E = getattr(torch.ops, "exl3xpu_C", None)
    if E is None or not hasattr(E, "exl3_spec_attn"):
        logger.warning("exl3: int8 KV fast path needs exl3_spec_attn; not installed")
        return
    orig = TritonAttentionImpl.forward
    _HAS_VARLEN = hasattr(E, "exl3_spec_attn_varlen") and os.environ.get("EXL3_INT8KV_VARLEN", "1") == "1"
    _VERIFY16 = os.environ.get("EXL3_VERIFY16", "0") == "1"
    # DRAFT-WIN (long-context decode): the MTP drafter's decode-time attention (<= 8 query rows
    # per request) sees page 0 (attention-sink tokens: system prompt start) + the last ceil(EXL3_DRAFT_WIN / block) pages
    # instead of the whole context. RoPE is already in the cached K, so a shortened block-table view + seq_len is exact
    # windowed attention (keys renumbered, causality kept). Device-only, fixed shapes (graph-safe). Drafts only: the
    # target's verify attention is untouched, so outputs stay exact; acceptance may change. 0 = off.
    _DWIN = int(os.environ.get("EXL3_DRAFT_WIN", "0") or 0)
    _DWIN_LOGGED = []

    def _dwin_view(bt, sl, bs):
        nw = (_DWIN + bs - 1) // bs
        npg = bt.shape[1]
        if npg < nw + 2:
            return bt, sl
        sl = sl.to(torch.int32)
        lastp = (sl - 1).clamp(min=0) // bs
        p0 = (lastp - nw + 1).clamp(min=1)
        idx = (p0[:, None] + torch.arange(nw, device=bt.device, dtype=p0.dtype)[None, :]).clamp(max=npg - 1).long()
        win = torch.cat([bt[:, :1], torch.gather(bt, 1, idx)], 1)
        use = lastp > nw
        bt2 = torch.where(use[:, None], win, bt[:, : nw + 1]).contiguous()
        sl2 = torch.where(use, sl - (p0 - 1) * bs, sl).contiguous()
        return bt2, sl2

    _FAST_MODES = (KVQuantMode.INT8_PER_TOKEN_HEAD,) + (
        (KVQuantMode.INT4_PER_TOKEN_HEAD,) if hasattr(KVQuantMode, "INT4_PER_TOKEN_HEAD") and hasattr(E, "exl3_i4_gather_dequant") else ())

    def forward(self, layer, query, key, value, kv_cache, attn_metadata, output, output_scale=None,
                output_block_scale=None):
        if (attn_metadata is not None and getattr(self, "_is_per_token_head_quant", False)
                and self._kv_quant_mode != KVQuantMode.INT8_PER_TOKEN_HEAD and self._kv_quant_mode in _FAST_MODES
                and self.head_size == 256 and output_scale is None and output_block_scale is None):
            # int4: attend in vLLM's rotated space (q in, output back); the KV write path is vLLM's own
            H = self.num_heads
            # decode / verify / drafter steps (every request <= 8 query rows, so only the
            # exl3_spec_attn kernels run) rotate q and the output inside the kernels (no fp32 matmuls, casts, copies)
            if attn_metadata.max_query_len <= 8 and _i4_rot():
                return _fwd8(self, layer, query, key, value, kv_cache, attn_metadata, output, rot=_i4_signs(query.device))
            fm, bm = _i4_m(query.device)
            q_r = (query.view(-1, H, 256).float() @ fm).to(query.dtype)
            o_r = torch.empty_like(output.view(-1, H, 256))
            _fwd8(self, layer, q_r, key, value, kv_cache, attn_metadata, o_r)
            output.view(-1, H, 256).copy_((o_r.float() @ bm).to(output.dtype))
            return output
        return _fwd8(self, layer, query, key, value, kv_cache, attn_metadata, output, output_scale, output_block_scale)

    def _fwd8(self, layer, query, key, value, kv_cache, attn_metadata, output, output_scale=None,
              output_block_scale=None, rot=None):
        if (attn_metadata is None or not getattr(self, "_is_per_token_head_quant", False)
                or self._kv_quant_mode not in _FAST_MODES or self.head_size != 256
                or output_scale is not None or output_block_scale is not None
                or getattr(self, "sliding_window", None) not in (None, (-1, -1))):
            return orig(self, layer, query, key, value, kv_cache, attn_metadata, output, output_scale,
                        output_block_scale)
        key_cache, value_cache = self._pth_key_value_caches(kv_cache)
        md = attn_metadata
        T = md.num_actual_tokens
        nr = md.query_start_loc.shape[0] - 1
        mq = md.max_query_len
        _is_draft = "mtp" in str(getattr(layer, "layer_name", ""))
        if _DWIN and mq <= 8 and _is_draft:
            bt2, sl2 = _dwin_view(md.block_table[:nr], md.seq_lens[:nr], key_cache.shape[1])
            if not _DWIN_LOGGED:
                _DWIN_LOGGED.append(1)
                logger.info("exl3: DRAFT-WIN on: drafter attention = page 0 + last %d tokens (layer %s)", _DWIN,
                            getattr(layer, "layer_name", "?"))
            md = copy.copy(md)
            md.block_table, md.seq_lens = bt2, sl2
        H = self.num_heads
        q = query.view(-1, H, 256)
        out = output.view(-1, H, 256)
        # uniform decode/spec batch; under FULL graphs the batch is padded (q has nr*mq rows, padded requests have
        # seq_len 0 -> the kernel writes zeros), so accept either form; never host-sync on this path
        if mq <= 8 and (T == nr * mq or q.shape[0] == nr * mq):
            n = nr * mq
            E.exl3_spec_attn(q[:n], key_cache, value_cache, md.block_table[:nr], md.seq_lens[:nr], out[:n], mq,
                             float(self.scale), _i8_splits_for(nr, _is_draft), None, None, rot)
            return output
        # varlen: every request with <= 8 query rows (decode / spec verify / the MTP drafter's first pass after
        # different acceptance counts) in ONE launch, no host sync (the old per-request loop
        # synced the GPU each step at >= 2 streams); only requests with > 8 rows (prefill chunks) go per request
        qsl_d = md.query_start_loc[: nr + 1]
        # VERIFY-16: 9..16 query rows per request (long lookup-chain verify,
        # the MTP drafter's first pass after it) without the prefill path: each request is split on the device into two
        # pseudo-requests sharing its block table, rows [0, 8) with seq_len - (rows after them) and the rest with seq_len;
        # the kernel's causal limit (seq_len - q_len + row) then equals the original one for every row. No host sync.
        assert rot is None or mq <= 8, "I4-ROT: only for <= 8 query rows"
        if _HAS_VARLEN and 8 < mq <= 16 and _VERIFY16:
            qi = qsl_d if qsl_d.dtype == torch.int32 else qsl_d.to(torch.int32)
            st, ql = qi[:-1], qi[1:] - qi[:-1]
            first = ql.clamp(max=8)
            sl = md.seq_lens[:nr].to(torch.int32)
            qsl2 = torch.cat([torch.stack([st, st + first], 1).flatten(), qi[-1:]]).contiguous()
            sl2 = torch.stack([sl - (ql - first), sl], 1).flatten().contiguous()
            bt2 = md.block_table[:nr].repeat_interleave(2, dim=0)
            E.exl3_spec_attn_varlen(q, key_cache, value_cache, bt2, sl2, qsl2, out, float(self.scale), 32)
            return output
        if _HAS_VARLEN:
            E.exl3_spec_attn_varlen(q, key_cache, value_cache, md.block_table[:nr], md.seq_lens[:nr],
                                    qsl_d if qsl_d.dtype == torch.int32 else qsl_d.to(torch.int32), out,
                                    float(self.scale), _i8_splits_for(nr, _is_draft), None, None, rot)
            if mq <= 8:
                return output
        qsl = qsl_d.cpu().tolist()
        sl = md.seq_lens[:nr].cpu().tolist()
        for i in range(nr):
            a, b = qsl[i], qsl[i + 1]
            ql = b - a
            if ql <= 0:
                continue
            if ql <= 8:
                if not _HAS_VARLEN:
                    E.exl3_spec_attn(q[a:b], key_cache, value_cache, md.block_table[i: i + 1], md.seq_lens[i: i + 1],
                                     out[a:b], ql, float(self.scale), 64)
            else:
                _i8_prefill(q[a:b], key_cache, value_cache, md.block_table[i], sl[i], float(self.scale), out[a:b])
        return output

    TritonAttentionImpl.forward = forward
    TritonAttentionImpl._exl3_i8_patched = True

    # K smoothing (SageAttention): store K - m (per layer / kv head / channel constant). Q.(K - m) = Q.K - Q.m and Q.m is
    # the same for every key of a query, so softmax is unchanged; m removes the common channel offsets / outliers that
    # eat the per-token int8 range (attention sinks). EXL3_KSMOOTH=<file> applies, EXL3_KSMOOTH_CALIB=<file> measures
    # (eager engine only: accumulates key sums of real tokens and saves {layer: mean[H, D]}).
    ks_path, cal_path = os.environ.get("EXL3_KSMOOTH"), os.environ.get("EXL3_KSMOOTH_CALIB")
    if ks_path or cal_path:
        orig_upd = TritonAttentionImpl.do_kv_cache_update
        kmeans = {}
        if ks_path:
            try:
                kmeans = {k: v for k, v in torch.load(ks_path, map_location="cpu").items()}
                logger.info("exl3: K smoothing on (%d layers from %s)", len(kmeans), ks_path)
            except Exception as e:  # noqa
                logger.warning("exl3: K smoothing file %s not loaded (%s)", ks_path, e)
        cal = {"n": 0}

        def do_kv_cache_update(self, layer, key, value, kv_cache, slot_mapping):
            name = getattr(layer, "layer_name", None)
            if (getattr(self, "_kv_quant_mode", None) in (KVQuantMode.INT8_PER_TOKEN_HEAD, getattr(KVQuantMode, "INT4_PER_TOKEN_HEAD", None))
                    and name is not None):
                if cal_path:
                    valid = slot_mapping[: key.shape[0]] >= 0
                    ks = key[: valid.shape[0]][valid].float()
                    if ks.shape[0] > 8 and os.environ.get("EXL3_KSMOOTH_ROBUST", "1") == "1":
                        # robust: drop token-heads whose key norm exceeds 3x the median (attention sinks / massive
                        # activations would otherwise dominate the mean and offset every normal key)
                        nrm = ks.norm(dim=-1)                                  # [T, H]
                        keep = nrm <= 3.0 * nrm.median(dim=0, keepdim=True).values
                        ks = ks * keep.unsqueeze(-1)
                        cnt = keep.sum(0).double().cpu()                      # [H]
                        acc = cal.setdefault(name, [torch.zeros(key.shape[1:], dtype=torch.float64), torch.zeros(key.shape[1], dtype=torch.float64)])
                        acc[0] += ks.sum(0).double().cpu(); acc[1] += cnt
                        cal["n"] += 1
                        if cal["n"] % int(os.environ.get("EXL3_KSMOOTH_CALIB_EVERY", "1")) == 0:   # every batch (a larger interval lost the tail)
                            torch.save({k: (v[0] / v[1].clamp_min(1).unsqueeze(-1)).float() for k, v in cal.items() if k != "n"}, cal_path)
                        ks = ks[:0]
                    if ks.numel():
                        # per-head count tensors in both modes (a <= 8-row batch after a robust
                        # one crashed the save with max(1, tensor): "Boolean value of Tensor ... ambiguous")
                        acc = cal.setdefault(name, [torch.zeros(key.shape[1:], dtype=torch.float64), torch.zeros(key.shape[1], dtype=torch.float64)])
                        acc[0] += ks.sum(0).double().cpu(); acc[1] += ks.shape[0]
                        cal["n"] += 1
                        if cal["n"] % int(os.environ.get("EXL3_KSMOOTH_CALIB_EVERY", "1")) == 0:   # every batch (a larger interval lost the tail)
                            torch.save({k: (v[0] / v[1].clamp_min(1).unsqueeze(-1)).float() for k, v in cal.items() if k != "n"}, cal_path)
                # I4-KVW: int4 writes in one fused kernel (K smoothing + RHT + pack; bit-exact vs
                # vLLM's reshape_and_cache_int4, whose single_rht runs as a PyTorch butterfly on XPU). EXL3_I4_KVW=0: off
                if (self._kv_quant_mode == getattr(KVQuantMode, "INT4_PER_TOKEN_HEAD", None)
                        and key.dim() == 3 and key.shape[-1] == 256 and key.dtype == torch.float16 and _i4_kvw()):
                    mf = None
                    if name in kmeans:
                        mf = getattr(self, "_exl3_kmf", None)
                        if mf is None or mf.device != key.device:
                            mf = self._exl3_kmf = kmeans[name].to(device=key.device, dtype=torch.float32).contiguous()
                    key_cache, _vc = self._pth_key_value_caches(kv_cache)
                    torch.ops.exl3xpu_C.exl3_i4_kv_write(key, value, key_cache, slot_mapping, _i4_signs(key.device), mf)
                    return
                if name in kmeans:
                    m = getattr(self, "_exl3_km", None)
                    if m is None or m.device != key.device:
                        m = self._exl3_km = kmeans[name].to(device=key.device, dtype=key.dtype)
                    key = key - m
            return orig_upd(self, layer, key, value, kv_cache, slot_mapping)

        TritonAttentionImpl.do_kv_cache_update = do_kv_cache_update
    logger.info("exl3: int8_per_token_head KV: exl3 int8 decode kernel + block-dequantized FA2 prefill")




def register():
    """vllm.general_plugins entry point (runs in every vLLM process)."""
    _patch_fused_mlp()
    _patch_fast_norm()
    _patch_gdn_recompute()
    _patch_int8kv_fast()
    if os.environ.get("EXL3_DS_SUB", "0") == "1" or os.environ.get("EXL3_DRAFT_GRAPH", "0") == "1":
        from . import draftgraph   # DS-SUB / DRAFT-GRAPH (before the graded-vocab propose wrapper)
        draftgraph.patch()
    if os.environ.get("EXL3_DRAFT_VOCAB"):
        _patch_mtp_draft_logits()
    if os.environ.get("EXL3_VLLM_PATCHES", "1") != "0":
        from . import vllm_patches
        vllm_patches.apply_all()
    return None
