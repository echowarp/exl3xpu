# exl3xpu, Arc Pro B70 production fork

This branch extends [exl3xpu](https://github.com/sybil-solutions/exl3xpu) (EXL3 weights in vLLM on Intel Arc, by
0xSero, MIT) with the kernels and plugin features used to serve Qwen3.8-27B (EXL3 4.0 bpw, MTP speculative decoding,
int4 KV cache, contexts up to 196k tokens, several concurrent agents) on one Intel Arc Pro B70 (BMG-G31, 32 GB).
The base is upstream commit 15ded2f; everything upstream documents in `README.md` still applies.

Every feature is a separate commit and is switched by an environment variable (pass them with
`scripts/serve.py ... --set env.NAME=value`; serve.py applies the model.yaml `env:` over the process environment).
`models/qwen3.8-27b-exl3-4.00bpw/model.yaml` is the production configuration: int4_per_token_head KV, MTP k = 7 with
the dynamic depth schedule, 196,608 context, and the production values of the switches below, except the ones
that belong to the memory layer (`EXL3_LAZY_VISION`,
keeper, RAM tier). `EXL3_GDN_PORT=1` loads the package's `gdnchunk.so`. GDN recompute is on by default; the companion vLLM runner
follows the plugin's switch, and the plugin turns it off (with a warning) when that runner is not installed and,
unless `EXL3_GDN_RECOMPUTE=1` is set, for models that are not EXL3. The companion vLLM branch's own switches default
to these production values too, but only while this plugin serves an EXL3 model (its `docs/llmt`).
Kernel features that are missing from a build, or vLLM internals that are not there, leave the stock path in place
with a log line.

Some features need companion patches to vLLM itself (the vLLM 0.26 XPU branch that goes with this fork; marked
"vLLM patch" below). The plugin imports nothing from the KV-tier client packages (`llmt_kv`, `llmt_kvbroker_arc`): the
production engine loads those only through vLLM's `kv_transfer_config` (OffloadingConnector + a broker spec).

## Building

    scripts/build_arc.sh            # all libraries into exl3xpu/
    pip install --no-deps --no-build-isolation -e .

Production toolchain: Intel oneAPI DPC++/C++ Compiler 2025.3.3 (2025.3.3.20260319), oneDNN 3.9.1, PyTorch
2.12.0+xpu (all from `intel/llm-scaler-vllm:0.26.0-b2`, the upstream base image) with the GPU runtime upgraded to
compute-runtime (NEO) 26.31 / IGC 2.40 / Level Zero 1.32. All libraries are JIT builds (`-fsycl-targets=spir64`).
`docker/Dockerfile.arc` is a recipe for such an image.

| Library | Source | Flags | Provenance of the flags |
|---|---|---|---|
| `exl3xpu/_C.so` | `csrc/exl3_ops.sycl` (+ headers) | `scripts/build_ext.sh` line + `-DEXL3_ALL_CODEBOOKS -DEXL3_K5_UNROLL24=4 -DEXL3_K5_MB128 -DEXL3_FOLD -DEXL3_HADOUT_B=8 -DEXL3_DPAS40_SMALLGRF -DEXL3_DPAS40_NT=1 -DEXL3_I4_U4DPAS -DEXL3_I4_K2T -DEXL3_SPEC_EPIU` (+ `-DEXL3_DNNL` when oneDNN is present) | **Verified**: a rebuild with `scripts/build_arc.sh c` in the production image is byte-identical to the production binary. |
| `exl3xpu/gdnchunk.so` | `gdnchunk/csrc/gdn_chunk_ops.sycl` | `-O3 -ffast-math -fsycl-device-code-split=per_kernel -DGDNC_PF=1 -DGDNC_PF_DIST=2 -DGDNC_PF_QK256=1 -DGDNC_CONV2=1 -DGDNC_VNNI2D=1 -DGDNC_PLANPAR=1 -DGDNC_CONV_VEC=4` | **Verified**: a rebuild has the identical device code (SPIR-V section) as the production binary; host code differs only by embedded source paths. |
| `exl3xpu/ng.so` | `ng/ng_ops.sycl` | `-O3` (no `-ffast-math`), no `-D` options | **Verified**: a rebuild has the identical device code (SPIR-V section) as the production binary; host code differs only by embedded source paths. |

Verified on the production image (icpx 2025.3.3, PyTorch 2.12.0+xpu): `_C.so` rebuilds byte for byte; `gdnchunk.so` and
`ng.so` rebuild with identical device code. Run the tests in `tests/` and `tests/arc/` after building with any other
toolchain.

## Scope: kernels here, memory policy in AllBroker

This fork is the fast math for the Arc: ESIMD EXL3 GEMM/GEMV, fused MLP / RMSNorm, verify attention, int8 / int4 KV,
sampling, GDN, LoRA decode kernels, and the plugin that registers them with vLLM. It does not decide what stays in GPU
memory. That is the memory layer, [AllBroker](https://github.com/echowarp/AllBroker) (`llmt_kv` package):

| Memory-policy piece | Where it lives | What this fork keeps |
|---|---|---|
| Weight keeper (weights stay resident across engine restarts, Level Zero IPC) | AllBroker `llmt-keeper` + engine-side `whx_loader` (XPU backend) | nothing; the XPU keeper works on the model's parameters |
| No-mmap safetensors loader | AllBroker `llmt_kv.nommap_loader` | `exl3xpu/nommap_loader.py`, the same code; the plugin uses `llmt_kv`'s copy when it is installed and falls back to the bundled one |
| KV lending (LoRA / vision), RAM KV tier | AllBroker (`llmt_kv`, vLLM plugin + connector) | the LoRA kernels (`lorak/`, `csrc/lora_esimd.h`), which the lend manager calls through `EXL3_LORA_KERNELS=llmt_lora_kernels` |
| Lazy vision tower | AllBroker's broker-managed version (VRAM budget shared with KV lending) | `exl3xpu/lazyvis.py`: the standalone device-only version production runs today (`EXL3_LAZY_VISION`). Kept for compatibility; with AllBroker installed, its module takes over (the standalone copy then stays off by itself) |

GPU-specific settings are configuration, not code: every feature is an `EXL3_*` (LoRA kernels: `LLMT_LK_*`) variable
listed below with its production value, and the per-shape split-K table is a data file (`EXL3_SHAPE_P_FILE`).

CPU-only checks (no Arc GPU): `python tests/cpu/test_cpu_plugin.py` (or pytest; no-mmap loader round trip, loader
preference, entry point, LoRA-hook fallback, defaults; `LLMT_KV_NOMMAP=<path>` also compares the bundled loader with
AllBroker's). Everything else in `tests/` needs the GPU.

### LoRA decode kernels (`lorak/`)

The LoRA terms run inside the kernels the EXL3 base linear launches anyway (`exl3xpu_C.linear_lora`,
`exl3xpu_C.mlp_small_lora`). `lorak/llmt_lora_kernels.py` is the hook module: `EXL3_LORA_KERNELS=llmt_lora_kernels`
makes AllBroker's lend manager call it; each call returns `None` (stock punica path) outside what the kernels cover
(rank 16, fp16, small-M rows). `LLMT_LK_LINEAR` (1) and `LLMT_LK_MLP` (1) switch the two paths. `lorak/run.sh` runs
the GPU checks (`test_lorak.py check|bench|step`).

## Environment variables

Production = the Qwen3.8-27B Arc B70 service (`kv_cache_dtype=int4_per_token_head`, MTP k = 7 with a dynamic depth
schedule, `max_model_len` 196,608). "default (x)" = not set there, code default x.

| Variable | Production | What it does |
|---|---|---|
| `EXL3_MAX_MB` | 128 | largest DPAS row block of the decode GEMM (MB96 / MB128 one-pass tiles for 65-128 rows; needs `-DEXL3_K5_MB128`) |
| `EXL3_SHAPE_P_FILE` | `models/qwen3.8-27b-exl3-4.00bpw/shape_p_c.json` | per-shape split-K count (and NT override) table for the decode linears, keyed `"N,K,MB"`; measured on a B70 |
| `EXL3_Q8_FOLD_MIN_M` | default (off) | int8 prefill GEMM: output Hadamard folded into the int8 weights from this many rows (setting it turns the fold on) |
| `EXL3_FUSED_MLP` | default (1) | fused decode MLP (`mlp_auto`: 3..8 rows in one op; prefill MLP middle fused for int8 prefill) |
| `EXL3_FAST_NORM` | default (1) | fused residual add + RMSNorm for decode batches (replaces `custom_esimd_kernels_vllm.esimd_fused_add_rms_norm_batched`) |
| `EXL3_FAST_NORM_WG` | default (256) | work-group size of that kernel |
| `EXL3_SPEC_HP` | 1 | HEAD-PACK: plain decode attention with q_len * G <= 8 packs the G query heads into the DPAS rows (the MTP drafter's 1-row calls) |
| `EXL3_SPEC_HP_BMAX` | 8 | HEAD-PACK up to this many sequences (default 1) |
| `EXL3_SPEC_HP_S / EXL3_SPEC_HP_MIN` | default (8 / 1) | HEAD-PACK split count max(MIN, S / B) |
| `EXL3_SPEC_VARR` | default (1) | VARR: 4-tile V loads in the int8 verify kernel for >= 2 sequences |
| `EXL3_SPEC_VARR_MAXL` | 84000 | VARR only for sequences up to this many tokens (in-kernel gate; default 40000) |
| `EXL3_SPEC_VAT` | default (0) | use the VARR kernel instance with plain loads (register-allocation variant) |
| `EXL3_INT8KV_FAST` | default (1) | int8 / int4 per-token-head KV: decode, verify and drafter attention on `exl3_spec_attn`, prefill on oneDNN SDPA or block-dequantized FA2 (Triton attention backend) |
| `EXL3_INT8KV_VARLEN` | default (1) | all requests with <= 8 query rows in one varlen launch, no host sync |
| `EXL3_I8_SPLITS` | 80,40,64,40 | decode-attention split count per number of requests (entry min(r, N) - 1) |
| `EXL3_I8_SPLITS_DRAFT` | default (unset = `EXL3_I8_SPLITS`) | split table for the MTP drafter's calls |
| `EXL3_I8_ONEDNN_MAXLEN` | 60000 | above this many keys, prefill attention uses the chunked FA path (running LSE merge, bounded memory) instead of one oneDNN SDPA call |
| `EXL3_INT8KV_ONEDNN` | default (1) | oneDNN SDPA for prefill chunks with a prefix |
| `EXL3_INT8KV_KB` | default (32768) | keys per dequantized FA2 slice |
| `EXL3_INT8KV_LBUCKET / _QBUCKET` | default (256 / 256) | oneDNN SDPA shape buckets |
| `EXL3_DRAFT_WIN` | default (0 = off) | DRAFT-WIN: the drafter's attention sees page 0 + the last N tokens (drafts only; output unchanged) |
| `EXL3_VERIFY16` | default (0) | 9..16-row verify on the decode kernel (split into two pseudo-requests) |
| `EXL3_KSMOOTH` | path to a K-mean table (see "Data files") | K smoothing: store K - mean (per layer / KV head / channel) in the quantized KV cache; softmax unchanged |
| `EXL3_KSMOOTH_CALIB` | unset | calibration: accumulate key means of the traffic (eager engine) and save the table to this path |
| `EXL3_KSMOOTH_CALIB_EVERY` | default (1) | save the table every N batches |
| `EXL3_KSMOOTH_ROBUST` | default (1) | calibration ignores token-heads with key norm > 3x the median (attention sinks) |
| `EXL3_I4_ROT` | default (1) | int4 KV: q / output rotation (vLLM's RHT) inside the decode kernels instead of host matmuls |
| `EXL3_I4_PREROT` | default (1) | rotate q once per call in a pre-kernel (instead of per split thread) |
| `EXL3_I4_KVW` | default (1) | I4-KVW: fused int4 KV write (K smoothing + RHT + pack, bit-exact vs vLLM's reshape_and_cache_int4); active with `EXL3_KSMOOTH` |
| `EXL3_FAST_TOPKP` | default (1) | fused top-k / top-p mask `exl3_topk_topp_mask` (vLLM patch: `topk_topp_sampler.py`) |
| `EXL3_DRAFT_VOCAB` | 512-block draft vocabulary (see "Data files") | pruned MTP draft head (upstream feature) |
| `EXL3_DRAFT_VOCAB_FAR` | 128-block draft vocabulary | GRADED-DRAFT: smaller head for draft positions >= `EXL3_DRAFT_FAR_FROM` |
| `EXL3_DRAFT_FAR_FROM` | 2 | first 0-based draft position that uses the far head |
| `EXL3_DRAFT_Q4` | 64 | DRAFT-Q4: the draft heads as plain int4 (group size 64) for torch's XPU int4 GEMV; drafter only |
| `EXL3_DRAFT_Q4_SCOPE` | default (all) | `far`: int4 only for the far head |
| `EXL3_DRAFT_Q4_KEEP` | default (0) | keep the EXL3 slice next to the int4 head |
| `EXL3_FAST_DRAFT_SAMPLE` | default (1) | fused draft sampler `exl3_draft_sample` (vLLM patch: `llm_base_proposer.py`) |
| `EXL3_DS_SUB` | 1 | DS-SUB: draft tokens sampled from the pruned head's compact logits (`exl3_draft_sample_sub`, device RNG counter) |
| `EXL3_DRAFT_GRAPH` | default (0) | DRAFT-GRAPH: draft passes 2..k as one captured XPU graph (needs DS-SUB) |
| `EXL3_DRAFT_GRAPH_MAX / _CHECK` | default (24 / 0) | graph count cap; debug comparison against the eager loop |
| `EXL3_DRAFT_DYN` | 256 | DRAFT-DYN: up to N extra far-head rows per request (its most frequent prompt tokens outside the static head; int4 rows gathered from a host copy of lm_head) |
| `EXL3_DRAFT_DYN_NEAR` | 128 | the same for the near head |
| `EXL3_DRAFT_BIAS` | 0 | drafter logit bias on prompt tokens (0 = off) |
| `EXL3_GDN_RECOMPUTE` | default (1) | GDN recompute: MTP spec decoding with one committed SSM state per request (num_speculative_blocks = 0); vLLM patch: gpu_model_runner GDN-RC hooks |
| `EXL3_GDN_RC_SLOTS` | default (16) | pending-pool slots |
| `EXL3_RC_FUSED_IN` | 1 | RC-FUSED-IN: q/k/v split, gating and o/z writes inside the recompute kernel |
| `EXL3_RC_FUSED_DBG` | default (0) | debug: compare against the unfused kernel |
| `EXL3_GDN_RPS` | 4 | state rows per sub-group in the recompute kernels |
| `EXL3_GDN_PORT` | 1 (= `<package>/gdnchunk.so`; or a path) | GDN-PORT: non-spec prefill rows on the 16-row chunked GDN prefill (deterministic); vLLM patch: `gdn_attn.py` metadata `pos0` |
| `EXL3_GDN_PORT_TOKINDX` | 1 | mixed steps: the port reads/writes the rows in place (no gather/scatter) |
| `EXL3_GDN_PORT_TRACE` | default (0) | debug: per-step checksums in the log |
| `EXL3_DYN_SD_FULL` | 2 | 1: keep FULL decode graphs with a dynamic speculative depth schedule; 2: also capture FULL graphs for every depth of the schedule |
| `EXL3_UNIFORM_SPEC` | 1 | uniform-decode dispatch only for all-spec-decode steps (a prefill chunk of exactly 1 + k tokens is not run as a spec decode) |
| `EXL3_DET_LINEAR` | 1 | unquantized fp16 linears through oneDNN with deterministic primitives (custom op) |
| `EXL3_DETERMINISTIC` | default (0) | torch.use_deterministic_algorithms for the whole process (test only) |
| `EXL3_FAST_NAN_GUARD` | 1 | fused NaN guard `exl3ng::nan_guard_rows` for the runner's `_nan_guard` (vLLM patch: gpu_model_runner NaN guard) |
| `EXL3_NG_SO` | default (`exl3xpu/ng.so`) | NaN-guard library |
| `EXL3_LAZY_VISION` | 1 | vision tower in pinned host RAM, loaded (or streamed block by block) only for image requests; vLLM patch: `qwen3_vl.py` calls `exl3xpu.lazyvis.install()` |
| `EXL3_LAZY_VISION_MODE / _TTL / _MARGIN_MB` | default (auto / 120 / 256) | resident vs streamed policy, release after idle seconds, free-memory margin |
| `EXL3_NOMMAP_LOAD` | default (1) | safetensors weights read with preadv, no mmap |
| `EXL3_NOMMAP_DROP` | default (0) | also drop the read pages from the page cache |
| `TORCH_CAP_MIB` | 29100 | `tools/oor/llmt_memprobe.py`: cap the torch XPU caching allocator (MiB) so the Level Zero driver keeps headroom |
| `MEMPROBE_S` | 30 | allocator statistics to stderr every N seconds (`MEMPROBE=0` disables the thread) |

Other production settings outside this package: vLLM `gpu_memory_utilization` 0.94,
`PYTORCH_ALLOC_CONF=max_split_size_mb:512,garbage_collection_threshold:0.85`,
`attention_backend=TRITON_ATTN`, `max_num_batched_tokens` 8192; the MTP depth schedule
`num_speculative_tokens_per_batch_size=[[1,4,7],[5,5,5],[6,6,4],[7,8,3],[9,256,2]]` with
`draft_sample_method=probabilistic` (stock vLLM 0.26 fields).

## Data files you supply

Nothing derived from traffic ships with this fork. The data files default to public builds shipped in the model
directory: the draft vocabularies `public-512.json` / `public-128.json` (`scripts/public_draft_vocab.py`; production used
a pair ranked on its own traffic) and the K-smoothing table `public_r.pt` (`scripts/public_ksmooth.py`, inputs and sha256
in `public_r.pt.inputs.json`). How to make your own:

- **Draft vocabularies** (`EXL3_DRAFT_VOCAB`, `EXL3_DRAFT_VOCAB_FAR`): upstream's generator ranks 128-token vocab
  blocks by token frequency on a text corpus. Rank on text that looks like your traffic:

      python3 scripts/draft_vocab.py --model <checkpoint> --blocks 512 --text '<your corpus>/*.txt' --out draft-512.json
      python3 scripts/draft_vocab.py --model <checkpoint> --blocks 128 --text '<your corpus>/*.txt' --out draft-128.json

  `models/qwen3.8-27b-exl3-4.00bpw/draft_vocab.json` (upstream, 512 blocks) works as a generic near head.
  Without traffic of your own, `scripts/public_draft_vocab.py` builds the shipped near + far pair (512 + 128 blocks) from a fixed
  public input set (Python stdlib, PyPI package docs, json-mode-eval + chat-template tool calls, MT-Bench, HumanEval,
  Gutenberg) and records every input's sha256 in the json. On held-out Qwen3.8-27B outputs (GSM8K greedy, IFBench) its
  512 / 128-block files cover the same token share as vocabularies ranked on production traffic (99.3 / 94.8 % and
  96.5 / 81.6 %):

      python3 scripts/public_draft_vocab.py --model <checkpoint> --data <dir with the public files> --blocks 512 128 \
          --out draft-{n}.json
- **K-smoothing table** (`EXL3_KSMOOTH`): run the engine eager (`enforce_eager`) with `EXL3_KSMOOTH_CALIB=ksmooth.pt`
  and the int8 or int4 KV cache over representative prompts; the table (a `{layer: mean[H, D]}` torch file) is
  rewritten after every batch. Then serve with `EXL3_KSMOOTH=ksmooth.pt` (and without `EXL3_KSMOOTH_CALIB`).
  `scripts/public_ksmooth.py` feeds such an engine the public input set of `public_draft_vocab.py` (the shipped
  `public_r.pt`). Measured on the Arc (int4 KV, 126,099 held-out GSM8K tokens, vs a table built on production agent
  traffic): NLL +0.037 %, top-5 KL 4.2e-3, single-stream decode 192 vs 193 tok/s; without a table NLL +0.43 % and
  142 tok/s (the fused int4 KV write needs a table).
- **Split-K table** (`EXL3_SHAPE_P_FILE`): `models/qwen3.8-27b-exl3-4.00bpw/shape_p_c.json` is a kernel timing
  measurement on a B70 for the model's matrix shapes (no traffic data); other GPUs or models need their own
  timings (`{"N,K,MB": P}` or `{"N,K,MB": [NT, P]}`).

### Tuning for your own traffic

- **Out of the box** the public draft vocabularies (`public-512.json`, `public-128.json`) and the public K-smoothing table
  (`public_r.pt`) in `models/qwen3.8-27b-exl3-4.00bpw/` are the model.yaml defaults. Measured against the tables production builds from its own agent traffic: the public K-smoothing
  table gives NLL +0.037 %, top-5 KL 4.2e-3 and single-stream decode 192 vs 193 tok/s (Arc B70, int4 KV, 126,099
  held-out GSM8K tokens; no table: NLL +0.43 %, 142 tok/s); the public vocabularies cover the same share of held-out
  model outputs (512 / 128 blocks: 99.3 / 94.8 % on GSM8K, 96.5 / 81.6 % on IFBench). Decode speed with the public
  vocabularies has not been measured yet.
- **Draft vocabulary from your traffic:** rank on text like your model's outputs with exl3xpu
  `scripts/draft_vocab.py --model <checkpoint> --blocks 512 --text '<your outputs>/*.txt' --out draft-512.json` (and
  `--blocks 128` for the far head). `scripts/public_draft_vocab.py` is the same ranking over its fixed public input set.
  Drafts only: the output distribution does not change, only how many draft tokens are accepted.
- **K-smoothing table from your traffic:** run ONE engine eager (`enforce_eager`) with `EXL3_KSMOOTH_CALIB=/path/ksmooth.pt`
  and without `EXL3_KSMOOTH`, on the KV dtype you serve with, over representative traffic or a replay of it. The table is
  rewritten as it goes (exl3xpu: every `EXL3_KSMOOTH_CALIB_EVERY` batches, default 1; exl3cuda: every 64 calls). Then
  restart serving with `EXL3_KSMOOTH=/path/ksmooth.pt` (and without `EXL3_KSMOOTH_CALIB`). exl3xpu
  `scripts/public_ksmooth.py` is this procedure fed with public text. The file format is the same on both GPUs.
- **Caveats:** calibration needs an eager engine (no graphs, so it is slower while it runs): use a separate engine or a
  quiet period. A new table invalidates cached KV and the RAM-tier entries (the table's checksum is part of the tier's
  compatibility tag), so switch at a planned restart; engines that share a RAM tier must all use the same table. This
  is how production does it.

## Layout added by this fork

| Path | What |
|---|---|
| `scripts/build_arc.sh` | builds `_C.so`, `gdnchunk.so` and `ng.so` with the production options |
| `csrc/` | ESIMD/SYCL kernels of `_C.so` (upstream files extended; new headers per feature) |
| `models/qwen3.8-27b-exl3-4.00bpw/shape_p_c.json` | B70 split-K table for the decode linears |
| `exl3xpu/draftdyn.py` | DRAFT-DYN row selection / batch assembly / sample fix-up (CPU-testable) |
| `exl3xpu/draftgraph.py` | DS-SUB and DRAFT-GRAPH |
| `gdnchunk/` | GDN chunked prefill port (Apache-2.0, derived from Magnitude; own `LICENSE` and `NOTICE`) |
| `ng/ng_ops.sycl` | fused NaN guard library |
| `exl3xpu/lazyvis.py` | lazy vision tower |
| `exl3xpu/nommap_loader.py` | weight loading without mmap |
| `tools/oor/` | torch XPU allocator cap + statistics (`.pth`-loaded) |
| `tests/arc/` | int4/int8 decode-attention and KV-write kernel tests and microbenchmarks (synthetic tensors) |
| `docker/Dockerfile.arc` | image with the newer GPU runtime and this fork's libraries |
| `THIRD_PARTY_NOTICES.md` | licenses of third-party code in this fork |
