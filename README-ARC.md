# exl3xpu, Arc Pro B70 production fork

This branch extends [exl3xpu](https://github.com/sybil-solutions/exl3xpu) (EXL3 weights in vLLM on Intel Arc, by
0xSero, MIT) with the kernels and plugin features used to serve Qwen3.8-27B (EXL3 4.0 bpw, MTP speculative decoding,
int4 KV cache, contexts up to 196k tokens, several concurrent agents) on one Intel Arc Pro B70 (BMG-G31, 32 GB).
The base is upstream commit 15ded2f; everything upstream documents in `README.md` still applies.

Every feature is a separate commit and is switched by an environment variable (pass them with
`scripts/serve.py ... --set env.NAME=value`; serve.py applies the model.yaml `env:` over the process environment).
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

| Library | Source | Flags | Provenance of the flags |
|---|---|---|---|
| `exl3xpu/_C.so` | `csrc/exl3_ops.sycl` (+ headers) | `scripts/build_ext.sh` line + `-DEXL3_ALL_CODEBOOKS -DEXL3_K5_UNROLL24=4 -DEXL3_K5_MB128 -DEXL3_FOLD -DEXL3_HADOUT_B=8 -DEXL3_DPAS40_SMALLGRF -DEXL3_DPAS40_NT=1 -DEXL3_I4_U4DPAS -DEXL3_I4_K2T -DEXL3_SPEC_EPIU` (+ `-DEXL3_DNNL` when oneDNN is present) | **Reconstructed**: the build log of the production binary records no flags; the list comes from the build scripts of that build round. Not verified by a rebuild + checksum comparison. |

Nothing here reproduces the production binaries byte for byte; rebuild and run the tests in `tests/` before relying
on a build.

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

Other production settings outside this package: vLLM `gpu_memory_utilization` 0.94,
`attention_backend=TRITON_ATTN`, `max_num_batched_tokens` 8192; the MTP depth schedule
`num_speculative_tokens_per_batch_size=[[1,4,7],[5,5,5],[6,6,4],[7,8,3],[9,256,2]]` with
`draft_sample_method=probabilistic` (stock vLLM 0.26 fields).

## Data files you supply

Nothing derived from traffic ships with this fork. The production values above point at files you make yourself:

- **Draft vocabularies** (`EXL3_DRAFT_VOCAB`, `EXL3_DRAFT_VOCAB_FAR`): upstream's generator ranks 128-token vocab
  blocks by token frequency on a text corpus. Rank on text that looks like your traffic:

      python3 scripts/draft_vocab.py --model <checkpoint> --blocks 512 --text '<your corpus>/*.txt' --out draft-512.json
      python3 scripts/draft_vocab.py --model <checkpoint> --blocks 128 --text '<your corpus>/*.txt' --out draft-128.json

  `models/qwen3.8-27b-exl3-4.00bpw/draft_vocab.json` (upstream, 256 blocks) works as a generic near head.
- **K-smoothing table** (`EXL3_KSMOOTH`): run the engine eager (`enforce_eager`) with `EXL3_KSMOOTH_CALIB=ksmooth.pt`
  and the int8 or int4 KV cache over representative prompts; the table (a `{layer: mean[H, D]}` torch file) is
  rewritten after every batch. Then serve with `EXL3_KSMOOTH=ksmooth.pt` (and without `EXL3_KSMOOTH_CALIB`).
- **Split-K table** (`EXL3_SHAPE_P_FILE`): `models/qwen3.8-27b-exl3-4.00bpw/shape_p_c.json` is a kernel timing
  measurement on a B70 for the model's matrix shapes (no traffic data); other GPUs or models need their own
  timings (`{"N,K,MB": P}` or `{"N,K,MB": [NT, P]}`).

## Layout added by this fork

| Path | What |
|---|---|
| `scripts/build_arc.sh` | builds the kernel libraries with the production options |
| `csrc/` | ESIMD/SYCL kernels of `_C.so` (upstream files extended; new headers per feature) |
| `models/qwen3.8-27b-exl3-4.00bpw/shape_p_c.json` | B70 split-K table for the decode linears |
| `THIRD_PARTY_NOTICES.md` | licenses of third-party code in this fork |
