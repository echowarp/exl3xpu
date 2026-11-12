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

Other production settings outside this package: vLLM `gpu_memory_utilization` 0.94,
`attention_backend=TRITON_ATTN`, `max_num_batched_tokens` 8192; the MTP depth schedule
`num_speculative_tokens_per_batch_size=[[1,4,7],[5,5,5],[6,6,4],[7,8,3],[9,256,2]]` with
`draft_sample_method=probabilistic` (stock vLLM 0.26 fields).

## Layout added by this fork

| Path | What |
|---|---|
| `scripts/build_arc.sh` | builds the kernel libraries with the production options |
| `csrc/` | ESIMD/SYCL kernels of `_C.so` (upstream files extended; new headers per feature) |
| `THIRD_PARTY_NOTICES.md` | licenses of third-party code in this fork |
