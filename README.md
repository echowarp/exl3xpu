# exl3xpu

A fork of [exl3xpu](https://github.com/sybil-solutions/exl3xpu) (EXL3 weights in vLLM on Intel Arc, as a plugin with
native ESIMD kernels, by 0xSero) that serves Qwen3.8-27B (EXL3 4.0 bpw) to several tool-calling agents at once on one
Intel Arc Pro B70 (32 GB). Upstream is archived; this fork continues it. On upstream's own benchmark (its
`bench/sweep.py --corpus real`, thinking on, T = 0.7) it decodes 125.0 tok/s at 1 stream and 530.2 at 16 (prose;
upstream 91.2 and 365), prefills a 32K-token prompt at 2,823 tok/s (upstream 2,124), and holds 500,193 tokens of int4
KV cache where upstream held 272,570 tokens of fp8 KV.

![Arc Pro B70, upstream exl3xpu's own benchmark](https://github.com/echowarp/AllRunner/raw/main/benchmarks/charts/out/arc-upstream-panel.png)

## What's different from upstream

Each feature is its own commit with its own `EXL3_*` switch. Every switch with its production value, the build flags
and the data files: [README-ARC.md](README-ARC.md). Release notes: [RELEASE-ARC.md](RELEASE-ARC.md).

| Change | What it does | Measured effect |
|---|---|---|
| int4 KV attention | decode, verify and drafter attention on int4 per-token-head KV, fused int4 KV write, q rotation inside the kernels | KV cache 500,193 tokens vs 277,333 with int8 KV and 146,400 with 16-bit KV; KL vs BF16 0.0080 vs 0.0060 with int8 KV (RTX 5090 engine, same weights) (1) |
| Decode path | MB96 / MB128 GEMM tiles with a measured split-K table, fused MLP and RMSNorm, head-packed drafter attention, MTP k = 7 with a per-batch-size depth schedule, per-request draft rows (DRAFT-DYN), int4 draft heads, DS-SUB | aggregate decode, prose: 125.0 / 187.2 / 311.1 / 451.5 / 530.2 tok/s at 1 / 2 / 4 / 8 / 16 streams vs upstream 91.2 / 151.4 / 262.6 / 357 / 365; code: 84.9 vs 66.1 at 1 stream, 532.7 vs 350 at 16 (1) |
| Prefill | GDN chunked-prefill port; int8 / int4 KV prefill attention on oneDNN SDPA or, for long prompts, chunked FA | cold prefill 2,722 (2K), 3,153 (8K), 2,823 (32K) tok/s; upstream 2,421 (4K), 2,124 (32K) (1) |
| K smoothing | stores K minus a per-channel mean in the quantized KV cache; enables the fused int4 KV write | shipped public table vs one built on production traffic: NLL +0.037 %; no table: NLL +0.43 %; table + fused write: 1-stream aggregate 111.8 -> 119.7 tok/s (2) |
| LoRA decode kernels (`lorak/`) | LoRA terms inside the EXL3 small-M linear and the fused decode MLP | LoRA decode 179.9 tok/s (1 stream), 161.2 (2 streams) with AllBroker's lend manager (2) |
| GDN recompute, fused NaN guard | MTP with one committed SSM state per request; one fused kernel for the model runner's NaN guard | not published separately |
| Serving config | [`models/qwen3.8-27b-exl3-4.00bpw/model.yaml`](models/qwen3.8-27b-exl3-4.00bpw/model.yaml): int4 KV, MTP k = 7, 196,608-token context, 16 sequences, 8192-token prefill chunks, up to 32 images per request; public draft vocabularies and K-smoothing table as defaults | the configuration measured above |

Sources: (1) AllRunner [benchmarks/README.md](https://github.com/echowarp/AllRunner/blob/main/benchmarks/README.md);
(2) [RELEASE-ARC.md](RELEASE-ARC.md). For scale, llama.cpp SYCL Q4_K_M on the same card: 20-37 tok/s single stream (1).

What we measured on the Arc and dropped (kernel layouts, deeper MTP drafts, Hadamard fusion, drafter window and more,
with the numbers and the Xe2 reason, including why MTP verify stops at 8 rows): [docs/arc-dead-ends.md](docs/arc-dead-ends.md).

## Use it

[AllRunner](https://github.com/echowarp/AllRunner) builds the Arc image from this repo, echowarp/vllm and AllBroker at
pinned commits, and starts it:

```sh
git clone https://github.com/echowarp/AllRunner && cd AllRunner
docker build -f docker/Dockerfile.arc -t allrunner-arc .
launchers/arc-b70.sh        # engine on 127.0.0.1:8000
```

Read AllRunner's [launchers/README.md](https://github.com/echowarp/AllRunner/blob/main/launchers/README.md) first:
weights (`turboderp/Qwen3.8-27B-exl3`, branch `4.00bpw`), the required hash-seed file, start order, and the router in
front.

By hand, inside a vLLM 0.26 XPU environment with oneAPI 2025.3 and the companion vLLM build below:

```sh
scripts/build_arc.sh                       # _C.so, gdnchunk.so, ng.so with the production options
pip install --no-deps --no-build-isolation -e .
python3 scripts/serve.py models/qwen3.8-27b-exl3-4.00bpw --gpu 0 --print   # shows the vllm serve command
```

Tested on `intel/llm-scaler-vllm:0.26.0-b2` (oneAPI DPC++ 2025.3.3, PyTorch 2.12.0+xpu) with compute-runtime 26.31 /
IGC 2.40 / Level Zero 1.32; on that image `scripts/build_arc.sh` rebuilds the production `_C.so` byte for byte.

Checks: `python tests/cpu/test_cpu_plugin.py` runs without a GPU. On the Arc: `tests/arc/` (int4 / int8 attention and
KV-write kernels on synthetic tensors) and upstream's bit-exactness gate `tests/test_bitexact_xpu.py` (every kernel
path vs exllamav3's `reconstruct()`). The benchmark above: AllRunner
[benchmarks/harness/arc-exl3xpu-panel.sh](https://github.com/echowarp/AllRunner/blob/main/benchmarks/harness/arc-exl3xpu-panel.sh).

## Requires / works with

- [echowarp/vllm](https://github.com/echowarp/vllm): the XPU build is on branch [`xpu`](https://github.com/echowarp/vllm/tree/xpu) (vLLM 0.26 with
  Intel's XPU patch plus our scheduler, speculative-decoding and KV-offload commits). Several switches need it; with
  stock vLLM they log and keep the stock path.
- [echowarp/AllBroker](https://github.com/echowarp/AllBroker): the memory layer (weight keeper, lazy vision tower, KV
  lending for LoRA and vision, host-RAM KV tier shared with the RTX 5090). Optional; this repo is the kernels only.
- [echowarp/AllRunner](https://github.com/echowarp/AllRunner): images, launchers, router and benchmarks.
- [echowarp/b70-linux](https://github.com/echowarp/b70-linux): optional host patches and tools for the Arc Pro B70
  (power limit, fans, monitoring).
- RTX 5090 counterpart: [echowarp/exllamav3](https://github.com/echowarp/exllamav3) (`exl3cuda`).

## Upstream

Based on [sybil-solutions/exl3xpu](https://github.com/sybil-solutions/exl3xpu) commit 15ded2f by 0xSero (archived),
which decodes [exllamav3](https://github.com/turboderp-org/exllamav3)'s EXL3 format. Upstream's
[README](https://github.com/sybil-solutions/exl3xpu#readme) has its fp8-KV results, its published image, how to add
another model, and the original design; its design notes and tuning log are kept here in
[docs/DESIGN.md](docs/DESIGN.md) and [docs/PROGRESS.md](docs/PROGRESS.md).

## License

MIT, as upstream ([LICENSE](LICENSE)). `gdnchunk/` is Apache-2.0 (its own [LICENSE](gdnchunk/LICENSE) and
[NOTICE](gdnchunk/NOTICE)); other third-party code: [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
