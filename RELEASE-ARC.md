# exl3xpu Arc B70 fork, 2026-11-16: first public release

Branch `publish-20261113` on upstream 15ded2f. Features and switches: `README-ARC.md`.

## What is in it

- ESIMD kernels and plugin features used in production on one Arc Pro B70: EXL3 decode GEMM tiles (MB96 / MB128) with
  a measured split-K table, W8A8 prefill linears, int4 / int8 per-token-head KV attention, GDN prefill port and
  recompute, MTP drafting with pruned / int4 draft heads (DRAFT-DYN, DS-SUB), fused NaN guard.
- LoRA terms inside the EXL3 small-M linear and the fused decode MLP (`lorak/`).
- `scripts/public_draft_vocab.py` and `scripts/public_ksmooth.py`, and the public files they built in
  `models/qwen3.8-27b-exl3-4.00bpw/` (draft vocabularies `public-512.json` / `public-128.json`, K-smoothing table
  `public_r.pt`): the model.yaml defaults.
- Scope: kernels only. Memory policy (keeper, lazy vision tower, KV lending, RAM tier) is AllBroker's `llmt_kv`; the
  no-mmap loader is taken from `llmt_kv` when installed.

## How it was validated

- CPU tests (`tests/cpu`): 9 passed.
- Production engine (same kernels and switches), upstream's own `bench/sweep.py --corpus real` panel: aggregate decode
  125.0 / 187.2 / 311.1 / 451.5 / 530.2 tok/s (prose, 1 / 2 / 4 / 8 / 16 streams) vs upstream's 91.2 / 151.4 /
  262.6 / 357 / 365; cold prefill 2,722 (2K), 3,153 (8K), 2,823 (32K) tok/s. KV capacity 497,302 int4 tokens.
- This branch's plugin with the published vLLM XPU branch and AllBroker (overlay on the production image): hooks
  native, KV 497,302 tokens, LoRA decode 179.9 tok/s (1 stream) / 161.2 (2 streams), 6 x 60k stress 0 OOM.
- Public K-smoothing table vs production's traffic-built one (int4 KV, 126,099 held-out GSM8K tokens): NLL +0.037 %,
  top-5 KL 4.2e-3; no table: NLL +0.43 %. Speed: the table plus the fused int4 KV write it enables add about +6-7 % on a fast host (Arc B70: 1-stream aggregate 111.8 -> 119.7 tok/s, per-stream decode 139.2 -> 148.5) and up to +35 % per-stream decode on a slow, host-bound CPU (142 -> 192 tok/s on HumanEval prompts).
- Public draft vocabularies, coverage of held-out Qwen3.8-27B outputs: 512 / 128 blocks cover 99.3 / 94.8 % (GSM8K
  greedy) and 96.5 / 81.6 % (IFBench), the same token share as vocabularies ranked on production traffic.

## Known limitations

- The numbers above used production's traffic-ranked draft vocabularies; speed with the shipped public pair is not
  measured yet.
- Engines that share the RAM tier must use the same K-smoothing table (its hash is in the tier's compatibility tag).
- The Docker image (AllRunner `docker/Dockerfile.arc`) has not been rebuilt from the published commits yet.
- Several switches need the companion vLLM XPU branch (echowarp/vllm `publish-20261113-xpu`); with stock vLLM they log
  and keep the stock path.
