# What did not work on the Arc Pro B70

This page lists kernel and serving ideas that were built or probed for this fork, measured, and dropped, with the
numbers and the hardware reason for each, so that others don't have to repeat them. Where a related idea did ship,
the entry says so. Dates are the measurement dates (2026).

**Setup.** Qwen3.8-27B, EXL3 4.0 bpw, served by vLLM 0.26 (XPU) with this fork on one Intel Arc Pro B70 (Xe2,
BMG-G31, 32 GB). MTP speculative decoding: up to 7 draft tokens (k = 7), so a verify step scores 8 query rows per
request. KV cache int8 per token-head until 2026-11-03, int4 per token-head since. The model has 16 full-attention
layers (6 query heads share each KV head; head dim 256); the other layers are GDN.

**How to read the numbers.**
- Kernel timings are **us per layer call** (one attention layer, one decode step), from microbenchmarks on synthetic
  tensors unless marked otherwise. B = number of sequences in the call, "196k" = keys per sequence.
- Engine timings are **steps/s**, **ms per step**, **tok/step** (accepted tokens per verify step, including the bonus
  token) and **tok/s**, from a synthetic long-context decode benchmark (one long prompt, cached prefix, production
  sampling) unless marked "real traffic".
- **Timing probes** are kernel builds with parts removed (for example no K loads). They produce wrong output by design;
  they only bound what a part costs.

## The hardware facts that explain most results

1. **16 KB of registers per thread.** Xe2 has 64-byte registers. The verify-attention kernel runs in large-GRF mode:
   256 registers = 16 KB per hardware thread, at the price of half as many threads per vector engine (4 instead of
   8). Whatever a thread keeps across the key loop (Q, the output accumulator O, scores, K/V tiles in flight) has to
   fit in those 16 KB, or the compiler spills to memory inside the loop.
2. **The DPAS tile is 8 rows.** The XMX (DPAS) instruction is a systolic multiply with a repeat count of at most 8,
   so one instruction covers M ≤ 8 rows of the left matrix. 9 to 16 rows need a second row tile (twice the
   instructions and twice the accumulator registers). In the decode GEMM the rows of a tile share the loaded and
   trellis-decoded weight tile; splitting rows across row blocks repeats that decode (forcing 16-row blocks at 18 rows:
   58.9 ms vs 36.5 ms of decode linears per step, 2026-10-19).
3. **SLM is not a cheap staging area here.** Reading K/V tiles from shared local memory measured about 3x the time of
   the 2D block loads it would replace (those loads hit L1/L3): 1,591 → 4,610 us at 196k, B=1 (int8 KV, 2026-10-31).
   Every design that shares data between threads through SLM pays this.
4. **Verify attention is bound by load latency and message issue, not by ALU or DRAM bandwidth.** Evidence:
   - timing probes (int4 KV, 196k, B=1): no K loads −24 %, no V loads −12 %, no K and V loads −44 %, no loads at all
     −70 %; compute alone is about 30 % of the call, unpacking int4 ≤ 5 %;
   - the kernel reaches 235–277 GB/s on int4 bytes against a practical ~531 GB/s; the int8 kernel showed the vector
     engines 82 % idle waiting on the scoreboard, at 60–70 % occupancy (2026-10-13);
   - more messages make it slower even when they carry less data (L1 prefetch, row split below), and so do fewer,
     wider messages (K layout re-check below): the limit is the latency each thread has in flight.

   Most of the load cost is not DRAM traffic: each of the 6 query-head threads of a KV head re-issues the same K/V
   messages. A probe that issues each K/V message once per work-group saves 21–31 % (int4, 100–196k).

## MTP width and draft depth: why verify stops at 8 rows (k ≤ 7)

A verify step scores 1 + k query positions per request. Production uses k = 7, i.e. exactly 8 rows, and a dynamic
depth schedule that lowers k as streams are added. Going to 9 or more rows was measured several ways; none pays at
long context.

**Why 8 rows fit and 9 don't.** The verify kernel gives each query head its own thread, which loops over the keys
with Q and O in registers. For 8 rows and head dim 256:

| per-thread state | size |
|---|---|
| Q, int8, 8 rows x 256 | 2 KB |
| O accumulator, fp32, 8 rows x 256 | 8 KB |
| scores for a key block, DPAS accumulators, K/V tiles in flight | the rest of the 16 KB |

That fills the 16 KB register file, with no room left to double-buffer the loads. 8 rows is also one DPAS row tile.
A 16-row O alone is 16 KB, so 9–16 rows cannot stay register-resident in this layout. Today 9–16 rows run on the
**16-row split path** (`EXL3_VERIFY16`): the request is split into two 8-row pseudo-requests, each on its own threads,
and the second one re-issues all K/V loads. Since the kernel is load-bound (fact 4), rows 9–16 cost about as much as
rows 1–8: **1.75–1.85x the 8-row call**.

**Kernel layouts tried to make 9+ rows (or sharing) cheaper:**

| layout | result | why |
|---|---|---|
| Row split: 2 threads x 4 rows (10-31, int8) | bit-exact, 22–29 % slower | twice the threads issue twice the K/V load messages (each block read by 12 threads instead of 6); halving the registers bought no overlap |
| PAIR: 2 heads x half the dims per thread (11-01, int8) | 3–8x slower (196k B=1: 1,587 → 8,697 us) | second head's Q (2 KB) + O (8 KB) + per-head scores exceed 16 KB, the key loop spills; fp16 O does not help because Q doubles |
| Producer/consumer: 2/3/6 score threads + 8 P·V threads, P through SLM (11-01, int8) | mostly slower; best cells −15 % (8k B=1), −6 % (196k B=1); B=2/4 8–79 % slower | the SLM exchange plus one barrier per 32-key block eat almost all of the bound |
| Two-pass 16-row kernel (11-17, int4) | mode 1 bit-identical to the split path, but 2.1–2.5x **slower than the split path** | one thread per head holds 16 rows by doing O in two halves of 128 dims (O 8 KB) and re-reading K; still spills 3.0–4.3 KB inside the key loops in every variant (16-row S 2 KB + four DPAS accumulators 2 KB + O 8 KB + K/V tiles > 16 KB even with Q moved to SLM) |

Two-pass kernel vs today's paths, us per layer call (int4 KV, best split per arm):

| ctx, B | 8 rows (production) | 16 rows, split path | 16 rows, two-pass (mode 1 / mode 2) |
|---|---|---|---|
| 36k, 1 | 308 | 334 | 784 / 709 |
| 100k, 1 | 468 | 807 | 1,895 / 1,850 |
| 196k, 1 | 900 | 1,569 | 3,704 / 3,419 |
| 196k, 2 | 1,569 | 2,917 | 7,953 / 7,379 |
| 196k, 4 | 3,043 | 5,588 | 14,286 / 13,183 |

Even without spills, two passes would only drop half of the V messages (V is 9–22 % of a call) while halving the
threads in flight.

**The rest of the step grows too.** Above 8 rows the decode linears leave the 8-row DPAS tile: 28.0 ms of decode
linears per step at 6 rows vs 30.0 ms at 12 (2026-10-19), and a step going from 7 to 13 rows costs about +2.0 ms in the
linears (2026-10-20). With int4 KV the q / output rotation also moves from inside the kernel to the host for more than
8 rows, and every extra draft position is one more serial drafter pass (~1.5 ms each).

**What deeper drafts could gain.** Per-position conditional acceptance on production traffic (340k verify steps,
production sampling) is 0.849 / 0.812 / 0.784 / 0.793 / 0.801 / 0.809, i.e. flat at about 0.80 after position 3.
On real agent traffic at depth 7 (4.52 tok/step), draft positions 8–12 would add about +0.7 accepted tokens per step
(~+15 %), against +20–40 % step time.

**Measured, 2026-11-17** (int4 KV, greedy, 1 and 2 streams; tok/s with tok/step and steps/s in brackets):

| depth | 64k | 128k | 190k | 2 streams agg tok/s, 64k / 128k / 190k |
|---|---|---|---|---|
| 7 (production) | 95.3 (4.19, 22.8) | 131.2 (6.53, 20.1) | 106.3 (5.89, 18.05) | 217.9 / 211.5 / 165.8 |
| 8 | 86.2 (4.38, 19.7) * | 119.5 (7.28, 16.4) −9 % | 90.5 (6.40, 14.1) * | 192.8 * / 179.6 −15 % / 131.5 * |
| 10 | 70.5 (3.73, 18.9) * | 133.6 (8.62, 15.45) +2 % | 98.1 (7.26, 13.5) * | 194.8 * / 192.2 −9 % / 102–143 * |
| 12 | 143.2 (7.94, 18.05) * | 132.5 (8.82, 15.0) +1 % | 98.8 (7.59, 13.0) * | 188–230 * / 189.8 −10 % / 115.9 * |

\* the greedy text differs from depth 7 (the verify GEMMs run at a different row count, so near-ties flip), so these
cells are not paired; only the 128k column produced the same text in every arm.

- **Depth 8 is a cliff:** the 190k step goes from 55.4 to 70.9 ms (+15.5 ms: 16 layers x ~0.7 ms of extra attention on
  the split path, the host-side rotation, one more drafter pass). Each further depth adds ~1.5 ms.
- At 190k depth 8–12 is 7–15 % slower than depth 7 at one stream; at 2 streams (128k, paired) 9–15 % slower.
- The benchmark text is copy-heavy and accepts far more than real traffic (8.6–8.8 tok/step at depth 10–12); even there
  deeper drafts only break even, at 128k (+1–2 %).

**Conclusion.** On the B70, k = 7 (8 verify rows) is the depth where one query head per thread, one DPAS row tile and
one set of K/V messages all still fit. Past it, attention nearly doubles, and the extra drafter passes alone eat most
of what positions 8–12 could add, so even a free 16-row attention would not make deeper drafts pay at long context.
What would change it:
- **cheaper drafting**: fewer launches per draft pass, or one drafter pass for all positions (parallel drafting, which
  needs a retrained drafter), so that extra positions cost little;
- **a GPU with more registers per thread**, so that one thread can keep 16 rows (16 KB of fp32 O alone) resident and
  the second set of K/V messages disappears.

## Long-context verify attention

At 190k and one stream, verify attention is 16 calls of 0.88 ms = 14 ms, about 25 % of the 56 ms step (about 30 % with
the drafter's attention); below ~64k it is under 15 %.

**int8 KV: six designs, none ships** (us per layer call, QL = 7):

| design | result |
|---|---|
| L1 prefetch of the next blocks (10-22) | +10 / +18 / +40 % slower at distance 1 / 2 / 4; message-issue bound |
| D-split: 4 threads x 64 dims, partial Q·K reduced through SLM (10-29) | 15–90 % slower; SLM reduction plus barriers cost more than the redundant K loads |
| SLM staging of K/V (10-31) | the read side alone is 3x slower (196k: 1,591 → 4,610) |
| Row split, 2 threads x 4 rows (10-31) | bit-exact, 22–29 % slower (doubles the load messages) |
| PAIR, 2 heads x half dims per thread (11-01) | 3–8x slower (spills, 16 KB GRF) |
| Producer/consumer, 2/3/6 producers + 8 consumers, P through SLM (11-01) | best cells −15 % (8k B=1), −6 % (196k B=1); mostly slower |

On Xe2's 16 KB of registers, one query head per thread is the only register-resident layout; sharing K/V between
the heads of a KV head needs SLM (about 3x a 2D load) or lower-precision accumulators.

**What did ship: int4 KV.** Halving the bytes per key is what moved long-context decode. Verify attention per layer
call, int4 (2026-11-17) vs int8 (2026-10-29):

| ctx | int4 B=1 | int4 B=4 | int8 B=1 | int8 B=4 |
|---|---|---|---|---|
| 36k | 0.175 ms (217 GB/s) | 0.589 ms (258 GB/s) | | |
| 100k | 0.453 ms (234 GB/s) | 1.569 ms (269 GB/s) | | |
| 196k | 0.881 ms (235 GB/s) | 3.091 ms (268 GB/s) | 1.588 ms | 4.727 ms |

At 196k that is −45 % (B=1) and −35 % (B=4). End to end, one stream:

| context | int4 step ms (steps/s) | int8 step ms |
|---|---|---|
| 4k | 39.1 (25.6) | 39 |
| 64k | 44.1 (22.7) | 48 |
| 128k | 50.5 (19.8) | 58 |
| 190k | 56.2 (17.8) | 70 |

The cost per 1k of context fell from 0.165 to 0.092 ms per step. The fixed ~39 ms (GEMMs, GDN, drafter) is now the
larger share, also at long context.

**What is left in the int4 kernel** (2026-11-17, ms per call, int4 probes):

| probe | 196k B=1 | 100k B=1 | 196k B=4 |
|---|---|---|---|
| production | 0.888 | 0.455 | 3.04 |
| no K loads | 0.655 (−24 %) | 0.370 (−19 %) | 2.37 (−22 %) |
| no V loads | 0.779 (−12 %) | 0.413 (−9 %) | 2.38 (−22 %) |
| no K and no V loads | 0.499 (−44 %) | 0.292 (−36 %) | 1.84 (−39 %) |
| no loads at all | 0.270 (−70 %) | 0.151 (−67 %) | 1.00 (−67 %) |
| each K/V message once per work-group | 0.613 (−31 %) | 0.361 (−21 %) | 2.29 (−25 %) |

The last row is the bound for any "share once" design: −21..−31 % of verify attention at 100–196k, which is about
+7–8 % steps/s at 190k, +4–5 % at 128k and ~+2 % at 64k. Not built (see "Open ideas").

**K load layout re-check (int4, 196k, B=1 / B=4):** production (two 8-dword transposed loads every 4 chunks) 0.869 /
3.006 ms; one 8-dword load per 2 chunks 0.948 / 3.036 (+9 / +1 %); one 16-dword line per 4 chunks, half the messages,
1.018 / 3.348 (+17 / +11 %). Fewer, wider messages are slower.

## Other dead ends

**Hadamard fusion in the decode GEMM.** EXL3 rotates every linear's input and output with a Hadamard transform; in
decode these are 444 small launches, 1.1–1.9 ms per step. Fusing them into the DPAS GEMM loses: under graphs at 6 rows,
27.8 ms of decode linears per step unfused vs 37.4 ms with the input rotation fused and 49.5 ms with both fused
(2026-10-19). The decode GEMM is sensitive to extra ALU work (the trellis decode already competes for it), and the
fused output path serializes the split-K tail. *Shipped instead:* the output Hadamard folded into the GEMM for
≤ 8 rows (`EXL3_FOLD`, decode linears −1.0..−1.1 % at 4–8 rows, not bit-exact: 1–2 fp16 ulp), batched loads in the
output Hadamard (`EXL3_HADOUT_B=8`, about −1.3 % of a layer's linear time at 5 rows), and the fused decode MLP. The
fold at 24-row tiles spilled and ran 2x slower. In prefill, folding the output Hadamard into the int8 weights kept
quality but gained only ~+0.3 % end to end (off by default, `EXL3_Q8_FOLD_MIN_M`); the fused prefill MLP middle shipped
(30k prefill +4.3 %).

**GEMM register double-buffering** (load the next weight words while decoding the current ones, 2026-09-30): 29.3 ms
vs 28.1 ms at 6 rows; spills at 12 rows. No registers to spare at large tiles.

**Activation sparsity (TEAL / CATS style).** Not applicable to EXL3: the Hadamard rotations (incoherence processing)
make every linear's input dense and Gaussian-like in the stored basis. Zeroing the smallest 40 % of such a vector
drops ~3.6 % of its energy, ~19 % relative output error per layer. It would need an unrotated weight format and new
kernels.

**Drafter attention window** (`EXL3_DRAFT_WIN`: the MTP drafter attends to page 0 + the last N tokens; drafts only,
output unchanged). On the synthetic benchmark an 8k window gave +4 % at 96–128k and +11 % at 190k (2026-11-01). On
a 200k-token retrieval suite on the Arc it lost about a third of the accepted tokens: 6.07 tok/step dense vs 4.08 with
the 8k window (2026-11-16); the same window on the RTX 5090 port lost 20 % per-request decode on real traffic at
24–52k. The drafter needs the far context. Keep it unset.

**VARR without the length gate** (4-tile V loads in the int8 verify kernel for ≥ 2 sequences at any length): −10..−12 %
verify attention at B ≥ 2 on long int8 contexts, soak-tested clean. Moot on int4 KV: the int4 kernel's V load already
uses 4-tile messages, and the int4 dispatch never uses the VARR instance, so `EXL3_SPEC_VARR_MAXL` changes nothing there.

**fp8 KV** (2026-10-12). 2x capacity, but no speed: Xe2 has no fp8 conversion instruction, so fp8 → half is software
(about 6 cycles per register); a naive fp8 verify kernel was 2–3.7x slower. *Shipped instead:* int8, then int4, per
token-head KV.

**Native int8 prefill attention** (2026-10-13 / 10-23). Vector-instruction bound: XMX 13 % busy, ~400 ALU instructions
per 32-key block vs ~190 in the cutlass fp16 kernel; 0.55–0.6x cutlass, and a variant with fp16 P·V still 0.38x
production. Production prefill attention uses oneDNN SDPA instead.

**Tree drafting** (a second candidate token at the first rejected depth, 2026-10-20). A second candidate is accepted in
25–26 % of rejected steps (+6.3–6.9 % tok/step if a sibling sat at every depth), but each sibling costs a verify row
every step (7 → 13 rows ≈ +7 % step) and 3+ streams cross the 16-row tile. Gating by draft confidence did not separate
good rows. Net ≈ +2 % at 1 stream; shelved.

**Two engines on one Arc** (render and compute engine queues, weights shared, 2026-10-31). Real concurrency, but the
render queue has hardware priority: the compute queue gets about 25 % while the other is busy, and no software knob
changed that. It is a priority choice (decode 67 % / prefill 25 %, or decode 14 % / prefill 88 %), not more capacity,
at about 6 GB of host RAM per extra engine and a split KV cache.

**Driver / runtime knobs** (2026-10-20): the compute runtime's graph-memory knob −10 % graph decode; Unified Runtime v1
batched command lists −3 % decode.

**llama.cpp Vulkan** (2026-09-29): 18.4 t/s decode and 350 t/s prefill on this model, vs ~85 / ~2,600 for this stack at
the time. The custom ESIMD kernels already run the int8 GEMM near hardware peak, so a different API cannot win much.

**AOT compilation** (`spir64_gen`, BMG-G31, large GRF, 2026-09-29): IGC crashed on 24 kernels. The libraries ship as
JIT builds.

## Open ideas, not built

- **Share-once K/V through SLM as raw int4 bytes.** Split the 6 head threads of a work-group into 2 groups of 3 that
  each issue half of the K/V messages and pass the other half through SLM as packed int4 (1 KB per 32-key block per
  KV head, half the bytes of the int8 SLM probe that lost 3x). Gate first with a read-side SLM probe on int4 bytes; build
  only if it costs < ~25 % of the 2D loads it replaces. Bound: ≤ +8 % steps/s at 190k, ≤ +2 % at 64k.
- **A lower-launch drafter.** The drafter is host-bound: with profiler syncs, the GPU was busy 6.0 ms of a 17.6 ms
  draft wall at k = 7, ~390 kernels per step (reported 2026-11-14). That figure needs re-measuring without syncs
  before any work; the plan is to capture the draft passes, verify and acceptance bookkeeping as one device-resident
  graph if the exposed host time is ≥ ~20 % of the step. This is also the lever that would make deeper drafts worth
  revisiting.
