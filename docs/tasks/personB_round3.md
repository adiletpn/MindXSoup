# Dilya — round 3 (31.10): faster NF4, memory without losing time

Start: `git checkout main && git pull`, read `README.md`, `docs/results.md`, `docs/QA_NOTES.md`,
`docs/memory_audit_solution.md`. Each task in its own branch (`budget`, `norm16`, …) and its own PR
into `main`. Rules as before (AGENTS.md §4–5): no `harness/`/`tests/`/`task/`, no hidden rows,
every claim backed by a Kaggle T4 run; for a speed claim an A/B on one machine.

## Task 0 — TOP PRIORITY: take the tied embedding / lm_head off the GPU (−445 MiB of a 1 739 MiB peak)

Context (31.10, 03:00): with every layer recomputed (`BUDGET_TOKENS = 0`) our peak is **1 739 MiB,
8.25×, 10/10**. The tied embedding / `lm_head` (151 936 × 1 536 fp16 = 445 MiB) is now a quarter of
it. The organizers allowed exactly this (QA_NOTES #47): keep frozen embed/lm_head in pinned host
memory for the whole run and stream the head through the GPU in chunks every step; copies and
syncs count in the time; values, compute dtypes and outputs unchanged.

Design (all in `solution/`, your branch `offload`):

1. **Where the weight lives.** After Soup prepares the model (our `embedding.py` already keeps it
   fp16), move it to a pinned CPU tensor; leave a tiny placeholder on the GPU so
   `get_input_embeddings()`/`get_output_embeddings()` still exist. Input lookup: gather the
   batch's rows on the CPU (`index_select` on the pinned table, ids copied D2H once per step) and
   copy the `[N, 1536]` fp16 rows H2D; output cast to fp32 as the base does.
2. **Head in ONE pass over the vocabulary** (no second pass, so the weight crosses PCIe once per
   step, ≈ 445 MB ≈ 37 ms at 12 GB/s): for each vocab chunk `Wc` (e.g. 8 192 rows), double-buffered
   H2D on a side stream with CUDA events, compute `Zc = H16 @ Wc^T` (cuBLAS, fp16) and with one
   Triton kernel per chunk update per row: running max `m`, `s = Σ e^(z−m)`, `t = Σ e^(z−m)·z`
   (entropy), argmax, the target logit; then the gradient accumulator like FlashAttention's
   output: `A ← A·e^(m_old−m_new) + Pc @ Wc` with `Pc = e^(Zc−m_new)` in fp16 (rows rescaled
   before the matmul or A in fp32). At the end `dH = (A/s − W[target]) / denominator` for labelled
   rows (gather `W[target]` rows on the CPU like the embedding), `lse = m + log s`, loss
   `lse − z_target`, entropy `lse − t/s`. Same outputs as our `fused_ce.py` today.
3. **Anything else** that calls the head or the embedding (eval, generation, Soup's tests with
   fake modules) must still work: fall back by bringing the weight to the device for that call.

Acceptance: per-token loss / entropy / argmax / dH vs the current `fused_head_ce` within fp16
rounding (test with the real vocab size); full `--repeats 2` with **10/10**, eval loss within
±0.05% of the base, peak ≤ 1 350 MiB, time not worse than +2.5 s against `BUDGET_TOKENS = 0`
without offload, on two seeds. Report the A/B in the PR. If PCIe makes it slower than +2.5 s, try
larger chunks, more buffers, or overlapping the copy of the first chunks with the last decoder
layer's forward.

## Why memory, and how we judge a change

Scores are `35·ln(s)/ln(s_best) + 30·ln(m)/ln(m_best)`. In the chat the best claims are about
45 s / 1.3 GB (s ≈ 1.85, m ≈ 11). With those, **−5% time ≈ +3 points and −20% memory ≈ +3
points**. So a change that trades time for memory one-for-one earns nothing; we want memory that
costs almost no time, or time that costs no memory. For every candidate report a row:

| variant | time s (2 repeats) | peak MiB | s | m | points = 35·ln(s)/ln(1.85) + 30·ln(m)/ln(11) | checks |

## Task 1 — map the recomputation budget (≈2 h, mostly waiting for runs)

`solution/recompute.py`, `BUDGET_TOKENS` (now 1024). Run `full --repeats 2` for **768, 512, 256,
0** (0 = every layer of every batch reruns). Fill the table above, plot points vs budget (a PNG in
`docs/`), and say which budget wins and by how much. If the best is within 0.5 points of 1024,
we keep 1024 (less risk).

## Task 2 — RMSNorm saves fp32 inputs: halve that (≈3–4 h)

Each non-recomputed layer keeps two fp32 `[N, 1536]` norm inputs for backward (≈0.35 GB at the
peak, see the audit). Ideas, measure each:

1. **fp16 copy for backward.** Forward stays bit-identical; backward recomputes `x_hat` from an
   fp16 copy of x. Organizers allowed compact activation storage (QA_NOTES #32) but warned about
   fp16 overflow: check `max|x|` of the residual stream over a real run (log it once per layer)
   and refuse (keep fp32) above a safe bound, e.g. 30 000.
2. **Residual add fused into the norm.** Qwen does `h = h + attn_out` then `norm(h)`, and
   `h = h + mlp_out` then `norm(h)`. One Triton kernel that adds and normalizes saves one pass over
   memory and one tensor. It must keep returning the fp32 sum `h` for the next residual.

Deliverables: kernel(s) in `solution/rmsnorm.py` (or a new file you own), tests in
`solution/test_rmsnorm.py` against PyTorch fwd+bwd (also with large values near the bound),
switch in `FEATURES`, A/B on one machine (time, peak), full `--repeats 2` with 10/10 and loss
curve (check 6) at ≤0.05%.

## Task 3 (if time remains) — what attention keeps

After your GQA fix SDPA keeps q, the repeated k and v (12 heads), the output and the log-sum-exp
for each row. Measure what attention saves per layer now (a saved-tensors hook, like
`tools/memory_audit.py`), and propose a way to keep k/v at 2 heads for backward (repeat inside a
custom autograd function and recompute the repeat in backward). Prototype only if it saves
≥100 MiB at the peak.

## Priority order for the day

**Task 0 first** (biggest score gain now), then **Task 4** (safe, pure speed, also grows our kernel share), then **Task 1** (runs in the
background while you code), then **Task 5** (the biggest memory lever, riskier), then Task 2,
then Task 3. Merge each one separately as soon as it has its own 10/10 run.

## Task 4 — make `nf4_unpack_kernel` run at memory speed (HIGH VALUE, ≈3–4 h)

Facts: it is our largest kernel, **8.2% of the training time** (≈2.2 s of a ≈27 s timed loop,
≈395 launches per step: every Linear4bit in forward, in backward and again in recomputed layers).
It is purely memory-bound: it reads 0.5 byte and writes 2 bytes (fp16) per weight. Writing
≈5–6 GB per step at the T4's ≈320 GB/s should take ≈18–20 ms per step; it takes ≈36 ms now, so
there is ≈1 s of wall time to win, with no change in values or memory.

Ideas to try (measure each with a microbenchmark on the real shapes — 1536×1536, 256×1536,
1536×8960, 8960×1536 — against bandwidth = bytes moved / time, then A/B in the full run):

1. **Wider loads and stores.** Load the packed bytes as `int32` (4 bytes = 8 values) instead of
   one `uint8` per lane; build the 8 fp16 values and store them as one 16-byte vector. Today each
   lane loads 1 byte and `tl.interleave` makes the store pattern 2-wide.
2. **Bigger programs, fewer launches.** `BYTES_PER_PROGRAM = 1024` gives ≈1150 programs for the
   1536×1536 matrix and ≈6700 for the MLP ones; try 2048/4096 bytes per program and
   `num_warps` 4/8 (fixed values, no autotune: compile time counts).
3. **Scale once per 64 values, not per value.** The absmax of a block is the same for its 32
   bytes; load `absmax_8bit`, `code[q8]`, `absmax2` once per block (a `[BLOCKS, 32]` tile) instead
   of once per byte through a gather.
4. **Table in registers.** The 16-entry NF4 table can be a `tl.constexpr`-built select chain or a
   tiny tensor loaded once per program; compare with the current gather from global memory.
5. **Skip the second dequantization.** In backward bitsandbytes calls `dequantize_4bit` again for
   the same weight that forward just unpacked (and recomputation unpacks it a third time). Check
   whether caching **only the most recent layer's** fp16 weight across forward→backward of the
   same module is possible without raising the peak (it never is for all layers: 94 MB per layer).

Must stay bit-identical to bitsandbytes (`solution/test_nf4.py` already checks it; keep those
tests and add the new shapes). Deliverable: faster kernel, microbenchmark table (GB/s before and
after for each shape) in the PR, A/B on one machine, `full --repeats 2` with 10/10.

## Task 5 — compact storage of the MLP activations (BIGGEST MEMORY LEVER, ≈4–5 h, experimental)

Facts: the organizers' SwiGLU example keeps `gate` and `up` in fp16 for backward:
2 × N × 8960 × 2 bytes ≈ **72 MB per layer at N = 2048**, ≈1 GB at the peak (the audit's largest
group with attention). Forward does not depend on how these are stored; only the gradient that
flows back through SwiGLU does. The organizers allow storing activations in low precision in our
own code (QA_NOTES #32: "yes, the choice of precision is yours; the LoRA adapter does not change;
watch for fp16 overflow").

This needs our **own** SwiGLU autograd function and kernels (the example's kernels are not ours
and its code must not be copied: write it from scratch; the AST-similarity check reads the
example). Options, from safest to boldest:

1. **Recompute `up` instead of storing it?** No — that is a matmul; skip.
2. **Store `silu(gate)` and `up` packed differently**: the backward needs `gate` (for silu') and
   `up`. Check if storing `gate` in fp16 and `up` in **bf16-like 8-bit** loses anything measurable.
3. **8-bit with a per-row (per-token) scale**: `q = round(x / s * 127)`, `s = max|x|` of the row,
   int8 storage + one fp16 scale per row → half the bytes. Error ≤ s/254 per element.
4. **fp8 e4m3 / e5m2 by bit manipulation** (T4 has no fp8 compute, but storage is just bytes):
   e5m2 is the top byte of fp16 with rounding, so pack/unpack is one shift; e4m3 needs a scale.
5. **Store only one tensor**: keep `gate` and the product `h = silu(gate)·up` (needed anyway as
   the input of `down_proj`? check whether bitsandbytes keeps it — it does not, the weight is
   frozen), and recover `up = h / silu(gate)` where `silu(gate)` is not tiny; store an exception
   mask for the rest. Bold; only if 3/4 fail.

Acceptance (strict, because a FAIL costs everything): loss curve (check 6) first step 0.00%,
mean ≤ 0.05%; eval loss within ±0.05% of the base on **two seeds** (1 and 7) and on Dilya's
swapped-data run; peak lower by ≥ 300 MiB; time not worse by more than 1%. If any of those fails,
keep fp16 and write what you saw — a negative result with numbers is still useful for the README.

## Report (in each PR)

What changed, the table, A/B numbers, risks, and what you would do next. Finish by 17:00; the final
run is at 18:00 and only merged, measured changes go in.
