# Dilya — round 3 (31.10): memory without losing time

Start: `git checkout main && git pull`, read `README.md`, `docs/results.md`, `docs/QA_NOTES.md`,
`docs/memory_audit_solution.md`. Each task in its own branch (`budget`, `norm16`, …) and its own PR
into `main`. Rules as before (AGENTS.md §4–5): no `harness/`/`tests/`/`task/`, no hidden rows,
every claim backed by a Kaggle T4 run; for a speed claim an A/B on one machine.

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

## Report (in each PR)

What changed, the table, A/B numbers, risks, and what you would do next. Finish by 17:00; the final
run is at 18:00 and only merged, measured changes go in.
