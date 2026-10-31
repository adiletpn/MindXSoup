# Runtime bottlenecks on main 7384b69 — 2026-10-31

> **Measured before PR #11** (head/embedding streamed from host, fp16 recompute inputs, own tokenization). On main `1d3b11e` the numbers differ — current solution: 58.3/58.6 s, 1 077 MiB (captain) and 59.7/60.1 s, 1 037 MiB (seed 7, Ansar); see `docs/results.md`. Do not quote the figures below as the current solution.

Measured on aqniet's interactive Kaggle T4 session, notebook
[`notebook9d5744f7f4`](https://www.kaggle.com/code/aqniet/notebook9d5744f7f4/edit).
The measured solution source is `7384b69b78955bb5a84bd2f734e4741e0fd43941`.
PR #11 was still open at measurement time; its improvements are not part of
this baseline. Diagnostic tooling is on `codex/runtime-profile`.

## Unprofiled training and separate GPU profile

The unchanged harness ran `--only solution --profile`: one complete unprofiled
64-step training run, evaluation, then a separate short profiler run. Download
warmup took 37.98 s and is excluded from the training measurement.

| Quantity | Measured value |
|---|---:|
| Whole training process | 55.98 s |
| Training loop | 34.76 s (62.1% of whole-process time) |
| Outside training loop | 21.22 s (37.9%) |
| `apply()` (already included above) | 4.44 s |
| Median training step | 0.5012 s |
| First step | 1.626 s |
| Whole-process GPU peak | 1,585 MiB |
| Public eval loss | 2.04962 |

Both training and profiler exited 0. The profiled process took 133.60 s:
**that is profiler overhead, not training performance**. The profile contains
16 active steps. No base run or all-ten-check verdict is claimed here.

| GPU activity group | Sum per profiled step, ms | Share of summed GPU durations |
|---|---:|---:|
| Matrix multiplication | 186.08 | 43.49% |
| Pinned host-to-device copies | 99.07 | 23.16% |
| NF4 unpack | 37.72 | 8.82% |
| Attention | 21.19 | 4.95% |
| Generic elementwise | 20.87 | 4.88% |
| SwiGLU | 16.03 | 3.75% |
| Cast/copy kernels | 15.90 | 3.72% |
| Dropout | 15.27 | 3.57% |
| RMSNorm | 7.19 | 1.68% |
| Head loss reductions (excludes head GEMMs) | 3.06 | 0.71% |
| RoPE/QKV fusion | 2.70 | 0.63% |

Full category table and top 20 kernel names:
[`runtime_profile_7384b69.json`](runtime_profile_7384b69.json).
Categories are heuristics over kernel names, not module-level attribution.
In particular, the head's GEMMs are counted in matrix multiplication, so the
0.71% reduction category is **not** the entire language-model head cost.

GPU durations sum to 427.87 ms per profiled step. Copies run on a separate
stream and can overlap computation. Therefore neither these percentages nor
the sum can be read as fractions of wall time, and subtracting this sum from
the unprofiled median step does not measure idle time.

## Which changes are worth investigating

1. **Reduce recomputation to reduce matrix multiplication work.** This is a
   speed/memory tradeoff within the solution's own checkpointing logic, with
   no changes to Soup's training configuration. A same-machine 256/512-token
   budget comparison is recorded below. This does not replace final checks
   or establish the best leaderboard score.
2. **Reduce startup work.** Outside-loop time is substantial, but includes
   imports, model loading, data preparation and saving. It is not all
   removable. The captain reported ~1.8 s from tokenization in PR #11; that
   change is already under review. Ansar announced `ansar-preload` for model
   loading overlap; this audit does not duplicate that implementation.
3. **Preserve copy/compute overlap.** The largest individual activity is
   HtoD, but main already coalesces transfers and prefetches layers. Its
   summed duration is not proof of a 99 ms stall or an available 23% speedup.
4. **NF4 is secondary to GEMMs.** Reuse of unpacked weights within a layer
   could save work at a memory cost; no gain from that idea is established
   here. Another dequant kernel should not be prioritized based on event
   count alone.

The earlier LoRA audit found only 8/12 weight-shaped casts per layer-step and
~0.15–0.20% device time, which is too small to justify weight caching under
its explicit stop gate. Many `copy_` calls do not mean LoRA casts dominate.

## Measured speed/memory tradeoff: recompute budget 256 → 512

Same T4, same main solution, same task and seed. The only candidate change
was `BUDGET_TOKENS = 512` in a disposable copy of `solution/recompute.py`.
Order: initial A, GPU/CPU diagnostics, B, B, final A; evaluation occurred
between groups. Thus this is not an uninterrupted ABBA experiment, and the
baseline drift (55.98 → 58.30 s) is visible rather than hidden.

| Variant | Whole-process seconds | Median, s | Loop seconds | Peak, MiB | Public eval loss |
|---|---|---:|---|---:|---|
| A: budget 256 | 55.98 / 58.30 | 57.14 | 34.76 / 37.26 | 1,585 | 2.04962 / 2.04962 |
| B: budget 512 | 52.51 / 53.27 | 52.89 | 31.69 / 32.50 | 2,049 | 2.04967 (first B adapter) |

Observed whole-process reduction: **4.25 s, 7.44%, or 1.0804× speedup**.
Both B runs are faster than both A runs. Median step times: A 0.5012 / 0.5199 s,
B 0.4435 / 0.4542 s. This supports reduced training work, not merely startup
noise, as the source of the gain.

Cost: **+464 MiB (+29.27%)**. At 1024 tokens, the solution's recompute formula
selects 21 of 28 layers with budget 256 versus 14 with budget 512; retaining
more activations avoids seven repeated layer forwards for that token count.
Actual batches vary, so that example is not the number for every measured step.

The public eval difference is approximately +0.00244%, but this comparison
does not certify all ten checks: no fresh base run was made, and the harness
evaluates only the first B adapter. Do not promote the budget change without
the full paired base/solution validation and a score tradeoff decision.
The default solution remains unchanged. PR #11/head-tok was not benchmarked
with this budget here, so its performance cannot be inferred from this result.

## Startup instrumentation

An initial `cProfile` attempt produced inconsistent rows (self time greater
than cumulative time). Its per-function numbers were rejected. The final
tool uses explicit `perf_counter` spans and preserves the harness's import
order. Nested spans still overlap and must not be added together.

Boundaries measured by `tools/profile_training_startup.py` (one run on main 7384b69, exit 0):

| Span (nested, overlapping — do not add) | Seconds |
|---|---:|
| Process start → first training step | 18.577 |
| `import torch` | 1.656 |
| `import transformers` (includes the metadata scan below) | 3.210 |
| ↳ `importlib.metadata.packages_distributions()` scan | 2.419 |
| Model load (`load_model`) | 2.549 |
| `Dataset.map` (3 calls, total) | 3.253 |
| Tokenizer load | 0.495 |
| `save_model` | 0.392 |

Limits: the tool touches the Trainer early, so part of the imports move ahead of `apply()`; its
`apply()` of 0.053 s is **not** comparable to the official 4.44 s. `Dataset.map` (tokenization and
its hashing) is what PR #11's own tokenization replaces (~1.8 s reported by the captain); model
loading overlap is Ansar's `solution/preload.py` work. Neither is re-measured here.

## Reproduce

```bash
python task/prepare_data.py
python harness/run_task.py --only solution --profile --out runtime_runs
python tools/runtime_profile_summary.py runtime_runs --out runtime_summary.json
python tools/compare_recompute_budget.py --source . --initial-runs runtime_runs --out budget_comparison
```

The budget tool uses disposable checkouts, writes only the budget constant in
their `solution/recompute.py`, and runs the unchanged official harness with
fresh per-run compiler caches. Source `solution/`, `harness/`, `tests/` and
`task/` are not modified by this audit.
