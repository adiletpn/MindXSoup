# Streaming experiment — 2026-10-31

Target: at most 40 seconds, lower whole-process GPU peak, and preserved training quality.
No claim of meeting that target is made without completed T4 measurements.

The starting checkout contained a merge conflict in `solution/fused_ce.py` (head chunks
1024 versus 256). The working file uses 256, matching the documented low-memory run.
The pre-existing recomputation budget is 256, not the budget 0 behind the README's
58.4-second / 1209-MiB result. That historical result is not this experiment's baseline.

Candidate: consolidate each layer's seven pinned weight buffers into one contiguous buffer,
then issue one H2D copy instead of seven. Use two GPU slots instead of three. Quantized
bytes, compute dtypes, RNG, data, optimizer and loss calculation are unchanged. Fewer slots
may reduce prefetch overlap, so speed must be measured. Consolidation adds a host copy at
setup; its cost is included in the official wall time. CPU pinned-memory use is not reduced.

Controls in `solution/stream.py`:

| Variant | SLOTS | COALESCE_TRANSFERS |
|---|---:|---|
| Baseline | 3 | False |
| Candidate | 2 | True |

The expanded `solution/test_stream.py` tests both transfer paths with one through four
slots, with and without prefetch hints. It compares outputs and gradients bit for bit
over two steps and checks that consolidated module buffers share pinned storage.

Private Kaggle jobs (initial full runs, two repeats, default seed):

- https://www.kaggle.com/code/ansargabit/mindx-speed-memory-baseline
- https://www.kaggle.com/code/ansargabit/mindx-speed-memory-coalesced

For a same-machine ABBA comparison on seed 7, use a disposable checkout:

```bash
python tools/kaggle_run.py cmd --cmd "python solution/benchmark_streaming.py" --slug mindx-speed-memory-abba
```

This runs solution tests and four separate official solution measurements with empty
compiler caches. It restores the streaming source after the experiment, records source
hashes and raw results, and rejects training/evaluation failures. It does not replace
the full baseline/solution run and all ten official rule checks.

## Initial results (separate T4 sessions, seed 1, two repeats)

| Variant | Wall seconds | Peak MiB | Solution eval loss | Tests | Official checks |
|---|---:|---:|---:|---:|---:|
| Repaired starting checkout | 56.105 | 1603 | 2.04966 | 107 passed | 10/10 |
| Consolidated transfers, two slots | 56.600 | 1585 | 2.04966 | 107 passed | 10/10 |

Memory improved by 18 MiB in these runs. There is no demonstrated speed or loss
improvement. Different machines prevent attributing the 0.495-second difference to code.
Artifacts: `kaggle_out/speed_memory_baseline/out/` and
`kaggle_out/speed_memory_coalesced/out/` (local, ignored by Git).

The expanded tests and same-machine ABBA run are separate from these initial 107-test
runs. Local syntax parsing and `git diff --check` passed. Local Python has no
PyTorch/Triton test environment.

## Additional experiment

`solution/benchmark_recompute.py` temporarily tries a zero recomputation budget with
reentrant checkpointing when the input requires gradients. It runs solution tests,
the two-repeat official harness, and all rule checks, and restores the source afterward.
This targets autograd bookkeeping overhead and retained activations, at the cost of
recomputing more layers. It is not enabled in the default solution without validation.
Run: https://www.kaggle.com/code/ansargabit/mindx-speed-memory-recompute
