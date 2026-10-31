# Ansar — round 3 (31.10): submission notebook, where time still goes, reproducibility

Start: `git checkout main && git pull`, read `README.md`, `docs/results.md`, `docs/QA_NOTES.md`.
Branch per task, PR into `main` (no direct pushes). Rules as before: no `harness/`/`tests/`/`task/`,
no hidden rows, every number from a Kaggle T4 run, speed claims as A/B on one machine.

## Task 1 — the public submission notebook (first, ≈1.5 h)

Adilet gives you a GitHub token (classic, `repo`, 7 days). Kaggle → your notebook from
`templates/kaggle_notebook.ipynb`, secret `GITHUB_TOKEN`, GPU T4 x2, Internet on,
`REPO_URL = https://github.com/BAITC-Hacks/hack-3e361ee8-mindx.git`, `COMMIT` = current `main`.
**Save & Run All**; it must end with all 10 checks PASS. Make it **public**, open the link in an
incognito window, check the token never appears in the output. Send the link (with
`?scriptVersionId=`) to Adilet; at 18:00 you will rerun it with the final commit.

## Task 2 — profile of the current solution (≈3 h)

`kaggle_out/` from `python tools/kaggle_run.py quick` gives `runs_try/solution_profile/profile.json`
(16 profiled steps). Produce `docs/profile_solution.md`:

1. groups (matmul, attention, our kernels, casts/copies, elementwise, NF4, other): ms per step
   and %, next to the base profile (`docs/profile_base.md`);
2. top 30 kernels with calls per step;
3. **where the remaining casts and copies come from**: rerun one training step under
   `torch.profiler` with `record_shapes=True, with_stack=True` (a small script in `tools/`, model
   built and patched like the harness: `solution.apply()` before Soup) and list the 10 largest
   `aten::copy_` / `aten::to` / `aten::add` sources with Python stack, shape, dtype and ms. That
   list is what the kernel work picks next;
4. the time per step in forward vs backward vs optimizer.

## Task 3 — where the 19 s outside the training loop go (≈2 h)

Wall time of our run is ≈49 s, the loop ≈30 s. Make a timeline of one solution run: import end,
`apply()` start/end, model load end, dataset ready, trainer init, first step, last step, adapter
saved, process exit (wrap Soup's functions with timestamps in a probe script in `tools/`, without
changing `solution/`). Table in `docs/startup_time.md`. For each phase over 1 s say what it does
and whether the rules allow touching it (QA_NOTES #37–45: imports we do not use, hashing, etc.
are allowed; stubbing modules, `os._exit`, caches between runs are not).

## Task 4 — the head chunk A/B (≈1 h)

`CHUNK` in `solution/fused_ce.py`: 512 now; try 256 and 1024. A/B on one machine
(`run_task --only solution`, alternate), report peak and time. Keep the one that gives more points
by the formula in `docs/tasks/personB_round3.md`.

## Task 5 — reproducibility as the jury (≈1 h, after Adilet's README is final)

Follow `README.md` → "How to run" word for word as someone new to the repo, on Kaggle. Note every
place you had to guess, every error, every step that took longer than written. PR with fixes to
`README.md` wording only (not code).

Finish tasks 1–4 by 17:00. Final run at 18:00.
