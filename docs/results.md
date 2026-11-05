# Журнал прогонов на Kaggle T4

| Время (Астана) | Коммит | Что проверяли | База, с | Наше, с | База пик, MiB | Наш пик, MiB | eval loss база / наш | speedup | memory | Проверки |
|---|---|---|---|---|---|---|---|---|---|---|
| 30.10 14:30 | 9c6d12e | RMSNorm + fused CE + NF4 + MLP fp16 (+ пример SwiGLU), 1 повтор | 81,4 | 74,5 | 14 345 | 6 343 | 2,04964 / 2,0497 (+0,003%) | 1,092× | 2,261× | **10/10 PASS** |
| 30.10 15:50 | ca5a93b | + RoPE (кернел Человека B), 1 повтор | 92,4* | 74,0 | 14 345 | 6 341 | 2,04963 / 2,04963 (0,000%) | 1,249×* | 2,262× | **10/10 PASS** |
| 30.10 16:20 | e6ee5c3 | RoPE: один кернел на fwd+bwd (одна компиляция), 1 повтор | 78,9 | 69,7 | 14 345 | 6 341 | 2,0497 / 2,04965 (−0,002%) | 1,132× | 2,262× | **10/10 PASS** |
| 30.10 16:39 | 5a2ae0a | Проверка Человека A, **2 повтора** (как у жюри) | 85,25 | 73,31 | 14 345 | 6 341 | 2,04965 / 2,04965 (0,000%) | **1,163×** | **2,262×** | **10/10 PASS** |
| 30.10 18:10 | ae766a8 | main после слияния: RoPE одним кернелом + аудит A, **2 повтора** | 83,22 | 67,71 | 14 345 | 6 341 | 2,04965 / 2,0496 (−0,002%) | **1,229×** | **2,262×** | **10/10 PASS** |
| 30.10 19:10 | 75282b8 | + компиляция кернелов в фоне (ветка `warmup`), 2 повтора | 82,98 | 69,19 | 14 345 | 6 341 | 2,04965 / 2,04968 (+0,001%) | 1,199× | 2,262× | **10/10 PASS** |
| 30.10 20:00 | 2f7ce33 | + эмбеддинг/lm_head в fp16 (ветка `embedding`), **2 повтора** | 92,1* | 75,03 | 14 345 | **5 461** | 2,04965 / 2,04965 (0,000%) | 1,228× | **2,627×** | **10/10 PASS** |
| 30.10 21:00 | 8a424bb | + fp16 attention (Dilya), прогон Dilya, 2 повтора | 85,5 | 70,1 | 14 345 | 5 441 | +0,001% | 1,219× | 2,636× | **10/10 PASS** |
| 30.10 21:40 | (attention+unpad) | + без паддинга, прогон Dilya, 2 повтора | 80,9 | 57,0 | 14 345 | 5 487 | 0,000% | **1,419×** | 2,614× | **10/10 PASS** |
| 30.10 22:30 | f58504f | + адаптивный пересчёт слоёв (бюджет 1024 токена) + эмбеддинг fp16 без fp32-копии, **2 повтора** | 88,27 | 58,64 | 14 345 | **4 569** | 2,04962 / 2,04967 (+0,002%) | **1,505×** | **3,140×** | **10/10 PASS** |
| 30.10 23:00 | 13c0f36 | + старт: `packages_distributions()` один раз, без хэширования `map` датасетов в памяти, **2 повтора** | 83,81 | 53,20 | 14 345 | 4 569 | 2,04968 / 2,04967 (0,000%) | **1,575×** | **3,140×** | **10/10 PASS** |
| 31.10 00:00 | 4d866f4 | + не грузим неиспользуемые sklearn/wandb, **2 повтора** | 82,90 | 53,41 | 14 345 | 4 569 | 2,04964 / 2,04967 (+0,001%) | 1,552× | 3,140× | **10/10 PASS** |
| 31.10 00:20 | b17f70c | + attention по строкам на повторённых k/v: fp16 fused-кернел вместо fp32 math (Dilya), **2 повтора** | 79,9 | 49,9 | 14 345 | **3 561** | 2,04966 / 2,04971 (+0,002%) | **1,602×** | **4,028×** | **10/10 PASS** |
| 31.10 00:50 | ac8eca8 | main: всё вместе (GQA-правка + старт без sklearn/wandb), **2 повтора** | 86,17 | 49,10 | 14 345 | **3 561** | 2,04964 / 2,04965 (0,000%) | **1,755×** | **4,028×** | **10/10 PASS** (доля кернелов 11,4%, 68 тестов) |
| 31.10 01:00 | b17f70c | **seed 1234** (Dilya), 2 повтора | 84,6 | 51,3 | 14 345 | 3 639 | −0,001% | 1,651× | 3,94× | **10/10 PASS** (доля кернелов 10,5%) |
| 31.10 01:10 | 385c14a | **seed 7**, 2 повтора | 85,88 | 48,78 | 14 351 | 3 539 | 2,05114 / 2,05118 (+0,002%) | 1,761× | 4,055× | **10/10 PASS** |
| 31.10 02:30 | startup2 | + `TORCHDYNAMO_DISABLE` только на импорт bitsandbytes (apply 4,9 → 3,7 с), **2 повтора** | 87,92 | 49,34 | 14 345 | 3 561 | +0,002% | 1,782× | 4,028× | **10/10 PASS** |
| 31.10 02:40 | startup2 + BUDGET_TOKENS=0 | **пересчёт всех слоёв**, 2 повтора | 83,52 | 55,31 | 14 345 | **1 739** | 2,04968 / 2,04971 (+0,001%) | 1,510× | **8,249×** | **10/10 PASS** |
| 31.10 03:40 | startup2 (b0 + прямой NF4) | пересчёт всех слоёв + прямой путь Linear4bit, **2 повтора** | 87,73 | 59,60 | 14 345 | **1 739** | 2,04961 / 2,04967 (+0,003%) | 1,472× | **8,249×** | **10/10 PASS** (кернелы 11,8%, 90 тестов) |
| 31.10 04:40 | stream (bd1…) | + веса слоёв в pinned-памяти хоста, слой на GPU только пока работает, **2 повтора** | 83,23 | 55,78 | 14 345 | **1 413** | 2,04966 / 2,04967 (0,000%) | 1,492× | **10,152×** | **10/10 PASS** (95 тестов) |
| 31.10 05:30 | stream | + веса уходят на хост прямо при загрузке, без резерва памяти загрузчика, перевод bf16→fp16 на GPU, голова кусками по 256, **2 повтора** | 83,47 | 58,39 | 14 345 | **1 209** | 2,04966 / 2,04966 (0,000%) | 1,430× | **11,865×** | **10/10 PASS** |
| 31.10 06:30 | stream | + LoRA dropout на упакованных токенах одним Triton-кернелом в каждую сторону, **2 повтора** | 77,95 | 55,06 | 14 345 | 1 209 | 2,04967 / 2,04965 (−0,001%) | 1,416× | 11,865× | **10/10 PASS** (кернелы **15,6%**, 98 тестов) |
| 31.10 08:30 | stream | + фоновая компиляция стартует сразу после импорта bnb, без лишнего RoPE (первый шаг 2,0 → 0,85 с), **2 повтора** | 78,97 | 54,80 | 14 345 | 1 209 | 2,04962 / 2,04963 (0,000%) | 1,441× | 11,865× | **10/10 PASS** |

A/B правки GQA на одной машине (Dilya, 13c0f36 против 13c0f36 + b17f70c по очереди): время запуска 53,2 → 46,9 с (−6,3 с, −12%), шаги обучения −15%, пик 4 565 → 3 561 MiB (−1 000), eval loss тот же.
| 30.10 22:40 | d196dbc | + fp16 attention (ветка `attention`): один каст входа, кернел LoRA-сумма + RoPE + fp16, 1 повтор | 82,03 | 70,79 | 14 345 | **5 441** | 2,04969 / 2,04961 (−0,004%) | 1,159× | **2,636×** | **10/10 PASS** |
| 30.10 23:50 | 8a424bb | fp16 attention (ветка `attention`), **2 повтора** (как у жюри) | 85,5 | 70,1 | 14 345 | **5 441** | 2,04965 / 2,04967 (+0,001%) | **1,219×** | **2,636×** | **10/10 PASS** |
| 31.10 12:30 | 596acb0 | NF4: широкий кернел распаковки (ветка `nf4fast`, Dilya), **2 повтора** | 85,3 | 59,8 | 14 345 | 1 739 | 2,04967 / 2,04963 (−0,002%) | 1,426× | 8,249× | **10/10 PASS** (доля кернелов 10,2%) |

\* База в этой сессии была медленнее обычного (92,4 с против 81,4 с), поэтому ускорение завышено шумом. Честная оценка — по шагам: медиана шага 0,774 → **0,702 с** (−9% к прошлому прогону), train_loop 52,7 → 49,8 с, доля наших кернелов 6,9% → **8,2%**. Бенчмарк RoPE: fwd+bwd fp32 в 3,85× быстрее PyTorch, но компиляция двух кернелов ~5,6 с — следующая задача: фоновая компиляция в `apply()`.

Прогон 30.10 16:20 (ветка `rope`): база в норме (78,9 с), решение 69,7 с против 74,0 с с двумя кернелами RoPE; доля наших кернелов 8,2%; 36 тестов solution и 79 тестов Soup проходят.

Прогон 30.10 18:10 (`ae766a8`, 2 повтора): доля наших кернелов 8,5%, `apply` 7,4 с, медиана шага 0,646/0,668 с (база 0,874/0,945). Первый шаг 4,6 с против 1,1–1,5 с у базы: ~3,3 с — компиляция кернелов, цель фоновой компиляции.

Фоновая компиляция: первый шаг 4,6 → 0,84 с. Общий выигрыш в этих двух прогонах не виден: машины Kaggle были медленнее (у базы медиана шага 0,91–1,05 с против 0,87–0,94), время вне цикла +1,3 с (компиляция в потоке конкурирует с загрузкой модели за CPU). Эмбеддинг в fp16: пик 6 341 → 5 461 MiB (−880: и сами веса, и fp16-копия головы на каждом шаге), torch allocated 6 069 → 5 179 MiB. \* база 92,1 с — медленная машина.

Прогон 30.10 22:40 (ветка `attention`, `d196dbc`): в этой сессии медиана шага база 0,929 с / решение 0,734 с, train_loop 58,8 → 48,0 с, доля наших кернелов 8,0%, `apply` 7,4 с; 55 тестов solution и 79 тестов Soup проходят. A/B на одной машине (`--only solution`, патч вкл/выкл по очереди ABAB): шаги 3–64 35,70 → 33,19 с (**−7,0%**), медиана шага 0,584 → 0,544 с, пик 5 461 → 5 441 MiB, eval loss без изменений; общее время 60,7 → 60,0 с — вне цикла +0,4 с на компиляцию нового кернела в фоне (первый прогон A/B ещё и скачивал модель).

Прогон 30.10 23:50 (`8a424bb`, 2 повтора): база 81,8 / 89,1 с, решение 69,1 / 71,2 с; медиана шага база 0,919 / 1,070 с, решение 0,714 / 0,762 с; доля наших кернелов 7,9% (минимум из 2); кривая loss 0,00% / 0,01%.

## Детали прогона 30.10 14:30
- Шаги: train_loop 58,4 → 52,7 с; медиана шага 0,925 → 0,774 с.
- PyTorch пик allocated: 13 373 → 6 069 MiB.
- Доля наших кернелов: **6,9%** (NF4 2176 мс, RMSNorm 690 мс, fused CE 373 мс за шаги 3–64).
- Кривая loss: первый шаг 0,01%, среднее 0,01%. Метрики и rewind совпадают.
- `apply()` занимает 7,4 с — разобрать (импорты, которые иначе были бы позже, или лишнее).
- Версии: Python 3.13.15, torch 2.11.0+cu128, Triton 3.6.0.

## Person A verification: 2026-10-30 16:39 UTC+05:00

Code: `5a2ae0a` (solution unchanged); Kaggle T4; seed 1; `--repeats 2` (base, solution, solution, base).
Run: https://www.kaggle.com/code/ansargabit/mindx-audit-full-5a2ae0a (version 2).
Artifacts: `kaggle_out/full/out/runs/result.json`, `checks.json`, and `kaggle_out/full/out/pytest_solution.log`.

**36 kernel tests PASS; 79 Soup regression tests PASS; all 10 official checks PASS.**

| Run | Wall s | GPU peak MiB | Train loop s | Median step s | apply s | Torch allocated MiB | Torch reserved MiB |
|---|---:|---:|---:|---:|---:|---:|---:|
| base_1 | 85.02 | 14345 | 57.24 | 0.9002 | None | 13373.2 | 14202.0 |
| base_2 | 85.49 | 14345 | 59.08 | 0.9327 | None | 13373.2 | 14202.0 |
| solution_1 | 73.53 | 6341 | 47.16 | 0.6551 | 9.03 | 6069.2 | 6198.0 |
| solution_2 | 73.08 | 6341 | 46.56 | 0.6482 | 8.93 | 6069.2 | 6198.0 |

- Official summary: wall 85.25 -> 73.31 s (**1.1629x** speedup; approximately 14% less wall time).
- GPU peak 14,345 -> 6,341 MiB (**2.2623x** memory ratio; approximately 55.8% less memory).
- Eval loss: **2.04965 / 2.04965**, reported delta 0.000%.
- Own kernels: **8.7% minimum** across solution runs, above the 5% gate.
- Aggregation follows the actual harness: median wall time, maximum nvidia-smi peak across repeats. With two runs, the time median equals their mean.
- Environment: Python 3.13.15, torch 2.11.0+cu128, Triton 3.6.0; task library pins unchanged.
- The initial pre-verification launch (version 1) received CPU only and failed before tests; it is not a benchmark result.
- Extra seed 7 verification is in progress.

### Audit deliverables

- `docs/memory_audit.md`: requested 512-token diagnostic OOM plus successful 384-token diagnostic.
- `docs/profile_base.md`: 16 active baseline profile steps, grouped GPU times and top 30 kernels.
- `docs/startup_time.md`: diagnostic startup stages and official wall-minus-loop observations.
- The existing captain-owned Kaggle launcher assumes a Unix executable path; on Windows it was imported and its `KAGGLE` variable set to the installed `Scripts/kaggle.exe`, without editing it.
- The launcher does not propagate command failures; success above was verified from test logs and all ten entries in `checks.json`, not just Kaggle job status.


### Round 2 Person A diagnostics - 2026-10-30

- Solution source `5782a18`, audit branch `audit2`; no changes to solution, harness, task or tests.
- Memory run: https://www.kaggle.com/code/ansargabit/ansargabit-audit2 (v1), three successful synthetic forward/backward processes on T4 with expandable_segments:True. Allocated peaks: baseline T384 11912.085 MiB, solution T384 3869.568 MiB, solution T512 4217.173 MiB. These are not official training peaks or rule-check results. Full tables: docs/memory_audit_solution.md.
- Startup run: https://www.kaggle.com/code/ansargabit/ansargabit-startup2 (v2), all worker processes passed. Apply 17.516 s including lazy imports; three-kernel cold launch medians 2.776 s serial / 2.905 s with three threads. Details and limitations: docs/startup_time.md.
- Submission notebook: prepared locally from templates/kaggle_notebook.ipynb with team REPO_URL and empty COMMIT. Public Save & Run All is PENDING: Computer Use native pipe unavailable and a repository-scoped Kaggle GITHUB_TOKEN secret is not yet configured. No public notebook link or official checks claimed. User needs to import the prepared notebook and resolve repository secret access with the captain if fine-grained organization access is unavailable.

| 31.10 | stream 914d188 (budget 512, backward prefetch hints, one-kernel dropout) | mindx-b512-full | 50.88 / 50.67 s | 2067 MiB | 1.613× / 6.94× | -0.001% | 10/10, kernels 13.4% |

**31.10 int8 backward dX bench (T4, tools/bench_int8.py, mindx-int8-bench).** _int_mm vs fp16 at M=1168: q/o x1.35, k/v x1.12, gate/up x1.27, down x1.55; at M=2048 down x0.95. With ideal fused quantize/rescale passes (memory bound): net x0.70–1.26, about 0.7 ms per layer per step, ≈ 1 s per run at best. Not taken: too little for the accuracy change and the new code.

**31.10 head-offload (two walks, before the copy-stream fix), mindx-head-full:** 57.26/57.46 s, **1187 MiB (12.09x)**, but checks 5-7 FAIL: first step 0.01%, mean loss +5.3/6.0%, entropy 8.4%, eval +2.2%. Cause: a new walk over the vocabulary started copying into buffers the compute stream was still reading (fixed: the first copies wait for the compute stream). Not taken as is.

**31.10 head-tok (one walk + tok + stack + fp16 inputs), mindx-headtok-full:** 37.8/37.9 s, 949 MiB, FAIL 5-7 (first step 0.01%, mean +4.6%, eval +3.9%): the fp16-input recompute had no tensor input requiring grad (the first layer input comes from the frozen embedding), so autograd skipped its backward. Fixed with an anchor input.

**31.10 head-tok a0dfdee, mindx-headtok-full2: 10/10.** 58.32/58.56 s vs base 91.2/87.4 (1.528x), **1077 MiB (13.32x)**, eval +0.001%, own kernels 12.0%. PR #11.

| 31.10 | main 1d3b11e (PR #11 head-tok), **seed 7** | ansar-main-seed7 | 59.73 / 60.11 s (base 95.76 / 88.21) | 1037 MiB | 1.535× / 13.84× | +0.000% | 10/10, kernels 11.7%, pytest 129 passed |

**31.10 final-mem 196bd9a (no own tokenization + gate/up + host recompute inputs + head slice 4096), mindx-finalmem: 10/10, 133 tests.** 66.42/64.81 s vs slow base 92.3 (1.406x), **757 MiB (18.95x)**, eval +0.001%, own kernels 11.2%.

**31.10 final-fast2 7df91c3 (Dilya, aqniet/dilya-ff2): 10/10.** 61.82/62.11 s vs base 88.74/83.88 (1.393x), **673 MiB (21.3x)**, eval -0.001%, own kernels 12.8%; 133 tests passed + 1 test with a missing import (fixed in b3dee0d, test only).

**31.10 final-600 eb34afd (MLP in pieces in the rerun backward), mindx-finalfast2: 10/10, 136 tests.** 64.65/64.72 s vs base 80.0/82.2 (1.254x), 675 MiB (21.25x): no memory gain (the peak is not in the MLP), slower. Not taken.

**31.10 22:00 final-500 73402c2 (vocabulary table loaded straight into pinned host memory), mindx-final500b: 10/10, 135 tests.** 60.30/60.50 s vs base 79.75/82.35 (1.342x), **505 MiB (28.41x)**, eval 0.000%, own kernels 13.1%, torch max allocated 343 MiB. **Final.**

**31.10 final main d0ef96f, seed 7 (mindx-final-seed7): 10/10, 135 tests.** 65.60/64.79 s vs base 84.95/84.65 (1.301x), **507 MiB (28.31x)**, eval +0.003%, own kernels 13.8%.
**31.10 final-500 73402c2 on another machine (Dilya, aqniet/dilya-f500): 10/10, 135 tests.** 64.05/63.70 s vs base 92.49/85.56 (1.394x), **505 MiB (28.41x)**, eval -0.001%, own kernels 12.9%.

**31.10 final-500 73402c2, seed 7 (Dilya, aqniet/dilya-f500-s7): 10/10, 135 tests.** 62.62/64.58 s vs base 79.77/85.69 (1.301x), **507 MiB (28.31x)**, eval +0.001% (2.05115 / 2.05117), own kernels 12.1%.

**31.10 23:00 public Kaggle notebook adiletzholdasbek/notebook659c3b6e49, COMMIT d0ef96f: ALL CHECKS PASSED.** base 86.13 s / 14 345 MiB, solution 58.37 s / 505 MiB, **1.476x**, memory 28.406x, eval −0.003%, own kernels 14.8%.

**31.10 SwiGLU in-place A/B (branch swiglu-inplace c8b5a1f vs final-500, aqniet/dilya-swiglu-ab, one T4, alternating):** base 67.59/60.42 s, 505 MiB (torch 343.3); new 60.08/60.62 s, **485 MiB** (torch 329.1); eval 2.04964-2.04965 both; 142 tests passed. −20 MiB (−4%), time within noise. Not taken (below the 5% threshold, after the code freeze).

**4.11 floor-all, first runs (one run each, not --repeats 2; rules unless noted).** rules_time (NF4 weights cached in fp16): 42.35 s vs base 77.79 (1.837x), 6793 MiB, 10/10, own kernels 8.6%; with the packed bytes moved to host 44.16 s vs base 81.04 (1.835x), 6125 MiB, 10/10. Grouped layer rerun (rows of a rerun layer run in groups, with in-place SwiGLU and the block scales streamed): group 512: 91.06 s vs base 80.79, 429 MiB, check 8 FAIL. group 1024: 64.64 s vs 80.54, 449 MiB, 10/10; plus head buffers, dropout ones and the head gradient released through backward: **62.02 s vs 78.08 (1.259x), 409 MiB (35.07x), eval -0.001%, 10/10**. group 768 with the same release: 80.01 s vs 81.37, 389 MiB, 10/10.

**4.11 T4 benches (tools/floor_bench.py, mindx-fl-gemm).** Backward GEMM with W column-major: slower or equal to row-major. Padding N to a multiple, splitting a GEMM into pieces: no consistent gain. Context 107 MiB, first GEMM +12 MiB, SDPA +8, Triton +6 (133 MiB). Dropout draws from fp16 or expanded ones differ from fp32 ones, so a fresh fp32 buffer is the cheapest legal draw.

**4.11 whole-step microbatch (groups of 512 tokens, no layer rerun; idea from docs/PERF_ANALYSIS_INDEPENDENT.md), mindx-fl-ml1 / mf1.** Weights and table on the host: 72.09 s vs base 77.32, 1141 MiB, checks 10/10. Weights and table on the GPU: 58.38 s vs base 79.58, 2675 MiB, checks 10/10. Own test of the LoRA gradients against the whole batch failed at rtol 1e-4 (cause not established). Worse than the grouped rerun at 62 s / 409 MiB; code removed.

**4.11 int8 forward (same document): not built.** The 31.10 probe already shows the first-step loss over 1% on 10-55 of 64 batches, which breaks check 6.

**4.11 PERF_IDEAS idea 1 (W^T stored row-major for gate/up), tools/bench_layout.py, mindx-fl-layout, one session, two rounds:** forward -11...+13%, backward -12...+15% depending on N, no stable gain. Not built.

**4.11 PERF_IDEAS idea 2 (a rerun MLP skips down_proj and SwiGLU forward; group 1024), mindx-fl-tail1:** 60.06 s vs base 79.52 (1.324x), 389 MiB (36.88x), eval +0.001%, 10/10 (one run; 2 failures in pytest were a bug of the new test itself, fixed).

**4.11 rules_balanced (group 1024 + dead-value MLP rerun), jury order --repeats 2, mindx-fl-bal-r2: 57.75 / 59.61 s vs base 75.94 / 78.38 (1.315x), 389 MiB (36.88x), eval +0.001%, 10/10.**

**4.11 rules_balanced, seed 7, --repeats 2, mindx-fl-bal-s7: 61.74 / 61.67 s vs base 78.31 / 81.34 (1.294x), 407 MiB (35.26x), eval +0.002%, 10/10, 149 tests.**

**4.11 rules_memory (group 512), seed 7, --repeats 2, mindx-fl-mem-s7: 80.33 / 82.04 s vs base 77.11 / 79.32 (0.963x), 349 MiB (41.12x), eval -0.001%, 10/10.**

**4.11 rules_mid (no layer rerun, weights and table on the host), mindx-fl-mid1, one run: 45.42 s vs base 76.5 (1.684x), 3085 MiB (4.65x), eval 0.0%, 10/10.**

**4.11 same-machine A/B, group 1024, order A B C C B A, mindx-fl-ab1 (solution only, one T4).** A plain: 62.27 / 61.52 s, 387 MiB. B AdamW moments on the host: 60.41 / 61.83 s, 381 MiB. C cuBLAS workspace :4096:1: 66.09 / 65.94 s, 379 MiB (median step x1.165). AdamW on the host taken (-6 MiB, no measurable time cost); the smaller workspace not taken (-8 MiB, +4.2 s).


### A/B на одной машине: прямой проход, новые кернелы, причинность (floor-all, вне сданного коммита)

- ab2, group 1024, все 381 MiB, 154 теста: без правок 57.21 / 60.36 с; прямой первый проход 59.56 / 60.17 с; плюс `layer_input` и `head_acc` 59.71 / 59.72 с. Выигрыша нет, разница в пределах шума.
- ab3, задержка 100 мкс на CPU перед каждой распаковкой NF4 (630 запусков на шаг, ожидание +4.03 с): `rules_balanced` 60.21 / 59.12 / 58.11 с, цикл 40.81 / 39.98 / 39.26 с; с задержкой 63.93 / 62.27 / 61.96 с, цикл 44.63 / 43.25 / 43.14 с. Цикл +3.65 с, общее время +3.57 с: около 90% задержки проходит насквозь, GPU в этих местах ждёт CPU.
- Один кернел на все 7 весов слоя (`nf4_layer_kernel`, `rules_fast`), A/B на одной машине, 8 прогонов, 172 теста: `rules_fast` 56.69 / 59.60 / 59.91 / 59.30 с, цикл 37.36 / 40.87 / 40.73 / 40.82 с, 441–443 MiB; `rules_balanced` 62.31 / 62.57 / 62.21 / 62.41 с, цикл 43.25 / 43.56 / 43.44 / 43.62 с, 381 MiB. Таймлайн: простой 191 → 111 мс на шаг, запусков распаковки 630 → 54. Полный прогон с 10 проверками ещё не делался.
- Прочие проверки на `rules_fast` (1024): группы 1536 (фактически без групп) 55.75 / 56.05 с, 501–503 MiB, против 57.55 с у базы: около −1.5 с за +60 MiB. Копия входа слоя на хост в отдельном потоке GPU: 56.3 / 57.15 с, 461–463 MiB, эффект в пределах шума. Срез головы 8192: 57.15 / 57.0 с, 461 MiB, эффекта нет.
- LoRA + dropout кернелами Triton (один запуск вперёд, два назад, r=16): 67.42 / 67.13 с против 58.75 / 60.10 с у `rules_fast`, то есть +7–8 с; после исправления чтения памяти осталось то же. Отклонено: матмулы cuBLAS на тензорных ядрах кернел Triton не догоняет.
- Третий слот для весов: 57.10 / 56.85 / 57.21 с, 481–483 MiB, против 57.10 / 57.04 с, 441–443 MiB (без первого, быстрого прогона машины 52.94 с): эффекта нет. Таймлайн `rules_fast`: 538.7 мс на шаг, кернелы заняты 79.8%, простой 65.8 мс, только копии 43 мс.
- Полный прогон `rules_fast`, `--repeats 2`, 172 теста, 10/10: сид 1 — 58.8 / 57.89 с против базы 81.55 / 85.81 с (1.434x), 443 MiB (32.38x), eval −0.001%, свои кернелы 12.3%; сид 7 — 57.3 / 57.6 с против 79.81 / 83.47 с (1.421x), 461 MiB (31.13x), eval +0.002%, свои кернелы 13.3%.
- k и v сразу на 12 голов в `qkv_rope_kernel` (минус ~220 запусков на шаг): 54.69 / 54.85 / 54.70 / 55.19 с против 52.89 / 54.85 / 55.34 с у `rules_fast` без первого прогона, 443 MiB: эффекта нет.
- **Редкая ошибка:** в прогоне `rules_fast` (ab12, 9-й на машине) фоновая компиляция `compile_head_stream` упала в MLIR (`LLVMDIScope`, `PassManager::run failed`), затем обучение встало на нулевом шаге и было убито по таймауту 900 с. В сохранённых логах это единственный случай. Стресс-тест: `tools/floor_compile_stress.py`.
- Стресс-тест компиляции (`tools/floor_compile_stress.py loop`, `apply()` + загрузка модели, пустой кэш, без Soup): 538 раундов, 0 сбоев. Ошибку не воспроизвёл; условия отличаются от реального запуска.
- Имитация сбоя фоновой компиляции кернела головы (обе попытки падают): основной поток компилирует сам, прогон завершён, 54.85 с, 441 MiB, без зависания. Путь после обычного исключения исправен.
- Настоящие прогоны `rules_fast` с повтором компиляции и дампом стеков: 61 из 61 завершён, зависаний нет. Метка «bad» в этом логе ложная, ошибка в условии инструмента (исправлено). Сбой компиляции без зависания этот лог исключить не может: выводились только последние 150 строк. Всего по сохранённым логам: 1 зависание примерно на 150 настоящих прогонов; причина неизвестна.
- Полный прогон `rules_fast` без групп строк (теперь профиль по умолчанию), `--repeats 2`, 172 теста, 10/10: сид 1 — 53.4 / 52.76 с против базы 78.79 / 79.78 с (1.494x), 501 MiB (28.63x), eval +0.000%, свои кернелы 14.5%; сид 7 — 53.49 / 52.55 с против 77.22 / 79.16 с (1.475x), 481 MiB (29.84x), eval +0.001%, свои кернелы 14.3%.
