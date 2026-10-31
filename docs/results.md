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

A/B правки GQA на одной машине (Dilya, 13c0f36 против 13c0f36 + b17f70c по очереди): время запуска 53,2 → 46,9 с (−6,3 с, −12%), шаги обучения −15%, пик 4 565 → 3 561 MiB (−1 000), eval loss тот же.
| 30.10 22:40 | d196dbc | + fp16 attention (ветка `attention`): один каст входа, кернел LoRA-сумма + RoPE + fp16, 1 повтор | 82,03 | 70,79 | 14 345 | **5 441** | 2,04969 / 2,04961 (−0,004%) | 1,159× | **2,636×** | **10/10 PASS** |
| 30.10 23:50 | 8a424bb | fp16 attention (ветка `attention`), **2 повтора** (как у жюри) | 85,5 | 70,1 | 14 345 | **5 441** | 2,04965 / 2,04967 (+0,001%) | **1,219×** | **2,636×** | **10/10 PASS** |

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
