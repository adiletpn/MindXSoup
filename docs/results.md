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
