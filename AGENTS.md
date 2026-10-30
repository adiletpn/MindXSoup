# AGENTS.md — общий контекст проекта (читать ЛЮБОЙ нейросети перед работой)

Команда **MindX**, хакатон Decentrathon 6.0, кейс **Soup Kernel Challenge**.
Дедлайн: **31 октября 23:59 по Астане** (внутренний дедлайн команды — 22:00).
Репозиторий: https://github.com/BAITC-Hacks/hack-3e361ee8-mindx (вся работа только здесь).

## 1. Задача в двух абзацах

`soup train` (soup-cli 0.77.0) дообучает **Qwen2.5-1.5B-Instruct** (веса 4-bit NF4 через bitsandbytes,
LoRA r=16 на q_proj и v_proj, batch 4, до 512 токенов, 64 шага) на Kaggle **T4** (fp16, без bf16).
База: ~80 с на весь запуск, пик ~14 350 MiB видеопамяти.

Мы пишем **свои Triton-кернелы** в папке `solution/` и подключаем их функцией `solution.apply()`
(монкипатчи transformers / peft / bitsandbytes / Soup), чтобы тот же запуск шёл быстрее и занимал
меньше памяти, а результат (адаптер, метрики, rewind.jsonl) остался тем же в пределах допусков.

## 2. Баллы

- 20 — все 10 проверок `harness/check_rules.py` проходят. **Хоть одна FAIL = 0 баллов за всё.**
- 35 — скорость `s = время базы / наше`, 30 — память `m = пик базы / наш`. Обе шкалы логарифмические
  относительно лучшей команды; выигрыш < 5% = 0.
- 15 — README (что сделано + источники идей, как запустить, таблица результатов + честные ограничения).

## 3. Что известно о базовом запуске (из исходников, важно)

- `prepare_model_for_kbit_training` держит **эмбеддинги, нормализации и lm_head в fp32**, поэтому
  residual-поток (hidden_states) — **fp32**. Слои bnb `Linear4bit` считают матмул в fp16 и
  **приводят выход обратно к fp32** → много кастов и fp32-активаций.
- В MLP для backward хранятся `gate` и `up` в fp32 (~73 MB каждый на слой × 28 слоёв). Они получены
  из fp16-матмула, поэтому хранить их в fp16 — без потерь.
- Логиты `[4, T, 151936]` в fp32 (~1.25 GB при T=512) + log_softmax + градиент = 3–4 GB на пике.
  TRL дополнительно считает по логитам `entropy` и `mean_token_accuracy`, Soup — loss каждой строки
  для `rewind.jsonl`.
- В Soup уже есть «шаг без логитов»: `soup_cli/utils/stream_loss_chunk.py` (`ChunkedStep`,
  `STEP_ATTR = "_soup_chunked_step"`), его читает rewind (`soup_cli/trainer/rewind_hf.py`).
- Профиль шага: матмулы 36%, поэлементные операции и касты 44%, softmax по словарю 7%,
  attention 5%, распаковка NF4 5%.
- Нет gradient checkpointing, attention = SDPA, оптимизатор adamw_torch, LoRA dropout 0.05.

## 4. Жёсткие правила (нарушение = дисквалификация или FAIL)

- Нельзя: unsloth, liger-kernel, xformers, flash-attn, torchao и всё из `BANNED_MODULES`
  в `harness/common.py`; кернелы из `torch.compile` за свои не считаются.
- Нельзя менять задачу: модель, данные, батчи и их порядок, число шагов, LoRA (включая dropout),
  оптимизатор, lr-расписание, переключатели Soup (gradient_checkpointing, use_liger, use_cut_ce,
  packing, stream_loss_chunk и т.п.), нельзя добавлять обучаемые параметры, отключать/менять выходы Soup.
- **Нельзя трогать `harness/`, `tests/`, `task/`.** Всё наше — только в `solution/` (+ README,
  SUBMISSION.md, docs/).
- Чужой код кернелов копировать нельзя; идеи — можно, **с указанием источника в README**.
- Кернелы примера `solution/swiglu.py` (`example_*`) не засчитываются, даже переписанные
  (скрипт сравнивает структуру AST; сходство ≥ 0.8 читает жюри). Пишем своё с нуля.
- Свои кернелы должны занимать **≥ 5%** времени обучения (с 3-го шага, CUDA events). Цель ≥ 8%.

## 5. Ловушки (обязательно учитывать в коде)

1. **Время компиляции Triton входит в замер, кэш пустой.** Без `@triton.autotune`. Фиксированные
   `BLOCK`/`num_warps`. Размеры, которые меняются от батча к батчу (число строк, длина), передавать
   в аргументах из `do_not_specialize=[...]`, иначе Triton перекомпилирует кернел на каждую новую длину.
2. **Проверка 10**: `apply()` вызывается и перед тестами Soup с фейковыми модулями. Каждый патч
   защищён: `if not x.is_cuda: return original(...)`, `if not isinstance(self.model, torch.nn.Module)` и т.д.
3. **Проверка 5**: имена метрик в логе, `num_tokens`, `entropy`, `mean_token_accuracy`, rewind
   (строки, токены, loss по строкам) — совпадают с базой (числа в пределах 3%).
4. **Проверка 6**: loss первого шага в пределах 1%, среднее по шагам в пределах 3%. Аккумулировать в fp32.
5. Не использовать CUDA graphs (харнесс их не видит). Не создавать фоновых процессов.
6. Dtype-контракт: что пришло в fp32 — возвращаем fp32 (если не доказано, что fp16 без потерь).

## 6. Архитектура `solution/` и владельцы файлов

| Файл | Что | Владелец |
|---|---|---|
| `solution/__init__.py` | `apply()` — вызывает `patch_*()` всех модулей, каждый в своём try/guard | Капитан |
| `solution/swiglu.py` | пример организаторов (оставляем как есть, он экономит ~2 GB) | — |
| `solution/fused_ce.py` | fused lm_head + cross-entropy + entropy + accuracy + loss по токенам | Капитан |
| `solution/mlp.py` | экономия памяти MLP (fp16-хранение / пересчёт) | Капитан |
| `solution/rmsnorm.py` | fused RMSNorm forward/backward (fp32) | Капитан |
| `solution/nf4.py`, `solution/mlp.py` | распаковка NF4; MLP на fp16-активациях | Капитан |
| `solution/rope.py`, `solution/test_rope.py`, `tools/bench_rope.py` | кернел RoPE, тест, бенчмарк | Человек B (`docs/tasks/personB_rope_kernel.md`) |
| `tools/memory_audit.py`, `tools/profile_summary.py`, `docs/memory_audit.md`, `docs/profile_base.md`, `docs/results.md`, `docs/startup_time.md` | замеры, профиль, аудит памяти | Человек A (`docs/tasks/personA_kaggle_audit.md`) |
| `solution/test_*.py` | тесты каждого кернела против PyTorch (fwd + bwd) | автор кернела |
| `tools/kaggle_run.py` | запуск репо на Kaggle T4 из терминала | Капитан |
| `README.md`, `SUBMISSION.md`, презентация, видео | сдача | Капитан + команда |

Интерфейс каждого модуля кернела одинаковый:

```python
# solution/<name>.py
@triton.jit
def <name>_fwd_kernel(...): ...       # имена НЕ начинаются с example_
@triton.jit
def <name>_bwd_kernel(...): ...

class <Name>Function(torch.autograd.Function): ...   # forward / backward

def patch_<name>():                    # идемпотентный монкипатч класса transformers
    ...                                # с fallback на оригинал, если не CUDA / неожиданный dtype
```

и в `solution/__init__.py`:

```python
def apply():
    patch_qwen2_mlp()      # пример
    patch_rmsnorm()        # и т.д.
```

## 7. Как работать

- Ветка на человека: `rmsnorm`, `docs`, `kaggle` и т.д. В `main` — только через PR или после
  зелёного прогона на Kaggle. **`main` всегда должен проходить все 10 проверок.**
- Коммитить маленькими шагами, со своего GitHub-аккаунта (жюри смотрит вклад каждого).
- Локально GPU нет. Тестируем на Kaggle (GPU T4 x2, Internet on, секрет `GITHUB_TOKEN`).
  - быстрая проверка кернела: `python -m pytest solution -q`
  - только наше решение: `python harness/run_task.py --only solution --out runs_try --profile`
  - полный прогон: `python harness/run_task.py` → `python harness/check_rules.py` → `python harness/score.py runs/result.json`
- Результаты каждого прогона записывать в `docs/results.md`.
