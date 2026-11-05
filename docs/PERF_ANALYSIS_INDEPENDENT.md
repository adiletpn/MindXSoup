# Независимый разбор: время + пиковая память GPU (Soup Kernel Challenge)

Контекст: команда MindX, Decentrathon 6.0, кейс Soup Kernel Challenge.  
Репозиторий прочитан (`docs/CHALLENGE.md`, `task/task.yaml`, `harness/`, `solution/`, профили, `docs/results.md`, QA).  
Ниже — независимая модель затрат. Где цифры из прогонов команды — **факт**; остальное — **оценка**. Код в конце — эскизы, не готовый merge; на GPU в этой сессии не запускался.

---

## Задача (кратко)

`soup train` (soup-cli 0.77.0) дообучает Qwen2.5-1.5B-Instruct:

- веса 4-bit NF4 с двойным квантованием (bitsandbytes), compute dtype fp16;
- LoRA r=16, alpha=32, цели q_proj и v_proj, dropout 0.05;
- batch 4, до 512 токенов, 64 шага, adamw_torch, косинусное расписание, logging_steps=1;
- SDPA, без gradient checkpointing;
- Kaggle Tesla T4, fp16 (bf16 нет), `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.

Решение — `solution.apply()`: свои Triton-кернелы + монкипатчи transformers / peft / bitsandbytes / Soup.

**Замер:** wall-clock всего `soup train` (включая импорты, load, компиляцию Triton; кэш пустой) + пик nvidia-smi каждые 100 мс за весь процесс.

**Баллы:** `35·ln(s)/ln(s_best) + 30·ln(m)/ln(m_best)`, s/m = база/решение. Выигрыш &lt; 5% = 0. Одна FAIL-проверка → 0 за всё.

Известные цифры (T4):

| | Значение |
|---|---|
| База | 78–92 с, 14 345 MiB |
| Лучшее по времени (команда) | 42,4 с @ 6793 MiB |
| Лучшее сданное по памяти | 58–65 с @ 505 MiB |
| Лучшая память вообще | 411–429 MiB @ 91–99 с (проверка 8 FAIL) |
| CUDA-контекст | ~133 MiB |

Текущий `solution/` (`PROFILE = "rules_memory"`): fused CE / head_stream, unpad, fp16 MLP, layer recompute + host fp16 inputs, stream weights, startup, warmup → сдача ~**505 MiB / 58–65 с**, 10/10.

---

## 1. Модель затрат

### 1.1. Физика шага (база)

Qwen2.5-1.5B, NF4 + LoRA на `q_proj`/`v_proj`, batch 4×≤512. Residual после `prepare_model_for_kbit_training` — **fp32** (эмбеддинги, нормы, lm_head в fp32 → весь residual-поток fp32). bitsandbytes считает матмул в fp16 и **приводит выход обратно в fp32**.

**Память шага (база)** — доминируют не веса:

| Статья | Порядок | Источник |
|---|---|---|
| Сохранённые fp32-активации (MLP gate/up/SiLU ×28, attn, norms) | ~8–10 GiB | **факт** audit: fp32 saved ≈10 080 MiB при 4×512 |
| Полные логиты + CE/TRL/Soup | ~1.2–4 GiB пика | **факт** CHALLENGE + audit: `[N,V]` fp32 ~1.2 GiB |
| NF4 веса + absmax | ~0.65 GiB | **факт** |
| tied embed/lm_head | ~0.45–0.9 GiB | **факт** |
| LoRA + Adam | десятки MiB | **оценка** |
| CUDA context / cuBLAS / SDPA | ~133 MiB context; workspace сверху | **факт** (context); workspace — **оценка** 100–300 MiB |

Пик базы **14 345 MiB** — **факт** харнесса.

**Время GPU шага (база, profiler)** — `docs/profile_base.md`:

| Группа | Доля GPU time |
|---|---:|
| matmul | ~39% |
| elementwise | ~23% |
| cast/copy | ~18% |
| attention | ~6% |
| softmax/CE | ~6% |
| NF4 dequant | ~5% |

Важно: ~44% «поэлементное + касты» — следствие fp32-потока и round-trip bnb fp16→fp32, а не «тяжёлой математики».

**Физический предел compute на T4:** fp16 tensor-core GEMM по *реальным* токенам + минимальный softmax по словарю V. Всё остальное (касты, лишние сохранения, pad, полные логиты, Python launches) — налог.

В MLP для backward база хранит `gate` и `up` в fp32 (~73 MB каждый на слой × 28). Они получены из fp16-матмула → хранить в fp16 **без потерь**.

Логиты `[4, T, 151936]` в fp32 (~1.25 GB при T=512) + log_softmax + градиент = 3–4 GB на пике. TRL дополнительно считает по логитам `entropy` и `mean_token_accuracy`, Soup — loss каждой строки для `rewind.jsonl`. В Soup уже есть «шаг без логитов»: `ChunkedStep` / `_soup_chunked_step`.

### 1.2. Вне цикла (весь `soup train`)

**Факт:** wall ≈ 78–92 с, train loop ≈ 56–61 с → вне цикла ≈ **20–28 с** (импорты, load/quantize, tokenize/`Dataset.map`, save, холодная компиляция Triton).

Пик nvidia-smi — **за весь процесс**. Если при `from_pretrained` на GPU одновременно лежат все NF4 + head, этот пик **нельзя** убрать стримингом «только в train» — веса должны уходить на host **в момент материализации**.

### 1.3. Что уже закрыто в `solution/` (калибровка)

| Фича | Файл / флаг |
|---|---|
| fused CE + метрики без полных логитов | `fused_ce.py` |
| head в pinned + срезы словаря | `head_stream.py` |
| unpad | `unpad.py` |
| fp16 MLP / gate-up recompute | `mlp.py`, `mlp_recompute.py` |
| layer recompute + host fp16 inputs + group tokens | `recompute.py` |
| NF4 stream с host | `stream.py`, `linear4bit.py`, `nf4.py` |
| RMSNorm / RoPE / attention fp16 | `rmsnorm.py`, `rope.py`, `attention.py` |
| startup / warmup | `startup.py`, `warmup.py` |

Дальше речь про **оставшийся зазор** и независимый Pareto, не про «с нуля».

---

## 2. Оценка нижней границы

### 2.1. Память (процессный пик)

Нижняя граница при offload всех крупных статичных тензоров при load + полный recompute + group tokens:

- context ≈ 133 MiB (**факт**)
- 1–2 слота весов слоя ≈ 25–50 MiB
- live активации одной группы токенов (не всего батча)
- head slice buffers (SLICE×hidden×2) ≈ 12–24 MiB
- LoRA/Adam + мелочь
- cuBLAS workspace (сжимаемый)

**Оценка пола:** **~350–450 MiB**.  
Согласуется с абсолютным минимумом команды **411–429 MiB** (**факт**), который ломал проверку 8 по времени.  
Сдача **505 MiB** уже близка к полу; дальше — копейки и риск времени.

### 2.2. Время (wall)

**Оценка несжимаемого:**

- вне цикла после всех легальных трюков: **~15–18 с**
- 64 шага при unpad (~30–40% меньше токенов), без полных логитов, с overlap H2D: медиана шага **~0.25–0.35 с** → loop **~16–22 с**

**Оценка совместного оптимума «и то и то»:** **~40–48 с @ ~0.7–2.0 GiB**, а не 505 MiB.

| Край | Результат | Комментарий |
|---|---|---|
| Только память | ~411 MiB, wall → к базе или хуже | проверка 8 |
| Только скорость | **42.4 с @ 6793 MiB** (**факт**) | веса/активации на GPU |
| Балльный компромисс | **~45–55 с и 1.0–2.5 GiB** | **оценка** |

По формуле `35 ln s / ln s* + 30 ln m / ln m*`: при неизвестных best’ах и уже m≳20× маржинальный **1% wall** часто дороже **1% MiB**. Имеет смысл целиться в скорость при удержании m≳10–15×, а не в абсолютный memory floor.

---

## 3. Идеи по убыванию ожидаемого выигрыша в баллах

Оценки Δt/ΔMiB — **не результаты прогона**; где есть A/B из репо — указано как **факт**.

### 1. Убить полные логиты: fused/chunked head + метрики через `ChunkedStep`

- **Механизм:** cuBLAS по чанкам строк × (или срезам V) + Triton online-softmax → loss / entropy / argmax / dH; патч `Qwen2ForCausalLM.forward` + TRL metrics; Soup rewind через `_soup_chunked_step`. Режим Soup `stream_loss_chunk` **не** включать (запрещённый переключатель).
- **Ожидание:** −3…4 GiB; −несколько % GPU-времени CE/копий. У команды уже есть (`fused_ce.py` / `head_stream.py`).
- **Риски:** пр. 5 (entropy / accuracy / rewind), 6–7 (накопление fp32, порядок argmax «первый максимум»).
- **Дешёвая проверка:** A/B `--only solution`; сравнить `trainer_state` + rewind; **fail** если \|Δentropy\| > 3% или first loss > 1%.

### 2. Не хранить fp32-активации: fp16 save + recompute elementwise / слоёв

- **Механизм:** MLP/attn сохраняют fp16 (бит-идентично выходам fp16-GEMM); поэлементное пересчитывать в bwd; опционально `torch.utils.checkpoint` на слоях (**разрешено** своим кодом, не флагом Soup `gradient_checkpointing`).
- **Ожидание:** −4…8 GiB vs база; скорость ± от бюджета.  
  **Факт:** budget 0 → 1739 MiB; host inputs + stream → 505 MiB; budget 256→512: **−4.25 с, +464 MiB** (`docs/runtime_profile.md`).
- **Риски:** 6 (dropout masks — только штатный RNG / `F.dropout`, свой генератор **запрещён**), 8 (перекос времени), 10 (фейковые модули в тестах Soup).
- **Проверка:** ABBA на одном T4 по `BUDGET_TOKENS`; **fail** если wall > base×1.1 или кривая loss вне допуска.

### 3. Unpad внутри батча + attention по строкам

- **Механизм:** упаковать реальные токены; SDPA per-row `is_causal`; патч decoder / attention / loss indexing. Смешивать строки между батчами нельзя; pad внутри батча убирать — можно.
- **Ожидание:** **факт** у команды ~−12% wall на шагах и −1 GiB в середине пути; типично 30–40% лишних pad-токенов при скошенном распределении длин.
- **Риски:** 5 (`num_tokens`, rewind lengths), 6 (маски / порядок).
- **Проверка:** побитово сравнить loss на одном батче packed vs padded; **fail** при расхождении labelled-token count.

### 4. Load-time offload NF4 + head в pinned + prefetch по PCIe

- **Механизм:** при квантовании/load увести `Params4bit` и таблицу V×H на host; ring GPU slots; side-stream H2D.  
  **Факт PCIe:** 12.4 GB/s; 24 MiB слоя ≈ 2 мс полностью прячется за GEMM слоя.
- **Ожидание:** пик с ~1.7 GiB → ~0.5–1.2 GiB; время ≈ 0…+3 с если overlap плохой. У команды сделано (`stream.py`, `head_stream.py`).
- **Риски:** 5–7 если copy stream гоняет буфер, который ещё читает compute (**факт:** был FAIL до sync); 8.
- **Проверка:** bit-identical vs GPU-resident weights; **fail** на любом mismatch.

### 5. Int8 GEMM замороженных проекций (fwd и/или только dX) — главный недожатый рычаг скорости

- **Механизм:** NF4→int8 per-row scale; активации dynamic int8; `torch._int_mm`; LoRA / norm / attn / loss в точности базы. Патч `solution/linear4bit.py` / `NF4LinearFunction`.  
  QA 50 / 54 / 59 / 60: **разрешено** (в т.ч. lm_head int8 по QA 60; уточнение QA 54 про lm_head сначала считали неподтверждённым, QA 60 — да).
- **Ожидание:** **оценка** −2…6 с wall при удачном покрытии больших GEMM; память почти без изменений (или чуть меньше, если не держать fp16 unpack). Bench dX команды: **~1 с ceiling** — мало; **fwd int8 не взят** — там потенциал больше, но с численным риском.
- **Риски:** **высокие** — пр. 6–7. Правила ок при 10/10; серая зона только если «тихо» меняется оптимум без прохождения допусков.
- **Проверка:** сначала forward-only int8 на 8 шагах; **fail** если first step loss Δ > 0.5% уже на smoke (запас до лимита 1%). Два seed (1 и 7).

### 6. Срезать launch tax (тысячи мелких кернелов)

- **Механизм:** fused residual+RMSNorm; один Triton dropout keep-mask (у команды есть); меньше Python в Linear4bit (`nf4_direct`).  
  **Факт профиля:** при streaming HtoD и matmul доминируют; раньше отмечали ~33% «idle» на Python / launches.
- **Ожидание:** **оценка** −1…4 с; память ~0.
- **Риски:** 6 (нумерика RMSNorm — HF cast нормализованного ряда до умножения на weight); 4 (свои кернелы ≥5% — fusion обычно помогает share).
- **Проверка:** CUDA events share + median step ABBA; **fail** если median не −≥3%.

### 7. Startup: метаданные, optional imports, Triton cache key, nice warmup, preload весов

- **Механизм:** уже в `startup.py` / `warmup.py`; QA 63 — дешёвый hash ключа Triton (~−1 с **оценка**). Компиляция только в **потоке** того же процесса (не отдельный процесс). Можно понизить nice потока компиляции.
- **Ожидание:** суммарно уже выжали несколько секунд (**факт**); остаток **оценка** −0.5…2 с.
- **Риски:** 9 / 2 (не stub `sys.modules`); 10.
- **Проверка:** wall outside-loop timestamps; **fail** если apply растёт сильнее выигрыша.

### 8. Микробатч 2×2 строки на один optimizer step (QA 64)

- **Механизм:** два forward/backward полубатча; тот же суммарный loss / знаменатель; маски в том же порядке; один `optimizer.step`.
- **Ожидание:** пик активаций ≈×0.5 при том же recompute budget → **оценка** −100…400 MiB на текущем Pareto **или** возможность **поднять** `BUDGET_TOKENS` (меньше recompute) при том же пике → выигрыш во времени.
- **Риски:** 5–6 (нормализация loss, dropout order), 10.
- **Проверка:** bit-compare grads LoRA vs один полный backward; **fail** при max \|Δ\| > 1e-3 rel.

### 9. Сжать CUDA stack / cuBLAS workspace / skip allocator warmup

- **Механизм:** `cudaDeviceSetLimit` (у команды 256 B); уменьшить workspace cuBLAS; skip `caching_allocator_warmup` / `empty_cache` (уже частично).
- **Ожидание:** **оценка** −20…80 MiB; на фоне 505 MiB мало баллов.
- **Риски:** редкие kernel stack faults → нужен штатный fallback.
- **Проверка:** peak smi + stress длинный батч.

### 10. Int8 / NF4 хранение таблицы lm_head на host (QA 62)

- **Механизм:** меньше PCIe на голову → быстрее memory-режим без роста пика.
- **Ожидание:** **оценка** −10…30 мс/шаг на H2D; риск нумерики головы.
- **Риски:** 5–7 сильнее, чем у layer int8.
- **Проверка:** только после стабильного fp16 `head_stream`.

---

## 4. Самая неочевидная идея

**Управлять не «памятью шага», а моментом пика nvidia-smi и формой score-Pareto.**

Пик — max по **всему** процессу с дискретизацией 100 мс. Команды оптимизируют tensors в backward, а выигрывают те, кто:

1. **Никогда** не держит на GPU одновременно load-резидентность (все NF4 + head) и train-рабочие буферы — offload **в момент материализации**, не «после первого шага».
2. Сознательно выбирает точку на кривой  
   `BUDGET_TOKENS × GROUP_TOKENS × head SLICE × microbatch`,  
   максимизируя `35 ln(s) + 30 ln(m)` при **гипотезе** о `s_best` / `m_best`, а не минимум MiB.  
   При уже m≈28× добивание 505→420 даёт мало ln m; прыжок 60 с→45 с даёт заметно больше ln s — **оценка**.

Дополнительные «тихие» ходы:

- **int8 только в backward dX** (QA 59) — first-step loss и eval path как у базы, меньше риска пр. 6; bench команды сказал, что выигрыш мал.
- Наоборот, неочевидно: **int8 lm_head при стриминге** (QA 60) — при memory-профиле голова часто на **критическом пути PCIe**, а не compute.

---

## 5. Что НЕ стоит делать

- Писать **матрицы в Triton** на T4 (`tl.dot` ≪ cuBLAS) — подтверждено участниками / QA.
- `@triton.autotune` и specialize по длине батча — компиляция в замере, кэш пустой; длины батчей разные → перекомпиляции. Переменные размеры — в `do_not_specialize=[...]`.
- Копировать Liger / Unsloth / пример `example_*` (AST-сходство ≥ 0.8 читает жюри).
- Включать Soup-переключатели: `gradient_checkpointing`, `use_cut_ce`, `use_liger`, packing, `stream_loss_chunk` и т.п.
- Свой RNG для LoRA dropout; свой `Dataset.map` fingerprint / своя токенизация вместо Soup `Dataset.map`.
- `gc.freeze`, stubs в `sys.modules`, фоновый **процесс** компиляции, CUDA graphs ради «5%» (не засчитываются в долю своих кернелов).
- Гнать пик &lt;450 MiB ценой wall хуже base×1.1 (проверка 8) — **факт** прогона 411–429 MiB.
- Int8 everywhere без двух seed на пр. 6–7.
- Кэшировать все NF4→fp16 на GPU в memory-профиле (`nf4_cache_fp16`) — скорость ценой памяти.
- Трогать `harness/`, `tests/`, `task/`.
- Фоновые процессы, другие настройки аллокатора, готовые кэши компиляции, другое поведение под замером.

---

## 6. Жёсткие проверки (одна FAIL = 0)

1. Все запуски exit 0.  
2. Нет banned libs (unsloth, liger, xformers, flash-attn, torchao, …).  
3. Свои `@triton.jit` реально работают в обучении.  
4. Свои кернелы ≥ 5% времени с 3-го шага (CUDA events); `example_*` не считаются.  
5. Шаги, адаптер, метрики, tokens, rewind; entropy/accuracy/row loss в пределах 3%.  
6. Loss первого шага ≤ 1%, среднее ≤ 3%.  
7. Eval loss не хуже базы больше чем на 0.5%.  
8. Время и память не хуже базы больше чем на 10%.  
9. Чистый замер: T4, один GPU, аллокатор харнесса.  
10. 79 регрессионных тестов Soup с патчами; каждый патч с fallback на оригинал (фейковые модули).

---

## 7. Практический вывод

- **С нуля** максимальный dual-win — связка идей **(1)+(2)+(3)+(4)** (fused head, fp16/recompute активаций, unpad, load-time stream).
- **У MindX** эта связка уже сдана около memory floor (~505 MiB).
- Дальнейшие баллы почти наверняка в **сдвиге влево по времени при удержании m≳10–15×**:
  1. int8 на больших frozen GEMM (с жёстким гейтом по loss),
  2. микробатч / budget Pareto (`BUDGET_TOKENS` ↑ при том же пике),
  3. остаточный launch / startup tax —  
  а **не** в охоте за ещё −50 MiB.

Целевая зона: **~45–55 с @ 1.0–2.5 GiB** (**оценка**), а не минимум MiB.

---

## 8. Код-эскизы двух сильнейших оставшихся идей

### A) Int8 frozen Linear4bit (fwd + dX) — главный рычаг времени на текущем Pareto

Точка патча: заменить тело `NF4LinearFunction` / stream-вариант в `linear4bit.py`.

```python
# solution/int8_linear.py — sketch only
import torch
from .nf4 import unpack_nf4  # or unpack_nf4_to_int8 kernel

def _quantize_activation(x: torch.Tensor):
    # x: [M, K] fp16/fp32 -> int8 + per-row scale
    x32 = x.float()
    scale = x32.abs().amax(dim=-1).clamp_min(1e-6) / 127.0  # [M]
    x8 = (x32 / scale[:, None]).round().clamp(-128, 127).to(torch.int8)
    return x8, scale

def _nf4_to_int8_weight(packed, state, absmax=None):
    # Prefer a Triton kernel that writes int8 + row scales directly (no fp16 temporary).
    w_fp16 = unpack_nf4(
        packed, state.shape, state.state2.absmax, state.blocksize,
        torch.float16, absmax_8bit=absmax, code=state.state2.code,
        offset=state.offset,
    )
    w_scale = w_fp16.abs().amax(dim=1).clamp_min(1e-6) / 127.0  # [N]
    w8 = (w_fp16.float() / w_scale[:, None]).round().clamp(-128, 127).to(torch.int8)
    return w8.contiguous(), w_scale  # _int_mm wants B column-major via .t()

class NF4Int8LinearFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, packed, state, bias, absmax=None):
        w8, w_scale = _nf4_to_int8_weight(packed, state, absmax)
        x8, x_scale = _quantize_activation(x)
        # sm75: layout of B matters — bench first (_int_mm vs fp16)
        y32 = torch._int_mm(x8, w8.t())  # [M, N] int32
        y = y32.float() * (x_scale[:, None] * w_scale[None, :])
        if bias is not None:
            y = y + bias
        ctx.save_for_backward(x)
        ctx.packed, ctx.state, ctx.absmax = packed, state, absmax
        return y.to(x.dtype)

    @staticmethod
    def backward(ctx, grad):
        if not ctx.needs_input_grad[0]:
            return None, None, None, None, None
        # Safer: fp16 dX as today. QA59: int8 dX only.
        w8, w_scale = _nf4_to_int8_weight(ctx.packed, ctx.state, ctx.absmax)
        g8, g_scale = _quantize_activation(grad)
        dx32 = torch._int_mm(g8, w8)  # grad @ W
        dx = dx32.float() * (g_scale[:, None] * w_scale[None, :])
        return dx.to(grad.dtype), None, None, None, None
```

**A/B план:** только слои intermediate ≥ 8960; first-step loss + eval на seed 1 и 7; **fail** если first loss Δ > 0.5% на smoke.

### B) Микробатч 2+2 на один шаг — память ↓ или тот же пик при меньшем recompute (скорость)

Точка патча: обёртка `training_step` / forward модели после `unpad` (не флаги Soup). QA 64 разрешает.

```python
# solution/microbatch.py — sketch only
import torch

def split_packed_batch(batch, lengths, n_first=2):
    """Split packed tokens into first n_first rows vs rest; keep row order for dropout."""
    assert len(lengths) == 4
    cut = sum(lengths[:n_first])
    return (0, cut), (cut, sum(lengths))

def microbatch_loss_and_backward(model, batch, lengths, dropout_draw_fn):
    """Same mean loss denominator as full batch; one optimizer step outside."""
    total_labelled = 0
    loss_sum = None
    # Draw ALL dropout masks once for full packed shape (order = base), then slice.
    masks = dropout_draw_fn(lengths)
    slices = [(0, 2), (2, 4)]
    for a, b in slices:
        tok0 = sum(lengths[:a])
        tok1 = sum(lengths[:b])
        mb = {
            k: (v[tok0:tok1] if torch.is_tensor(v) and v.shape[0] == sum(lengths) else v)
            for k, v in batch.items()
        }
        out = model(**mb, _mindx_masks={m: masks[m][tok0:tok1] for m in masks})
        n = out.num_tokens  # must match TRL/Soup accounting
        total_labelled += n
        part = out.loss * n
        loss_sum = part if loss_sum is None else loss_sum + part
        (part).backward()  # accumulate into LoRA grads; scale carefully vs base
    loss = loss_sum / total_labelled
    return loss
```

**Практичнее в текущем коде:** расширить уже существующий `GroupedLayer` / `GROUP_TOKENS` в `recompute.py` так, чтобы **сначала** резать по строкам батча (2+2), и только потом по `GROUP_TOKENS` внутри — пик ≈ max(строки группы), а не сумма. Тогда можно **поднять** `BUDGET_TOKENS` (меньше пересчёта) при том же smi-пике — движение к 42–50 с без возврата к 6+ GiB.

---

## 9. Разрешённое организаторами (шпаргалка из QA / ТЗ)

Можно:

- свои autograd + монкипатчи transformers / peft / bitsandbytes / Soup;
- свой пересчёт слоёв, включая `torch.utils.checkpoint` (запрещён только переключатель Soup);
- хранение активаций в fp16 / int8;
- замороженные веса и embed/lm_head в fp16 / int8 / pinned CPU;
- int8-матмулы замороженных весов и lm_head, если все проверки проходят;
- удаление паддинга внутри батча;
- fused head + loss + метрики без полного тензора логитов через `ChunkedStep`;
- компиляция кернелов в потоке того же процесса (+ понижение nice);
- обработка батча по частям (тот же loss/знаменатель, один шаг оптимизатора, маски в прежнем порядке);
- fused / свой Triton AdamW; состояние AdamW в pinned RAM;
- уменьшение стека CUDA и workspace cuBLAS; пропуск `caching_allocator_warmup`, `empty_cache`;
- не импортировать неиспользуемые sklearn / wandb / scipy через проверки доступности; кэш `packages_distributions()`; предчтение файлов весов в потоке; ранняя инициализация CUDA в потоке.

Нельзя (кроме уже перечисленного в §5): менять модель, данные, порядок батчей, шаги, LoRA, оптимайзер, расписание, выходы Soup; `PRAGMA synchronous=NORMAL`; `gc.freeze`; `os._exit`; stubs в `sys.modules`; свой fingerprint токенизатора.

---

## 10. Как гонять проверки (напоминание)

```bash
python -m pytest solution -q
python harness/run_task.py --only solution --out runs_try --profile
python harness/run_task.py            # base + solution
python harness/check_rules.py
python harness/score.py runs/result.json
SOUP_APPLY_SOLUTION=1 python -m pytest tests -q   # проверка 10
```

Результаты писать в `docs/results.md`. Не менять `harness/`, `tests/`, `task/`.
