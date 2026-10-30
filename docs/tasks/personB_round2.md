# ПРОМПТ ДЛЯ ЧЕЛОВЕКА B — раунд 2: блок attention без лишних fp32 и кастов

> Скопируй этот файл целиком в Claude Code (или другую нейросеть) в папке репозитория `~/mindx`.

## ⚙️ ЕСЛИ ТЫ — CLAUDE CODE: НАСТРОЙКА

Настройка уже сделана в раунде 1. Если машина новая — раздел «⚙️» из `docs/tasks/personB_rope_kernel.md`.

```bash
cd ~/mindx
git checkout main && git pull
git checkout -b attention
```

Перечитай `AGENTS.md`, `docs/ENGINEERING_BRIEF.md`, `docs/profile_base.md`, `docs/memory_audit.md`,
и код `solution/mlp.py`, `solution/rope.py`, `solution/rmsnorm.py`, `solution/__init__.py`.

**Трогаешь только:** новый `solution/attention.py`, новый `solution/test_attention.py`,
`solution/rope.py`, `solution/test_rope.py`, `tools/bench_*.py`, `docs/results.md` (добавлять строки).
`solution/__init__.py` подключает капитан — в своей ветке для прогона можешь временно добавить вызов,
но в PR явно напиши, что именно изменил. Никогда: `harness/`, `tests/`, `task/`, чужие файлы,
push в `main`, `--force`.

## Кто ты и что происходит

Твой RoPE в `main`, все 10 проверок проходят, доля наших кернелов 8,7%. Наш счёт сейчас:
**скорость 1,16×** (слабое место), память 2,26×. По профилю базы (`docs/profile_base.md`)
**касты и поэлементные операции — 41% времени GPU**. В MLP капитан уже перевёл всё на fp16
(`solution/mlp.py`). Блок attention пока работает «как в базе» — это твоя задача.

## Что сейчас происходит в attention (transformers 5.16.1, `Qwen2Attention.forward`)

```python
query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)   # LoRA
key_states   = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)   # без LoRA
value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)   # LoRA
cos, sin = position_embeddings
query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)  # = наш RoPE
attn_output, _ = attention_interface(self, q, k, v, attention_mask, dropout=0.0, scaling=..., ...)  # SDPA
attn_output = self.o_proj(attn_output.reshape(*input_shape, -1).contiguous())
```

Факты (проверь сам по исходникам в установленных пакетах):
- `hidden_states` приходит **fp32** (нормы/эмбеддинг держит в fp32 `prepare_model_for_kbit_training`).
- bnb `Linear4bit` кастит вход в fp16, считает матмул в fp16 (bias тоже приводится к fp16) и
  **возвращает результат в dtype входа** — fp32. Значит выход `k_proj` в fp32 — это точное fp16-число.
- peft LoRA (`peft/tuners/lora/bnb.py`, `Linear4bit.forward`): `result = base(x)`, затем
  `lora_B(lora_A(dropout(x))) * scaling`, приведение к dtype результата, сложение. LoRA есть только у
  `q_proj` и `v_proj`, dropout 0.05.
- cos/sin приходят в dtype hidden (fp32) → наш RoPE выдаёт fp32 → SDPA под autocast снова кастит
  q, k, v в fp16. Итого лишние fp32-тензоры и 3+ каста на слой, ×28 слоёв, туда и обратно.

## Задача

Сделать путь attention, где всё, что и так fp16, остаётся fp16, при этом **результат обучения тот же**
в пределах допусков (проверка 6: loss первого шага ±1%, среднее ±3%; eval loss — как у базы).

1. **Сначала замер (30 мин).** На Kaggle (режим `cmd`) профилем `torch.profiler` одного слоя
   `Qwen2DecoderLayer` fwd+bwd на реальных формах (batch 4, T=512, fp32 вход, с `solution.apply()`)
   посмотри: какие касты/копии и fp32-тензоры есть в attention, сколько мс и MiB. Запиши в PR.
2. **Патч `patch_qwen2_attention_fp16()` в `solution/attention.py`** (по образцу `solution/mlp.py`):
   - охранники: только CUDA, только fp32/fp16 вход, только когда проекции — bnb `Linear4bit` с
     compute_dtype fp16 (или peft-обёртка над ним), `past_key_values is None`, обучение без
     `output_attentions`; иначе — оригинальный forward (апплай вызывается и перед тестами Soup с
     фейковыми модулями — патч не должен их ломать);
   - один каст входа `x16 = hidden_states.half()` на все три проекции;
   - **LoRA dropout**: маска должна получаться той же — тот же порядок вызовов (q, потом v) и та же
     форма тензора; проверь, что numel и порядок генерации случайных чисел не поменялись;
   - RoPE: научи `apply_rope` выдавать fp16 сразу (параметр выходного dtype), считая внутри в fp32;
   - SDPA на fp16 q/k/v (под autocast он и так fp16), выход `o_proj` вернуть в dtype входа (fp32) —
     residual-поток остаётся fp32, как в базе.
   - Где можно, совмести касты/сложения в своём Triton-кернеле (например, LoRA-сумма
     `base + lora*scaling` → fp16 в один проход, или RoPE, читающий fp32 и пишущий fp16) — это ещё и
     растит долю наших кернелов. **Не писать матмулы в Triton** (на T4 `tl.dot` без tensor cores, в 30
     раз медленнее cuBLAS). Без `@triton.autotune`; размеры, меняющиеся от батча, — в
     `do_not_specialize`. Каждый новый кернел = ~2,5 с компиляции в замере, так что новый кернел
     оправдан, только если экономит больше; лучше расширить существующий `rope_kernel`.
3. **Тест `solution/test_attention.py`**: слой `Qwen2Attention` с реальными размерами и
   NF4-проекциями + LoRA (или их точными копиями) — патч против оригинала: выход и градиенты по входу
   и по LoRA-весам совпадают с допуском (fp16-уровень, `rtol≈1e-2`), dropout в eval-режиме для
   сравнения, и отдельный тест с dropout при одинаковом seed. Плюс тест на CPU/фейковый модуль →
   уходит в оригинал.
4. **Полный прогон**: в своей ветке временно добавь `patch_qwen2_attention_fp16()` в `apply()`, затем
   `~/.venvs/kaggle/bin/python tools/kaggle_run.py full --slug <ник>-attn-full`.
   Смотри: все 10 проверок PASS, eval loss, медиана шага, `train_loop`, пик памяти, доля кернелов —
   против последней строки в `docs/results.md`. Запиши строку в `docs/results.md`.

## Готово, когда

- тесты `python -m pytest solution -q` на Kaggle (режим `tests`) все зелёные;
- полный прогон: 10/10 PASS, eval loss в пределах 0,1% от базы, медиана шага ниже, чем без патча;
- если выигрыша нет или проверка падает — это тоже результат: опиши в PR, что мерил и почему не вышло.

## Сдача

```bash
git add solution/attention.py solution/test_attention.py solution/rope.py solution/test_rope.py docs/results.md
git commit -m "Add fp16 attention path: one input cast, RoPE writes fp16"
git push -u origin attention
gh pr create --base main --head attention --title "fp16 attention path" --body "…"
```
В PR: что сделано, таблица до/после (шаг, train_loop, пик, eval loss, доля кернелов), что изменено
в `__init__.py`. Ссылку на PR — пользователю, он перешлёт капитану.

Ориентир по времени: 3–4 часа. Если за 2 часа нет рабочей версии — коммить что есть и пиши в PR.
