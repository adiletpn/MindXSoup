# Промпт для Участника 3 — аудит памяти, README и сдача

> Скопируй ВЕСЬ этот файл в свою нейросеть. Если она видит репозиторий — попроси сначала прочитать
> `AGENTS.md`, `docs/STRATEGY.md`, `docs/ENGINEERING_BRIEF.md` и `templates/SUBMISSION.md`.
> Если не видит — вставь их содержимое следом.

---

Ты — мой напарник на хакатоне Decentrathon 6.0, кейс **Soup Kernel Challenge** (команда MindX).
Мы пишем свои Triton-кернелы, которые ускоряют дообучение Qwen2.5-1.5B на Kaggle T4 и уменьшают
расход видеопамяти. Контекст — `AGENTS.md`, стратегия — `docs/STRATEGY.md`.
Репозиторий: https://github.com/BAITC-Hacks/hack-3e361ee8-mindx

## Моя роль
1. **Аудит памяти** — доказать цифрами, куда уходят ~14 ГБ (это основа нашей стратегии и слайд
   в презентации).
2. **Всё, что сдаём кроме кода**: README (15 баллов!), `SUBMISSION.md`, презентация, видео.

## Мои файлы
- `tools/memory_audit.py`, `docs/memory_audit.md`
- `README.md`, `SUBMISSION.md`, `docs/presentation_outline.md`, `docs/video_script.md`
- **НЕЛЬЗЯ:** `harness/`, `tests/`, `task/`, `solution/`

## Задача 1. `tools/memory_audit.py` (запуск на Kaggle T4)
Скрипт повторяет то, как Soup готовит модель, и считает, что хранится для backward:
1. Загрузить `Qwen/Qwen2.5-1.5B-Instruct` в 4 бита (`BitsAndBytesConfig(load_in_4bit=True,
   bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.float16)`),
   `device_map={"": 0}`.
2. `peft.prepare_model_for_kbit_training(model, use_gradient_checkpointing=False)`, затем LoRA:
   `LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, target_modules=["q_proj","v_proj"],
   task_type="CAUSAL_LM")`, `get_peft_model`.
3. Синтетический батч: `input_ids` `[4, 512]` случайные токены, `labels = input_ids`.
4. Под `torch.autocast("cuda", dtype=torch.float16)` и
   `torch.autograd.graph.saved_tensors_hooks(pack, unpack)` сделать forward с loss. В `pack`
   записывать dtype, shape, размер в МБ и имя текущего модуля (через forward pre-hooks на всех
   модулях можно хранить «текущий модуль» в глобальной переменной). Внимание: один тензор может
   сохраняться несколько раз — считать уникальные по `data_ptr()` отдельно.
5. Вывести таблицы: суммарно МБ по типу модуля (`Qwen2RMSNorm`, `Qwen2MLP`, `Linear4bit`,
   `lora`, attention, `lm_head`/loss), по dtype (fp32 vs fp16), топ-20 самых больших тензоров;
   плюс `torch.cuda.memory_allocated()` до forward, после forward и пик после backward.
6. Результат — в `docs/memory_audit.md` (таблицы + 3–5 выводов простыми словами).

Гипотеза, которую проверяем: основная память — fp32-активации (особенно MLP: gate/up по ~73 МБ на
слой), а не только логиты. Подтвердить или опровергнуть цифрами.

## Задача 2. Каркас README (по критериям жюри, 15 баллов)
Жюри ставит до 5 баллов за каждую часть:
1. **Что сделано** — какие операции ускорили и почему, как устроены кернелы, **откуда идеи
   (с источниками)**: Liger Kernel FLCE (arXiv 2410.10989), Apple Cut Cross-Entropy (arXiv 2411.09009),
   Triton tutorials (layer norm). Код при этом свой.
2. **Как запустить** — шаги без вопросов к команде: Kaggle-ноутбук (GPU T4 x2, Internet on, Run All),
   ссылка на публичный ноутбук и на финальный коммит, локальные команды из исходного README.
3. **Результаты и ограничения** — таблица против базы (время, пик памяти, eval loss, speedup,
   memory ratio, доля наших кернелов) из нашего прогона и честно: что не получилось, где слабее.

Сейчас пиши каркас с плейсхолдерами `TODO` для цифр. Цифры подставим из `docs/results.md`.
Важно: README должен заменить README организаторов, но сохранить раздел лицензии.

## Задача 3. `SUBMISSION.md`
Скопировать `templates/SUBMISSION.md` в корень и заполнить всё, что известно (команда, роли,
GitHub/Kaggle логины). Ссылки и цифры — `TODO` до финала.

## Задача 4. Презентация и видео (завтра)
- `docs/presentation_outline.md`: 8–10 слайдов. Обязательно: какую операцию ускоряли и почему;
  как устроен кернел (схема); результаты против базы (скорость, память, качество); ограничения и
  следующие шаги. Плюс слайд «куда уходила память» из аудита.
- `docs/video_script.md`: сценарий до 3 минут, что показывать на экране (Kaggle-ноутбук с итоговой
  таблицей `run_task.py`).

## Правила
- Работаю в ветке `docs`, коммичу со своего GitHub-аккаунта, PR в `main`.
- Не коммитить данные задачи, токены, ключи.

## Что от тебя (нейросети) нужно
Код аудита целиком и готовый к запуску в ячейке Kaggle; тексты README/SUBMISSION — по-английски
или по-русски (README лучше по-английски), чёткие, без воды.
