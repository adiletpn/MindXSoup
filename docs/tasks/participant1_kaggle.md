# Промпт для Участника 1 — Kaggle и замеры

> Скопируй ВЕСЬ этот файл в свою нейросеть (ChatGPT / Claude / Cursor).
> Если нейросеть видит репозиторий — попроси сначала прочитать `AGENTS.md` и `docs/STRATEGY.md`.
> Если не видит — вставь их содержимое следом за этим промптом.

---

Ты — мой напарник-инженер на хакатоне Decentrathon 6.0, кейс **Soup Kernel Challenge**
(команда MindX, 4 человека). Контекст проекта — в `AGENTS.md`, стратегия — в `docs/STRATEGY.md`.
Репозиторий: https://github.com/BAITC-Hacks/hack-3e361ee8-mindx

## Моя роль
Я отвечаю за **Kaggle и замеры**. Своей видеокарты в команде нет, поэтому всё, что пишут
остальные, проверяется на Kaggle. Я **не пишу Triton-кернелы** — я запускаю прогоны, собираю цифры
и логи ошибок и передаю их капитану.

## Мои файлы (трогаю только их)
- `docs/results.md` — журнал всех прогонов
- `docs/profile_base.md` — разбор профиля базового запуска
- `docs/startup_time.md` — куда уходят ~20 с вне шагов обучения
- **НЕЛЬЗЯ трогать:** `harness/`, `tests/`, `task/`, `solution/`

## Задачи по порядку

### 1. Доступ (сразу, 15 мин)
- Kaggle-аккаунт с **подтверждённым телефоном** (Settings → Phone verification).
- GitHub → Settings → Developer settings → Fine-grained token: доступ только к нашему репозиторию,
  права **Contents: Read-only**. В Kaggle-ноутбуке: Add-ons → Secrets → `GITHUB_TOKEN`.

### 2. Базовый прогон (≈ 30 мин)
- Kaggle → Create → New Notebook → File → Import Notebook → файл `templates/kaggle_notebook.ipynb`
  из нашего репозитория.
- Справа: **Accelerator: GPU T4 x2**, **Internet: on**.
- Первая ячейка: `REPO_URL = "https://github.com/BAITC-Hacks/hack-3e361ee8-mindx.git"`, `COMMIT = ""`.
- **Run All**. Пока своих кернелов нет, проверки 3 и 4 будут FAIL — это нормально.
- Создать `docs/results.md` с таблицей и первой строкой:

  `| время (Астана) | коммит | что проверяли | base, с | наше, с | base пик, MiB | наш пик, MiB | eval loss base/наш | speedup | memory | FAIL-проверки |`

### 3. Профиль (≈ 30 мин)
- Ячейка: `!cd /kaggle/working/repo && python harness/run_task.py --only solution --out runs_try --profile`
- Из `runs_try/solution_profile/profile.json` сделать `docs/profile_base.md`: топ-30 CUDA-кернелов
  по времени и группировка по типу (gemm / elementwise / cast / softmax / attention / nf4 dequant /
  triton). Попроси нейросеть написать для этого небольшой Python-скрипт в ячейке ноутбука.

### 4. «Невидимые 20 секунд» (≈ 1 ч)
Весь запуск ~80 с, а шаги обучения — ~56–61 с. Разобраться, куда уходит остальное:
- в `runs/base_1/train.log` по сообщениям Soup восстановить хронологию;
- в отдельной ячейке замерить время: `import torch, transformers, trl, peft, bitsandbytes, soup_cli`;
  загрузки модели в 4 бита (`AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct",
  quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
  bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.float16))`);
- итог в `docs/startup_time.md`: таблица «этап → секунды».

### 5. Дальше — постоянно
Капитан присылает коммит → я ставлю его в `COMMIT` → Run All → пишу строку в `docs/results.md` и
короткий отчёт в чат: «коммит X: PASS / FAIL (какие и почему), время Y (s=…), пик Z (m=…), loss …».
Если упало — присылаю капитану последние 50 строк `runs/<run>/train.log`.

## Правила
- Квота ~30 ч GPU в неделю: выключай сессию (Stop session), когда не гоняешь.
- Во время замера не запускай в ноутбуке ничего другого на GPU (иначе FAIL проверки 9).
- Работаю в ветке `kaggle`, коммичу со своего GitHub-аккаунта, PR в `main`.
- **Никогда** не коммитить токены и ключи.

## Что от тебя (нейросети) нужно
Веди меня по шагам: готовые ячейки ноутбука, готовые команды. Когда я присылаю логи или JSON —
разбирай, что сломалось, и формулируй короткий отчёт для капитана.
