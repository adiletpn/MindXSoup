# ПРОМПТ ДЛЯ ЧЕЛОВЕКА A — Kaggle, замеры и аудит памяти

> Как пользоваться: лучше всего — **Claude Code**: открой терминал, запусти `claude` и вставь ВЕСЬ этот
> файл одним сообщением; он сам всё настроит (git, ветку, Kaggle) и будет вести тебя. Можно и в обычную
> нейросеть (ChatGPT / Claude / Gemini) — тогда команды выполняешь сам. Скопируй ВСЁ ниже линии одним
> сообщением. Потом делай по шагам, что она скажет, и присылай ей результаты (вывод ячеек, ошибки).
> В конце пришли капитану (Адилету) то, что указано в разделе «Что сдать капитану».

---

## ⚙️ ЕСЛИ ТЫ — CLAUDE CODE (агент в терминале): СНАЧАЛА НАСТРОЙКА

Пользователь — новичок в git и Kaggle. **Делай всё сам**, командами в терминале, и коротко объясняй,
что сделал. Спрашивай пользователя только то, что знает только он (имя, email, логины, пароли, SMS).

**1. Инструменты.** Проверь `git --version` и `gh --version`. Если чего-то нет — установи
(macOS: `brew install git gh`; Windows: `winget install Git.Git GitHub.cli`). Проверь
`gh auth status`; если не залогинен — запусти `gh auth login` (GitHub.com → HTTPS → login with a web
browser) и подскажи пользователю, что нажать в браузере.

**2. Доступ к репозиторию команды.** `gh repo view BAITC-Hacks/hack-3e361ee8-mindx`.
- Если доступа нет — проверь приглашения: `gh api user/repository_invitations` и прими нужное:
  `gh api -X PATCH user/repository_invitations/<id>`.
- Если приглашения нет — скажи пользователю: «Напиши капитану (Адилету): мне нужен доступ к репо
  команды через платформу edu.astanahub.com, мой GitHub-логин — …». А пока работай в локальной папке
  `~/mindx-work` (код возьми из публичного стартового репо
  `https://github.com/MakazhanAlpamys/soup-kernel-challenge`) и в конце отдай файлы капитану.

**3. Клонирование и контекст.**
```bash
gh repo clone BAITC-Hacks/hack-3e361ee8-mindx ~/mindx
cd ~/mindx
```
Прочитай **целиком** `AGENTS.md`, `docs/ENGINEERING_BRIEF.md`, `docs/QA_NOTES.md` — там вся задача,
правила жюри, что уже сделано и что запрещено. Это важнее, чем этот промпт, если они расходятся.

**4. Подпись коммитов.** Спроси у пользователя имя и email **его GitHub-аккаунта** и выполни в папке
репо: `git config user.name "Имя"` и `git config user.email "email@…"`. Email обязан быть привязан к
GitHub пользователя (github.com/settings/emails), иначе жюри не увидит его вклад — проверь это.

**5. Своя ветка.** `git checkout -b audit` (если ветка уже есть на GitHub — `git checkout audit`).

**6. Работа.** Трогай **только свои файлы**: `tools/memory_audit.py`, `tools/profile_summary.py`, `docs/memory_audit.md`, `docs/profile_base.md`, `docs/results.md`, `docs/startup_time.md`. **Никогда** не меняй `harness/`, `tests/`,
`task/`, не трогай чужие файлы в `solution/` и `solution/__init__.py` (подключает капитан).
Коммить маленькими шагами с понятными сообщениями на английском:
```bash
git add <свои файлы>
git commit -m "Add …"
git push -u origin audit
```
**Никогда:** push в `main`, `git push --force`, коммит токенов, `kaggle.json`, папок `runs*`,
`data/`, `kaggle_out/`.

**7. Kaggle из терминала (вместо кликов в браузере).**
- Пользователь: kaggle.com → регистрация → Settings → **Phone verification** → раздел API →
  **Create Legacy API Key** → скачается `kaggle.json`.
- Ты: положи его в `~/.kaggle/kaggle.json` (`chmod 600`), установи CLI в отдельное окружение:
  `python3 -m venv ~/.venvs/kaggle && ~/.venvs/kaggle/bin/pip install -q kaggle`.
- Запуск на Kaggle T4 одной командой из папки репо (код текущей папки упаковывается в задачу сам):
  - тесты всех кернелов: `~/.venvs/kaggle/bin/python tools/kaggle_run.py tests --slug <ник>-tests`
  - любая команда: `~/.venvs/kaggle/bin/python tools/kaggle_run.py cmd --cmd "python tools/имя.py" --slug <ник>-cmd`
  - результаты скачиваются в `kaggle_out/<режим>/` (логи — в `out/*.log`).
  - `--slug` — латиница и дефисы, уникальный для каждого вида запуска.
- Для этой роли: задачи 1–4 ниже оформи как скрипты в `tools/` и запускай режимом `cmd`; для базового прогона: `--cmd "python harness/run_task.py --only base --out runs_base"`. Файлы, которые скрипт пишет в `/kaggle/working/out`, скачиваются автоматически.

**8. Сдача.** Когда задача готова и проверена:
```bash
gh pr create --base main --head audit --title "…" --body "что сделано, результаты, как проверить"
```
Дай пользователю ссылку на PR и короткий отчёт (что сделано, цифры), чтобы он переслал капитану.

---

## Кто ты и что происходит

Ты — мой напарник-инженер. Я участвую в хакатоне **Decentrathon 6.0**, команда **MindX**
(4 человека), кейс **Soup Kernel Challenge**. Дедлайн — 31 октября 23:59 по Астане.

**Суть кейса.** Есть программа `soup train` (пакет `soup-cli==0.77.0`), которая дообучает языковую
модель **Qwen2.5-1.5B-Instruct** (веса сжаты в 4 бита NF4 через bitsandbytes, обучается LoRA r=16 на
слоях q_proj и v_proj, batch 4, длина до 512 токенов, 64 шага, 256 строк казахских инструкций) на
бесплатной видеокарте **Kaggle T4** (15 ГБ, fp16, без bf16). Сейчас запуск идёт ~80 секунд и
занимает ~14,3 ГБ видеопамяти. Наша команда пишет свои GPU-кернелы на **Triton**, чтобы тот же запуск
шёл быстрее и занимал меньше памяти. Баллы: 20 — пройти все 10 автоматических проверок, 35 —
скорость, 30 — память, 15 — README.

Код кернелов пишет капитан вместе с Claude. **Моя роль — данные и замеры**: базовые цифры, профиль
(куда уходит время GPU) и **аудит памяти** (какие тензоры занимают память). От этих цифр зависит, что
мы будем ускорять дальше. Я **не пишу Triton-кернелы**.

Официальный стартовый репозиторий (публичный):
https://github.com/MakazhanAlpamys/soup-kernel-challenge
Репозиторий нашей команды (приватный): https://github.com/BAITC-Hacks/hack-3e361ee8-mindx
Если у меня нет доступа к командному репо — все задачи ниже делаю на **публичном стартовом**
репозитории: базовые замеры от этого не зависят.

## Правила, которые нельзя нарушать
- Ничего не менять в папках `harness/`, `tests/`, `task/` (это скрипты жюри).
- Во время замеров не запускать в том же ноутбуке ничего другого на GPU.
- Не коммитить токены и ключи. Не публиковать чужие данные.
- Квота Kaggle ~30 часов GPU в неделю: выключать сессию (Stop session), когда не работаю.

---

## ЗАДАЧА 0. Kaggle (10 минут)
1. Зарегистрироваться на kaggle.com, **Settings → Phone verification** (без телефона Kaggle не даёт
   видеокарту и интернет).
2. Create → New Notebook. Справа в панели: **Accelerator: GPU T4 x2**, **Internet: On**.

## ЗАДАЧА 1. Базовый прогон организаторов (≈ 25 минут, из них 15 — ожидание)
В новом ноутбуке выполнить ячейки (нейросеть, помоги мне их оформить):
```bash
!git clone -q https://github.com/MakazhanAlpamys/soup-kernel-challenge /kaggle/working/repo
!python -m pip install -q "soup-cli[train]==0.77.0" "transformers==5.16.1" "peft==0.20.0" "trl==0.29.1" "bitsandbytes==0.50.2" "accelerate==1.14.0" pytest
!cd /kaggle/working/repo && python task/prepare_data.py
!cd /kaggle/working/repo && python harness/run_task.py --only base --out runs_base --profile
```
Внимание: флаг `--profile` профилирует только решение, поэтому для профиля базы нужен отдельный трюк —
см. задачу 2.

Записать из вывода: время `soup train` (с), пик памяти (MiB), eval loss.
Также из `runs_base/base_1/metrics.json` выписать: `train_loop_s`, `median_step_s`, `apply_s`,
`torch_max_allocated_mib`, `torch_max_reserved_mib`.

## ЗАДАЧА 2. Профиль базового шага (≈ 30 минут)
Цель — таблица «какие CUDA-кернелы сколько времени занимают» для **базового** Soup.
Трюк: решение без патчей = база. В копии репо сделать `solution/__init__.py` с пустым `apply()`
(только в ноутбуке, НЕ коммитить!):
```python
!cd /kaggle/working/repo && printf 'def apply():\n    pass\n' > solution/__init__.py
!cd /kaggle/working/repo && python harness/run_task.py --only solution --out runs_prof --profile
```
Профиль будет в `runs_prof/solution_profile/profile.json` (поля `kernels[].name`, `time_us`, `count`,
`total_gpu_us`). Попроси нейросеть написать ячейку, которая:
- печатает топ-30 кернелов по времени (имя, мс за 16 шагов, % от total, число вызовов);
- группирует по типу по ключевым словам в имени: gemm/gemv/cutlass/sm75_xmma → «matmul»;
  `dequantize`/`kDequantize` → «nf4 dequant»; `softmax`/`log_softmax`/`nll` → «softmax/loss»;
  `fmha`/`attention`/`efficient` → «attention»; `copy`/`cast`/`to_copy` → «cast/copy»;
  `elementwise`/`vectorized`/`mul`/`add` → «elementwise»; `reduce` → «reduce»; остальное → «other»;
- печатает итог по группам в процентах.

## ЗАДАЧА 3. Аудит памяти (самое важное, ≈ 1–1,5 часа)
Цель — **доказать цифрами, куда уходят 14 ГБ**. Гипотеза капитана: больше всего памяти занимают
промежуточные активации в fp32 (особенно в MLP: тензоры gate/up по ~73 МБ на каждый из 28 слоёв) и
логиты по словарю 152 тыс. токенов (несколько ГБ), а не сами веса.

Нейросеть, напиши мне ОДНУ ячейку Kaggle (новый ноутбук, GPU T4, Internet On, пакеты как в задаче 1),
которая делает следующее, повторяя подготовку модели как в Soup:
1. Загрузить модель:
```python
import torch, collections
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
q = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                       bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.float16)
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct", quantization_config=q,
                                             device_map={"": 0}, dtype=torch.float16)
model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=False)
model = get_peft_model(model, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05,
                       target_modules=["q_proj", "v_proj"], task_type="CAUSAL_LM"))
model.train()
```
2. Напечатать память после загрузки: `torch.cuda.memory_allocated()/2**20` и сколько весят параметры
   по dtype (сумма `p.numel()*p.element_size()` сгруппировать по `p.dtype` и по классу
   `type(p).__name__`).
3. Синтетический батч: `input_ids = torch.randint(0, 151936, (4, 512), device="cuda")`,
   `labels = input_ids.clone()`, `attention_mask = torch.ones_like(input_ids)`.
4. Зарегистрировать на всех модулях `register_forward_pre_hook`, который пишет имя модуля в
   глобальную переменную `CURRENT`. Через `torch.autograd.graph.saved_tensors_hooks(pack, unpack)`
   записывать каждый сохраняемый для backward тензор: имя текущего модуля, его класс, dtype, shape,
   размер в МБ, `data_ptr()` (для учёта дублей: один и тот же тензор может сохраняться несколько раз —
   считать уникальные по data_ptr отдельно).
5. Forward под `torch.autocast("cuda", dtype=torch.float16)` с `labels`, затем `loss.backward()`.
   До forward сделать `torch.cuda.reset_peak_memory_stats()`. Напечатать: память до forward, после
   forward, пик после backward (`torch.cuda.max_memory_allocated()`).
6. Таблицы (уникальные тензоры):
   - МБ по классу модуля (Qwen2RMSNorm, Qwen2MLP / Linear4bit внутри mlp, внутри self_attn, lora_A/B,
     Dropout, lm_head, Embedding);
   - МБ по dtype (float32 vs float16 vs другие);
   - МБ на один слой `model.layers.N` (для любого одного N) и умножить на 28;
   - топ-20 самых больших сохранённых тензоров (модуль, dtype, shape, МБ);
   - размер логитов `outputs.logits` (dtype, shape, МБ).

Если что-то падает — присылай ошибку нейросети, пусть чинит. Важно получить **цифры**, не идеальный код.

## ЗАДАЧА 4. «Невидимые 20 секунд» (≈ 30 минут, если остаётся время)
Весь запуск ~80 с, шаги обучения — ~56–61 с. Куда уходит остальное? В отдельной ячейке замерить
`time.perf_counter()` до/после: `import torch, transformers, trl, peft, bitsandbytes, soup_cli`;
загрузки модели (код из задачи 3, пункт 1); `prepare_model_for_kbit_training` + `get_peft_model`.
Плюс открыть `runs_base/base_1/train.log` и по сообщениям восстановить хронологию запуска.

---

## Что сдать капитану (одним сообщением в чат команды)
1. Таблица базового прогона: время (с), пик (MiB), eval loss, `train_loop_s`, `median_step_s`.
2. Итог профиля по группам (%), плюс топ-10 кернелов.
3. Аудит памяти: все таблицы из задачи 3 + 3–5 выводов простыми словами
   («больше всего памяти занимает …»).
4. (если сделал) хронология 20 секунд.
Можно текстом или скриншотами. Лучше всего — скопировать вывод ячеек как текст.

## Что от тебя (нейросети) нужно
Веди меня по шагам, давай готовые ячейки целиком (я новичок в Kaggle), объясняй простыми словами, что
значат цифры, и помоги свести итог для капитана. Если ячейка падает — исправь её.
