# ПРОМПТ ДЛЯ ЧЕЛОВЕКА A — Kaggle, замеры и аудит памяти

> Как пользоваться: скопируй ВСЁ ниже линии в свою нейросеть (ChatGPT / Claude / Gemini) одним
> сообщением. Потом делай по шагам, что она скажет, и присылай ей результаты (вывод ячеек, ошибки).
> В конце пришли капитану (Адилету) то, что указано в разделе «Что сдать капитану».

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
