# ПРОМПТ ДЛЯ ЧЕЛОВЕКА B — свой Triton-кернел RoPE

> Как пользоваться: скопируй ВСЁ ниже линии в свою нейросеть (ChatGPT / Claude / Gemini) одним
> сообщением. Делай по шагам, присылай ей вывод ячеек и ошибки. В конце пришли капитану (Адилету) то,
> что указано в разделе «Что сдать капитану».

---

## Кто ты и что происходит

Ты — мой напарник по GPU-программированию. Я участвую в хакатоне **Decentrathon 6.0**, команда
**MindX**, кейс **Soup Kernel Challenge**. Дедлайн — 31 октября 23:59 по Астане.

**Суть кейса.** Программа `soup train` (пакет `soup-cli==0.77.0`) дообучает модель
**Qwen2.5-1.5B-Instruct** (веса в 4 бита NF4, LoRA на q_proj/v_proj, batch 4, до 512 токенов,
64 шага) на бесплатной видеокарте **Kaggle T4** (sm_75, 15 ГБ, fp16, **без bf16**). Команда пишет
свои GPU-кернелы на **Triton** (Triton 3.6.0, torch 2.11), которые подключаются через монкипатч и
ускоряют обучение / экономят память, не меняя результат.

Важное правило жюри: **наши собственные Triton-кернелы вместе должны занимать ≥ 5% времени обучения**
(считается CUDA-событиями вокруг каждого запуска кернела). Капитан уже написал кернелы RMSNorm,
fused cross-entropy и распаковку NF4. **Моя задача — ещё один кернел: RoPE (rotary position
embedding), forward + backward.** Он добавит долю времени наших кернелов и немного ускорит attention.

## Жёсткие правила кейса (нарушение = дисквалификация)
- Код кернела пишем **сами с нуля**. Нельзя копировать код из unsloth, liger-kernel, flash-attn,
  xformers и подобных библиотек (идеи можно, с указанием источника). Нельзя их импортировать.
- Имена кернелов **не должны начинаться с `example_`**.
- Нельзя `@triton.autotune` (время компиляции входит в замер и съедает выигрыш).
- Нельзя `torch.compile` для этого кернела.

## Что сейчас считает RoPE (transformers 5.16.1, файл modeling_qwen2.py)
```python
def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed
```
Вызов в `Qwen2Attention.forward`:
```python
query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)   # [B, 12, T, 128]
key_states   = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)   # [B, 2,  T, 128]
cos, sin = position_embeddings                                                  # [B, T, 128] каждый
query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
```
Факты для нашей задачи:
- B = 4, T — от ~50 до 512, **разная в каждом батче**; 12 голов q, 2 головы k, head_dim = 128.
- **q и k приходят в fp32** (в этой модели весь поток данных fp32), cos/sin — тоже fp32. Но кернел
  должен работать и с fp16.
- q и k — **не contiguous** (это `view(...).transpose(1, 2)` от тензора `[B, T, heads*128]`).
  Кернел должен либо принимать strides, либо работать с исходной раскладкой `[B, T, heads, 128]`
  (это даже удобнее: память идёт подряд). Нельзя молча делать `.contiguous()` — это лишняя копия.
- cos одинаковый для первой и второй половины (emb = cat(freqs, freqs)), но не полагайся на это —
  читай полный cos/sin.

## ЧТО НАПИСАТЬ — файл `rope.py`
1. `@triton.jit` кернел `rope_fwd_kernel`:
   - одна программа обрабатывает одну позицию (b, t) и все головы q и k (или одну голову — на выбор,
     но меньше запусков лучше);
   - читает cos/sin [128] для этой позиции один раз, применяет к каждой голове;
   - формула для головы x длины 128 (x1 = x[:64], x2 = x[64:]):
     `out1 = x1*cos1 - x2*sin1`, `out2 = x2*cos2 + x1*sin2`;
   - вычисления в fp32, результат в dtype входа; пишет в новый выходной тензор той же формы и strides,
     что вход (или в contiguous-выход — тогда возвращать его как view с нужной формой).
2. `@triton.jit` кернел `rope_bwd_kernel` — поворот на противоположный угол:
   `dx1 = dy1*cos1 + dy2*sin2`, `dx2 = dy2*cos2 - dy1*sin1`.
   Проверь формулу через autograd (тест ниже) — не доверяй мне на слово.
3. Все размеры, которые меняются от батча к батчу (T, strides по T и B), передавать в аргументах из
   `@triton.jit(do_not_specialize=["n_tokens", ...])` — иначе Triton перекомпилирует кернел на каждую
   новую длину батча. `BLOCK` и `num_warps` — фиксированные константы.
4. `class RoPEFunction(torch.autograd.Function)` с forward/backward; сохраняет для backward только
   cos и sin (не q/k — они не нужны для градиента!).
5. `def apply_rope(q, k, cos, sin, unsqueeze_dim=1)` — тот же контракт, что у оригинала, возвращает
   `(q_embed, k_embed)`.
6. `def patch_rope()`:
```python
from transformers.models.qwen2 import modeling_qwen2
original = modeling_qwen2.apply_rotary_pos_emb
def patched(q, k, cos, sin, unsqueeze_dim=1):
    if (not q.is_cuda or q.dtype not in (torch.float16, torch.float32) or q.shape[-1] != 128
            or unsqueeze_dim != 1 or cos.dim() != 3):
        return original(q, k, cos, sin, unsqueeze_dim)
    return apply_rope(q, k, cos, sin)
modeling_qwen2.apply_rotary_pos_emb = patched
```
   (функция вызывается из модуля `modeling_qwen2` по имени, поэтому подменять надо там).

## ТЕСТ — файл `test_rope.py` (pytest)
- forward и backward против оригинальной формулы на PyTorch (через autograd):
  dtype fp32 и fp16; формы q `(4, 12, 512, 128)`, k `(4, 2, 512, 128)` и нечётные `(3, 12, 77, 128)` /
  `(3, 2, 77, 128)`;
- q и k делать **так же, как в модели**: `torch.randn(B, T, heads*128).view(B, T, heads, 128).transpose(1, 2)`
  (не contiguous!);
- cos/sin как в модели: `freqs = pos[:, :, None] * inv_freq[None, None, :]`, `emb = cat(freqs, freqs)`;
- допуски: fp32 `rtol=atol=1e-5`, fp16 `2e-3`.

## БЕНЧМАРК
Ячейка: время forward+backward нашего кернела против оригинала на формах задачи (прогрев 10 итераций,
замер 100 итераций, `torch.cuda.Event`), и **время первой компиляции** (первый вызов, в секундах).
Плюс проверка перекомпиляции: вызвать с T = 100, 200, 300 и убедиться, что второй и третий вызовы
не компилируют заново (время первого вызова для новой T должно быть маленьким).

## КАК ЗАПУСКАТЬ (своей видеокарты нет — используем Kaggle)
1. kaggle.com → регистрация → **Settings → Phone verification**.
2. Create → New Notebook → справа **Accelerator: GPU T4 x2**, **Internet: On**.
3. Ячейка установки: `!pip install -q "transformers==5.16.1" pytest` (torch и triton уже есть в образе).
4. Код кернела записать в файл ячейкой `%%writefile rope.py`, тест — `%%writefile test_rope.py`,
   запустить `!python -m pytest test_rope.py -q`.

## ГОТОВО, КОГДА
- все тесты зелёные на T4 в fp32 и fp16;
- fwd+bwd быстрее оригинала;
- первая компиляция ≤ 2 с, повторной компиляции на новых T нет.

## ЧТО СДАТЬ КАПИТАНУ
- файлы `rope.py` и `test_rope.py` целиком (текстом или файлами);
- вывод pytest и бенчмарка (скриншот или текст);
- если есть доступ к репо https://github.com/BAITC-Hacks/hack-3e361ee8-mindx — запушить в ветку
  `rope` со своего GitHub-аккаунта как `solution/rope.py` и `solution/test_rope.py` (это важно: жюри
  смотрит вклад каждого участника). Если доступа нет — просто отправить файлы капитану.

## Что от тебя (нейросети) нужно
Напиши код полностью и аккуратно. **Объясни мне каждую часть кернела простыми словами** — мне это
защищать в презентации. Если тесты падают — разбери ошибку и исправь.
