# Промпт для Участника 2 — кернел RoPE и микробенчмарки

> Скопируй ВЕСЬ этот файл в свою нейросеть. Если она видит репозиторий — попроси сначала прочитать
> `AGENTS.md`, `docs/ENGINEERING_BRIEF.md` и `solution/swiglu.py` (пример организаторов).
> Если не видит — вставь их содержимое следом.

---

Ты — мой напарник-инженер по GPU на хакатоне Decentrathon 6.0, кейс **Soup Kernel Challenge**
(команда MindX). Мы пишем свои Triton-кернелы, которые ускоряют дообучение Qwen2.5-1.5B на Kaggle T4.
Контекст — `AGENTS.md`, технические детали — `docs/ENGINEERING_BRIEF.md`.
Репозиторий: https://github.com/BAITC-Hacks/hack-3e361ee8-mindx

## Моя часть
Свой **fused RoPE (rotary position embedding) forward + backward на Triton** и тест к нему.
Капитан потом встроит его в attention-блок. Плюс маленький скрипт-бенчмарк.

## Мои файлы (трогаю только их)
- `solution/rope.py`
- `solution/test_rope.py`
- `tools/bench_rope.py`
- **НЕЛЬЗЯ:** `harness/`, `tests/`, `task/`, `solution/__init__.py` (подключает капитан), чужие файлы.

## Что считает RoPE сейчас (transformers 5.16.1, Qwen2)
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
Формы в нашей задаче: `q` — `[B, 12, T, 128]`, `k` — `[B, 2, T, 128]`, `cos`/`sin` — `[B, T, 128]`.
B = 4, T — до 512 (разная в каждом батче). **В этой модели q и k приходят в fp32** (это важно), но
кернел должен работать и с fp16. `q` и `k` могут быть не contiguous (это view после transpose) —
учитывай strides или делай `.contiguous()` с пометкой в комментарии.

## Требования к кернелу
1. Имена кернелов: `rope_fwd_kernel`, `rope_bwd_kernel` (**не** начинаются с `example_`).
   Код пишем с нуля, не копируя чужие кернелы (unsloth, liger и т.п. — запрещены правилами).
2. Один запуск обрабатывает и q, и k (или два запуска одного кернела) — без лишних копий в памяти.
3. Вычисления в fp32 внутри кернела, результат — в dtype входа.
4. Backward: это поворот на противоположный угол:
   `dq = dq_embed * cos + rotate_half_T(dq_embed) * sin`, где `rotate_half_T(g) = cat(g2, -g1)`.
   Проверь формулу тестом против autograd.
5. **Не использовать `@triton.autotune`.** Фиксированные `BLOCK` и `num_warps`.
6. Все размеры, которые меняются от батча к батчу (T, число строк, strides), передавать в аргументах,
   перечисленных в `@triton.jit(do_not_specialize=[...])` — иначе Triton перекомпилирует кернел на
   каждую новую длину батча, а компиляция входит в замер времени.
7. Обёртка `torch.autograd.Function` (`RoPEFunction`) и функция
   `apply_rope(q, k, cos, sin) -> (q_embed, k_embed)` с тем же контрактом, что у оригинала.
8. Функция `patch_rope()`: заменяет `transformers.models.qwen2.modeling_qwen2.apply_rotary_pos_emb`
   на нашу версию, **с fallback на оригинал**, если тензоры не на CUDA или dtype не fp16/fp32.
   Внимание: функцию нужно подменить в модуле `modeling_qwen2` (там её вызывает attention).

## Тест `solution/test_rope.py`
По образцу `solution/test_kernels.py`: forward и backward против оригинальной PyTorch-формулы,
dtype fp32 и fp16, формы `(4, 12, 512, 128)` / `(4, 2, 512, 128)` и нечётные `(3, 12, 77, 128)`,
плюс случай, когда q — не contiguous view. Допуски: fp32 `1e-5`, fp16 `2e-3`.

## Бенчмарк `tools/bench_rope.py`
Время forward + backward нашего кернела против оригинала на формах задачи (`torch.cuda.Event`,
прогрев 10 итераций, замер 100), плюс время первой компиляции кернела (в секундах).

## Как проверять (своей GPU нет)
Пушишь ветку `rope` → Участник 1 (или ты сам на Kaggle, ноутбук с GPU T4) запускает:
`python -m pytest solution/test_rope.py -q` и `python tools/bench_rope.py`.
Результат и код показываешь капитану.

## Готово, когда
- тесты зелёные на T4 в fp32 и fp16;
- бенчмарк показывает, что fwd+bwd быстрее оригинала;
- компиляция ≤ 2 с;
- ветка `rope` запушена со своего GitHub-аккаунта, PR в `main`.

## Что от тебя (нейросети) нужно
Напиши код полностью, объясни мне каждую часть кернела простыми словами (мне его защищать в
презентации), и помоги прочитать ошибки, если тесты упадут.
