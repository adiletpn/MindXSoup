# ПРОМПТ ДЛЯ ЧЕЛОВЕКА A — раунд 2: аудит памяти НАШЕГО решения, время старта, сдаточный ноутбук

> Скопируй этот файл целиком в Claude Code (или другую нейросеть) в папке репозитория `~/mindx`.

## ⚙️ ЕСЛИ ТЫ — CLAUDE CODE: НАСТРОЙКА

Настройка уже сделана в раунде 1 (git, gh, Kaggle CLI, `~/.kaggle/kaggle.json`). Если машина новая —
выполни раздел «⚙️» из `docs/tasks/personA_kaggle_audit.md`. Затем:

```bash
cd ~/mindx
git checkout main && git pull
git checkout -b audit2
```

Перечитай `AGENTS.md`, `docs/ENGINEERING_BRIEF.md`, свои `docs/memory_audit.md`, `docs/startup_time.md`.
Пользователь — новичок: делай всё сам командами, коротко объясняй, спрашивай только то, что знает
только он (логины, пароли, клики в браузере).

**Правила git (важно, в раунде 1 был push прямо в main — так больше не делаем):**
- работаем только в ветке `audit2`, в `main` — только через PR;
- сообщения коммитов — понятные, на английском (`Add solution memory audit`, а не `mytask`);
- трогаешь только: `tools/memory_audit.py`, `tools/profile_summary.py`, новые `tools/startup_probe.py`,
  `docs/memory_audit_solution.md`, `docs/startup_time.md`, `docs/results.md` (только добавлять строки);
- никогда: `harness/`, `tests/`, `task/`, `solution/`, `push --force`, токены и `kaggle.json` в git.

## Кто ты и что происходит

Решение команды в `main` проходит все 10 проверок: **1,16× по скорости, 2,26× по памяти**
(6 341 MiB против 14 345). Капитан сейчас ускоряет старт (`apply()` ≈ 9 с, компиляция кернелов),
Человек B — блок attention. Им нужны точные цифры: **куда уходят наши 6,3 ГБ и наши 9 секунд**.
Без них оптимизируем вслепую.

## Задача 1 — аудит памяти С НАШИМ решением (главное)

1. В `tools/memory_audit.py` добавь флаг `--apply-solution`: перед созданием модели
   `import solution; solution.apply()` (как это делает харнесс). Без флага поведение прежнее.
2. Прогони на Kaggle T4 (с `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`, как в харнессе):
   - `--apply-solution --seq-length 512` — главный результат;
   - для сравнения база `--seq-length 384` и решение `--seq-length 384` (одинаковые условия).
   Запуск: `~/.venvs/kaggle/bin/python tools/kaggle_run.py cmd --cmd "…" --slug <ник>-audit2`
   (перед аудитом в команде нужен `python task/prepare_data.py`, если скрипт этого требует).
3. Отчёт `docs/memory_audit_solution.md`:
   - пик allocated, память после загрузки модели, после forward;
   - сохранённые тензоры по группам (MLP, attention, RMSNorm, LoRA, dropout, голова/CE, прочее)
     и по dtype — **база против решения**, таблицей;
   - топ-20 самых больших сохранённых тензоров в решении (модуль, dtype, форма, MiB);
   - вывод одной фразой на группу: «осталось X MiB, это <что>, можно/нельзя уменьшить».
   - Отдельно: сколько весят параметры fp32 (эмбеддинг `embed_tokens` и `lm_head`, нормы) — капитан
     собирается держать эмбеддинг/lm_head только в fp16, нужна точная цифра.

## Задача 2 — на что уходят 9 секунд `apply()` и компиляция

Сделай `tools/startup_probe.py`, прогони на Kaggle T4 (чистый кэш Triton — новый процесс,
`TRITON_CACHE_DIR` во временную папку) и допиши `docs/startup_time.md`:
1. `python -X importtime -c "import solution"` → топ-15 самых долгих импортов (сек).
   Особо: какие из них (`transformers`, `peft`, `trl`, `bitsandbytes`, `triton`) харнесс всё равно
   импортировал бы позже — это не наши потери.
2. Время каждого `patch_*()` из `solution/__init__.py` по отдельности.
3. **Время первой компиляции каждого нашего кернела** на настоящих формах (Qwen2.5-1.5B:
   hidden 1536, 12 q-голов, 2 kv-головы, head_dim 128, intermediate 8960, словарь 151936,
   batch 4, T=512, fp32 вход как в обучении): `rmsnorm` fwd/bwd, `rope_kernel`, `head_ce_rows_kernel`,
   `nf4_unpack_kernel`, а также пример `example_*` из `solution/swiglu.py`. Табличка
   «кернел → секунд на первый вызов → секунд на второй вызов».
4. Проверь: параллелится ли компиляция Triton в нескольких потоках Python (скомпилировать 3 кернела
   последовательно и в 3 `threading.Thread` одновременно, сравнить время). Это решает, сработает ли
   фоновая компиляция, которую делает капитан.

## Задача 3 — сдаточный Kaggle-ноутбук (снимаем риск в последний день)

1. На kaggle.com: New Notebook → File → Import → `templates/kaggle_notebook.ipynb` из репо.
   Accelerator **GPU T4 x2**, Internet **on**.
2. Первая ячейка: `REPO_URL = "https://github.com/BAITC-Hacks/hack-3e361ee8-mindx.git"`, `COMMIT = ""`.
3. Секрет `GITHUB_TOKEN` (Add-ons → Secrets): github.com → Settings → Developer settings →
   Fine-grained tokens → Repository access: только `hack-3e361ee8-mindx`, Permissions: **Contents:
   Read-only**. Если организация не разрешает такой токен — скажи пользователю написать капитану.
4. **Save & Run All**. Ноутбук должен дойти до конца: обе тренировки, оценка, проверки.
5. Сделать ноутбук **публичным** (Share → Public). Токен в секретах не виден другим — проверь, что он
   нигде не печатается в выводе.
6. Ссылку на ноутбук и итог (время, пик, проверки) — в `docs/results.md` новой строкой.
   В последний день капитан скажет финальный коммит — тогда впишешь его в `COMMIT` и сделаешь
   ещё один Save & Run All.

## Сдача

```bash
git add <только свои файлы>
git commit -m "Add solution memory audit and startup probe"
git push -u origin audit2
gh pr create --base main --head audit2 --title "Solution memory audit, startup probe, submission notebook" --body "…"
```
В описании PR: 5–10 строк главных цифр (что занимает память у нас, что съедает время старта,
параллелится ли компиляция, ссылка на публичный ноутбук). Дай пользователю ссылку на PR —
он перешлёт капитану.

## Порядок и время

Задача 1 → 2 → 3. Ориентир: 1 — ~1,5 ч, 2 — ~1 ч, 3 — ~1 ч (сам прогон ~15 мин).
Если что-то не выходит за 30 минут — коммить что есть, опиши проблему в PR и иди к следующей.
