# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Что это за проект

«Компилятор» запуска LLM под конкретное железо: найти и сохранить самый быстрый способ запуска модели на этой машине без потери качества. Постановка задачи — `ТЗ для Claude Code «компилятор» запуска LLM под конкретное железо.md`, прочитай её целиком перед содержательной работой.

**Веха 1 достигнута (2026-11-07):** на Qwen3.6-35B-A3B (UD-Q4_K_M), RTX 4070 SUPER 12 ГБ + 64 ГБ RAM наш движок даёт 124.1 / 130.9 ток/с (short / long 17K) против 106.0 / 108.9 у лучшего llama.cpp. Корректность подтверждена (BENCHMARKS.md, «ИТОГ ВЕХИ 1»).
- Движок — форк llama.cpp «moe-zc» (гибридные эксперты GPU+CPU, DECISIONS.md D6), «компилятор» — `aic.py`.
- Ночью 2026-11-08 (D9) движок и сборка обобщены на другие модели: GLM-4.7-Flash, Gemma 4 26B-A4B, Qwen3.8-Flash-Next, маленькие Qwen3.5. Проведено сравнение с ik_llama.cpp и Strata.
- Дальнейшие направления — конец D9.
- Замеры делать при схеме питания «Высокая производительность» с закрытым фоном и сравнивать только участников внутри одного прогона.
- Владелец разрешил скачивать любые модели, если хватает места на диске. ktransformers отложен до сервера с двумя RTX 3090.

**Новая сессия начинает с `HANDOFF.md`:** состояние, воспроизведение одной командой, где лежат модели и Strata, известные проблемы и незаконченные эксперименты.

## Документы и правила работы (из ТЗ)

- Вся документация и отчёты на **русском**. Владелец пишет на C++/Qt, CUDA не знает: объясняй GPU-понятия простыми словами, когда это нужно для понимания решения (словарь — `docs/concepts.md`).
- `DECISIONS.md` — решения с вариантами и причинами, **включая неудачные эксперименты** (записи D1, D2, …; пересмотр — новой записью вроде D4a, старая не удаляется).
- `BENCHMARKS.md` — каждый замер с командой, условиями (включая свободную VRAM) и путём к сырым логам.
- `PROGRESS.md` — короткий журнал для владельца, новые записи добавляются в конец с датой и временем.
- `docs/survey.md` — обзор похожих проектов (llama.cpp, Strata, MLC-LLM и др.).
- Любое ускорение подтверждается **сквозным замером** реального запуска (`bench/serverbench.py`), а не микробенчмарком. Всё, что меняет вывод модели (квантование, приближения), явно помечается и измеряется.
- Спроси владельца, прежде чем скачивать модели больше нескольких ГБ, ставить системные пакеты, драйверы или CUDA Toolkit, или делать что-то за пределами папки проекта.
- Заимствования из открытых проектов (llama.cpp/ggml — MIT) отмечаются.

**Важно:** в `.gitignore` стоит `*.md`, поэтому новые `.md` в git попадают только через `git add -f`.

## Окружение

Windows 11, оболочка по умолчанию — PowerShell; `.bat`-скрипты сборки сами вызывают `vcvars64.bat`.

- MSVC 17.8: `D:\MicrosoftVS\2022\Enterprise\VC\Auxiliary\Build\vcvars64.bat`.
- CUDA **13.5**: `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.5`. В системе есть и устаревшая 11.8, поэтому путь к 13.5 в скриптах указывается явно, на `CUDA_PATH` не полагаться. Архитектура `sm_89`.
- Python 3.12 (системный, без venv); `cachesim.py` требует numpy.
- GPU одновременно выводит рабочий стол: Windows занимает 1.2–2.5 ГБ VRAM, и это плавает. Свободную VRAM надо учитывать при раскладке и записывать в замерах.
- CPU i7-13700KF: логические 0–15 — 8 P-ядер с HT, 16–23 — E-ядра (маска `-C 0xFFFF` = только P-ядра).

## Что не в git (большие или внешние данные)

| Путь | Что там |
|---|---|
| `models/` | основная модель `Qwen3.6-35B-A3B-UD-Q4_K_M.gguf`, драфты `mtp-…Q4_0.gguf`, `dflash-…Q8_0.gguf` |
| `third_party/llama-bin/` | официальные бинарники llama.cpp b12189 (CUDA 13.5) — эталон по умолчанию |
| `third_party/llama.cpp/` | исходники того же коммита `80d2c0c2c`, держим нетронутыми |
| `build/llama-ref/` | сборка этих исходников нашим тулчейном (для сверки с официальной и для линковки инструментов) |
| `data/wikitext-2-raw/` | текст для perplexity и калибровочных промптов |
| `bench/logs/`, `bench/prof/` | сырые логи замеров и профили nsys |

## Команды

Сборка (из корня проекта):
```
tools\build_llamacpp.bat [src_dir] [build_dir]   # по умолчанию third_party\llama.cpp -> build\llama-ref; цели llama-server, llama-completion, llama-perplexity, llama-bench
tools\hwbench\build.bat                          # потолки железа: RAM, VRAM, PCIe, zero-copy, запуск ядер
tools\trace_experts\build.bat                    # линкуется с build\llama-ref, поэтому сначала build_llamacpp
tools\vmmtest\build.bat                          # можно ли смешать страницы VRAM и RAM в одном виртуальном диапазоне
```

Сквозной замер (главный инструмент; работает с любым сервером, говорящим на API `/completion` llama-server, поэтому наш движок будет мериться им же):
```
python bench/serverbench.py --tag <имя> [--only short0,long16k] [--reps 3] [--n-probs N] [--server path.exe] -- <аргументы llama-server>
```
Пишет `bench/logs/<дата>_<tag>.jsonl`: ток/с, приём драфта, текст и id токенов, свободную VRAM. Перед замером идёт прогрев промптом `warmup`, намеренно непохожим на тестовые (чтобы не «обучить» кэш экспертов на тестовых данных). Промпты фиксированы в `bench/prompts.json` (`short0..7`, `long16k`); их генерирует `bench/make_prompts.py` один раз, результат закоммичен.

Перебор настроек эталона (дописывает `bench/sweep_results.csv`, он в git):
```
python bench/sweep_llamacpp.py <placement|threads|moecache|spec|combo|combo2> [--base "<общие аргументы>"] [--only ...] [--reps N]
```

Корректность — побитового совпадения нет даже между режимами самого llama.cpp, поэтому сравниваем id токенов и смотрим, приходится ли первое расхождение на «почти ничью» (top-N вероятности через `--n-probs`), плюс perplexity (`llama-perplexity` на wikitext-2):
```
python bench/compare_outputs.py <reference.jsonl> <candidate.jsonl> [...]
```

Профиль шага генерации:
```
nsys profile --trace=cuda,nvtx --cuda-graph-trace=node --export=sqlite <llama-completion ...> -n 300
python tools/profsplit.py bench/prof/<file>.sqlite --tokens 200 --n-gen 300
```

Трассы выбора экспертов и офлайн-модель кэша (проверка гипотезы D4a до написания ядра):
```
python bench/make_calib_prompts.py                                  # -> bench/calib_prompts.json (отдельно от тестовых промптов)
python bench/export_prompts.py <prompts.json> <out.txt>              # формат «один промпт в строке» для C++
tools\trace_experts\trace_experts.exe <model.gguf> <prompts.txt> <out.bin> [n_gen=256]
python tools/cachesim.py <eval.bin> [--calib <calib.bin>] [--mib 5500 6500 7500]
```
Горячий набор экспертов оценивается на трассе, отличной от той, на которой он выучен (`--calib`); без него результат оптимистичен.

Тестов и линтера в проекте нет; проверка — это замеры и сравнение выводов выше.

## Текущая картина (подробности и цифры — в DECISIONS.md и BENCHMARKS.md)

- **Эталон для вехи:** лучший llama.cpp — `-ngl 99 -cmoe --moe-cache-mib 6500 -lm none -t 8 -md models/mtp-Qwen3.6-35B-A3B-Q4_0.gguf --spec-type draft-mtp --spec-draft-n-max 3`. Даёт 106 / 109 ток/с; шум между запусками ±3 ток/с.
- **Движок:** форк `third_party/llama-fork` (ветка `engine` — git worktree от `third_party/llama.cpp`), патчи — в `engine/patches/`. Сборка: `tools\build_llamacpp.bat third_party\llama-fork build\llama-fork`. Включается переменными окружения:
  - `LLAMA_MOE_ZC_MIB` — VRAM под кэш экспертов;
  - `LLAMA_MOE_ZC_THREADS` — CPU-потоки, считающие экспертов;
  - эксперты должны лежать в RAM (`-cmoe`), а собственный пул llama — `-t 1`.

  Код — `ggml/src/ggml-cuda/moe-zc.cu` (менеджер, почта GPU↔CPU, CPU-потоки), хуки в `mmvq.cu` и `mmq.cu`, `src/llama-moe-zc.cpp`.
- **Компилятор:** `python aic.py build|run|show --model <gguf>`, профили — в `cache/`, копии — в `bench/profiles/`.
  - Режимы: `moe-zc`, `gpu`, `fit`.
  - Драфты: встроенный MTP, файлы `mtp-<base>*.gguf` рядом с моделью (все пробуются), маленькая модель того же токенизатора.
  - Длина драфта в `moe-zc` не больше 3 (`ZC_MAX_TOKENS` = 4 токена на шаг).
  - В профиль пишется схема питания.
- **Strata:** `third_party/strata/ctl2/bin/strata.cmd launch ... --model-path <наши файлы> --no-tui --yes`, остановка — `strata launch --stop`. Замер — `bench/strata_bench.py`.
- **Итоговые замеры:** `python bench/milestone.py [--model M] [--contenders F]` (участники — `bench/contenders/*.json`, в т.ч. «наш движок с заданной конфигурацией» через `ours_cfg`), `MODEL=… bash bench/quality.sh`. Промпты для новой модели — `bench/make_model_prompts.py` (шаблон чата из GGUF, при его отсутствии — `bench/templates/`). ik_llama.cpp: `tools/build_ik.bat`, сервер запускать через `serverbench.py --plain`, вместо `-lm none` — `-nmm`.
- **Важные грабли:**
  - на WDDM вызовы CUDA API из второго потока задерживают запуск графа в основном потоке — копировать пакетом и после запуска графа;
  - CUDA-графы переигрывают решения, принятые на хосте при записи;
  - spin-барьеры CPU-потоков ломаются, если с ними конкурирует другой пул потоков.
- Архитектура модели, важная для планирования:
  - 40 слоёв: 30 Gated DeltaNet + 10 обычного внимания, KV-кэш ~20 КБ/токен;
  - 256 экспертов на слой по ~1.81 МиБ;
  - на токен 8 выбранных экспертов + 1 общий.
