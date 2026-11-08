# LLM «компилятор» под конкретное железо

Программа «собирает» самый быстрый способ запуска выбранной LLM на конкретном компьютере (по аналогии с компиляцией шейдеров под видеокарту), сохраняет результат и дальше запускает модель по нему. Постановка задачи: `ТЗ для Claude Code «компилятор» запуска LLM под конкретное железо.md`.

**Статус:** веха 1 достигнута. Машина: RTX 4070 Super 12 ГБ + i7-13700KF + 64 ГБ DDR5.

Проверено на моделях (итоги — в DECISIONS.md, D9, цифры — в BENCHMARKS.md):
- Qwen3.6-35B-A3B — основная модель вехи 1;
- GLM-4.7-Flash, два файла;
- Gemma 4 26B-A4B;
- Qwen3.8-Flash-Next;
- маленькие Qwen3.5 0.8–9B.

Сравнение с llama.cpp, ik_llama.cpp и Strata.

**Чего ждать при повторе.** Абсолютные цифры зависят от состояния машины: схемы питания и фоновых программ, которые занимают GPU. Сравнивать нужно участников внутри одного прогона.
- Веха 1 (до перезагрузки 2026-11-08): наш движок 124.1 / 130.9 ток/с, llama.cpp 106.0 / 108.9.
- После перезагрузки, схема «Сбалансированная», браузеры в фоне: наш ~118 / 131, llama.cpp ~98 / 100.
- Схема «Высокая производительность», фон закрыт: наш ~121 / 137, llama.cpp ~106 / 107 (с кэшем 7500).

## Как это работает (коротко)

Модель не помещается в VRAM. Движок — наш форк llama.cpp с «гибридными экспертами» (подробно в DECISIONS.md, D6):
- часто используемые эксперты лежат в кэше VRAM, их считает GPU;
- остальных одновременно считает CPU прямо из RAM;
- GPU и CPU общаются через закреплённую память, без остановок на каждом слое.

«Сборка» (`aic build`) реальными короткими прогонами подбирает под машину:
- сколько VRAM отдать под кэш экспертов;
- длину MTP-драфта для спекулятивного декодирования;
- сколько потоков CPU считают экспертов.

Результат сохраняется в `cache/*.json`.

Для модели, которая целиком помещается в VRAM, сборка выбирает обычный путь llama.cpp (режим `gpu`), для плотной модели больше VRAM — раскладку `--fit` (режим `fit`). Драфт ищется трёх видов: встроенный MTP-слой, файл MTP-головы `mtp-<модель>*.gguf` рядом с моделью или маленькая модель того же семейства с тем же токенизатором.

## Использование

```
python aic.py build --model models/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf   # один раз, ~2 минуты
python aic.py run   --model models/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf   # сервер llama.cpp-совместимый, порт 8080
python aic.py show  --model models/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf   # что выбрано и почему (все пробы)
```

`run` поднимает `llama-server` нашего форка: в нём есть `/completion` и OpenAI-совместимый `/v1/chat/completions`. Если сменились модель, железо, драйвер или движок, `run` попросит пересобрать.

## Подготовка (один раз)

1. CUDA Toolkit 13.5, Visual Studio 2022 (MSVC), CMake, Ninja, Python 3.12.
2. Эталон и исходники llama.cpp: официальная сборка b12189 в `third_party/llama-bin`, исходники коммита `80d2c0c2c` в `third_party/llama.cpp`.
3. Форк: `git -C third_party/llama.cpp worktree add ../llama-fork -b engine 80d2c0c2c`, затем `git -C third_party/llama-fork am ../../engine/patches/*.patch`.
4. Сборка: `tools\build_llamacpp.bat third_party\llama-fork build\llama-fork`, эталона из исходников — `tools\build_llamacpp.bat` (по умолчанию `third_party\llama.cpp` → `build\llama-ref`).
5. Модели в `models/` (UD-Q4_K_M от unsloth, MTP-голова от ggml-org) и wikitext-2 в `data/`. SHA256 — в BENCHMARKS.md.

## Воспроизведение замеров

| Команда | Что меряет |
|---|---|
| `tools\hwbench\build.bat && tools\hwbench\hwbench.exe` | потолки железа: RAM, VRAM, PCIe, zero-copy, запуск ядер |
| `python bench/sweep_llamacpp.py <stage>` | перебор настроек llama.cpp (эталон) → `bench/sweep_results.csv` |
| `python bench/milestone.py --contenders bench/contenders/milestone1_Qwen3.6.json --rounds 3` | итоговое сравнение вехи 1: наш движок (сохранённая сборка из `cache/`; копии профилей — в `bench/profiles/`) против трёх лучших конфигураций llama.cpp |
| `python bench/milestone.py --model models/GLM-4.7-Flash-MTP-Q4_K_M.gguf --rounds 2` | то же для второй модели |
| `python bench/milestone.py --contenders bench/contenders/qwen_split.json --rounds 1` | разделение выигрыша: вклад движка и вклад подбора |
| `python bench/strata_bench.py --model <gguf> --url http://127.0.0.1:8091` | замер запущенного сервера Strata на тех же промптах |
| `bash bench/quality.sh` | perplexity и KL-дивергенция против эталонных логитов |
| `python bench/compare_outputs.py <ref.jsonl> <cand.jsonl>` | где расходятся жадные ответы |

## Файлы

| Путь | Что там |
|---|---|
| `aic.py` | «компилятор»: build / run / show |
| `engine/patches/` | изменения в llama.cpp (наш движок), применяются к коммиту `80d2c0c2c` |
| `DECISIONS.md` | решения, варианты, неудачные эксперименты |
| `BENCHMARKS.md` | все замеры с условиями |
| `PROGRESS.md` | короткий журнал для владельца |
| `docs/concepts.md` | понятия GPU и LLM простыми словами |
| `docs/survey.md` | обзор похожих проектов |
| `bench/` | стенд: сервер-бенчмарк, промпты, перебор, проверка качества |
| `tools/` | микробенчмарки (`hwbench`, `vmmtest`, `pingpong`), трассы экспертов, симулятор кэша, разбор профилей Nsight |
| `third_party/`, `models/`, `data/`, `build/`, `cache/`, `bench/logs/` | не в git |
