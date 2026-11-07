"""Builds the benchmark and calibration prompts for any model, using the model's
own chat template (read from its GGUF by llama-server's /apply-template).

  python bench/make_model_prompts.py --model models/GLM-4.7-Flash-MTP-Q4_K_M.gguf

Writes bench/prompts/<model>.json (warmup, short0..7, long16k: same texts as
bench/prompts.json) and bench/prompts/<model>.calib.json (calibration texts as
bench/calib_prompts.json). Thinking is disabled (enable_thinking=false) so the
models answer directly and outputs are comparable between engines.

bench/prompts.json and bench/calib_prompts.json stay the files of the first
model (Qwen3.6-35B-A3B): they were built with a hand-written Qwen template,
which is identical to what /apply-template gives for that model.
"""
import argparse
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "bench"))
import serverbench as sb  # noqa: E402

DOCS = ROOT / "third_party" / "llama.cpp" / "docs"

SHORT = [
    "Explain in simple terms how a hash map works and when it is a bad choice.",
    "Write a C++ function that parses an IPv4 address string into a uint32_t, with error handling.",
    "Кратко перескажи сюжет романа «Мастер и Маргарита» и назови три главные темы.",
    "List the planets of the solar system with one interesting fact about each.",
    "Solve step by step: a train leaves at 9:40 at 72 km/h, another at 10:15 at 90 km/h from the same station in the same direction. When does the second catch up?",
    "Write a Python script that walks a directory tree and reports the 10 largest files, with argparse options.",
    "Напиши вежливое письмо коллеге с просьбой перенести совещание на следующую неделю, объясни причину.",
    "Compare TCP and UDP: guarantees, typical use cases, and what happens on packet loss.",
]
WARMUP = "Write a short story about a lighthouse keeper who finds a message in a bottle."
CALIB = [
    "Write a Rust function that merges two sorted vectors into one sorted vector, and explain its complexity.",
    "Объясни, как работает ипотека с аннуитетными платежами, на простом примере с цифрами.",
    "What are the main causes of the French Revolution? Answer in a structured way.",
    "Write a SQL query that finds the top 5 customers by total order value in the last 90 days, and explain it.",
    "Give me a weekly workout plan for a beginner who can train 3 times a week at home.",
    "Translate into English and explain the idioms: «Без труда не вытащишь и рыбку из пруда», «Семь раз отмерь, один раз отрежь».",
    "Explain the difference between processes and threads in operating systems, with examples.",
    "Write a JavaScript debounce function and show how to use it with an input field.",
    "Напиши короткий рассказ о роботе, который учится готовить борщ.",
    "Derive the formula for the sum of a geometric series and give two applications.",
]


def long_doc() -> str:
    parts, total = [], 0
    for f in sorted(DOCS.rglob("*.md")):
        t = f.read_text(encoding="utf-8", errors="replace")
        parts.append(f"\n\n===== {f.relative_to(DOCS).as_posix()} =====\n{t}")
        total += len(t)
        if total > 64_000:
            break
    return "".join(parts)[:64_000]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    a = ap.parse_args()
    model = pathlib.Path(a.model).resolve()

    wiki = (ROOT / "data" / "wikitext-2-raw" / "wiki.valid.raw").read_text(encoding="utf-8")
    chunks = [c for c in wiki.split("\n = ") if len(c) > 3000][:4]
    calib_texts = CALIB + [f"Continue this encyclopedia article in the same style:\n\n{c[:3000]}" for c in chunks]

    port = 8096
    log = open(ROOT / "bench" / "logs" / "make_model_prompts.server.log", "w", encoding="utf-8")
    proc = sb.start_server([str(sb.DEFAULT_SERVER), "-m", str(model), "-ngl", "0", "-c", "512", "--port", str(port), "--no-webui"], log)
    base = f"http://127.0.0.1:{port}"
    try:
        sb.wait_ready(base, proc)

        def chat(text: str) -> str:
            r = sb.post(base + "/apply-template", {"messages": [{"role": "user", "content": text}],
                                                   "chat_template_kwargs": {"enable_thinking": False}})
            return r["prompt"]

        prompts = [{"id": "warmup", "prompt": chat(WARMUP)}]
        prompts += [{"id": f"short{i}", "prompt": chat(t)} for i, t in enumerate(SHORT)]
        prompts.append({"id": "long16k", "prompt": chat("Below is documentation of a software project.\n" + long_doc() +
                                                        "\n\nSummarize the main build options described above as a bulleted list.")})
        calib = [{"id": f"calib{i}", "prompt": chat(t)} for i, t in enumerate(calib_texts)]
    finally:
        sb.stop_server(proc)
        log.close()

    out = ROOT / "bench" / "prompts"
    out.mkdir(exist_ok=True)
    (out / f"{model.stem}.json").write_text(json.dumps(prompts, ensure_ascii=False, indent=1), encoding="utf-8")
    (out / f"{model.stem}.calib.json").write_text(json.dumps(calib, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"wrote bench/prompts/{model.stem}.json and .calib.json; example:\n{prompts[1]['prompt'][:300]!r}")


if __name__ == "__main__":
    main()
