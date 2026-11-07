"""Calibration prompts: deliberately different from bench/prompts.json.

Used to learn usage statistics (which experts are hot) on this machine's
"typical" workload. Evaluation is then done on bench/prompts.json, so we never
grade a learned hot set on the data it was learned from.
"""
import json
import pathlib

ROOT = pathlib.Path(__file__).resolve().parent.parent


def chat(user: str) -> str:
    return f"<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"


user_prompts = [
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

wiki = (ROOT / "data" / "wikitext-2-raw" / "wiki.valid.raw").read_text(encoding="utf-8")
chunks = [c for c in wiki.split("\n = ") if len(c) > 3000][:4]
wiki_prompts = [f"Continue this encyclopedia article in the same style:\n\n{c[:3000]}" for c in chunks]

prompts = [{"id": f"calib{i}", "prompt": chat(p)} for i, p in enumerate(user_prompts + wiki_prompts)]
out = ROOT / "bench" / "calib_prompts.json"
out.write_text(json.dumps(prompts, ensure_ascii=False, indent=1), encoding="utf-8")
print(f"wrote {out}: {len(prompts)} prompts")
