"""Builds bench/prompts.json: fixed prompts for every speed/correctness run.

Short prompts check plain generation speed. The long prompt (~16K tokens of
llama.cpp docs at the pinned commit) checks generation speed with a large
context, where attention over the KV cache starts to cost something.

Run once; the output is committed so runs stay comparable even if
third_party/ changes.
"""
import json
import pathlib

ROOT = pathlib.Path(__file__).resolve().parent.parent
DOCS = ROOT / "third_party" / "llama.cpp" / "docs"


def chat(user: str) -> str:
    # Qwen3.6 chat template with thinking disabled (empty <think> block),
    # so the model answers directly and outputs are easy to compare.
    return (f"<|im_start|>user\n{user}<|im_end|>\n"
            f"<|im_start|>assistant\n<think>\n\n</think>\n\n")


short = [
    "Explain in simple terms how a hash map works and when it is a bad choice.",
    "Write a C++ function that parses an IPv4 address string into a uint32_t, with error handling.",
    "Кратко перескажи сюжет романа «Мастер и Маргарита» и назови три главные темы.",
    "List the planets of the solar system with one interesting fact about each.",
]

# Long context: concatenate docs until ~64K characters (~16K tokens).
parts, total = [], 0
for f in sorted(DOCS.rglob("*.md")):
    t = f.read_text(encoding="utf-8", errors="replace")
    parts.append(f"\n\n===== {f.relative_to(DOCS).as_posix()} =====\n{t}")
    total += len(t)
    if total > 64_000:
        break
long_doc = "".join(parts)[:64_000]

prompts = [{"id": f"short{i}", "prompt": chat(p)} for i, p in enumerate(short)]
prompts.append({"id": "long16k", "prompt": chat(
    "Below is documentation of a software project.\n" + long_doc +
    "\n\nSummarize the main build options described above as a bulleted list.")})

out = ROOT / "bench" / "prompts.json"
out.write_text(json.dumps(prompts, ensure_ascii=False, indent=1), encoding="utf-8")
print(f"wrote {out}: {len(prompts)} prompts, long prompt {len(prompts[-1]['prompt'])} chars")
