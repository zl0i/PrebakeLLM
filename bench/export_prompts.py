"""Writes prompts as one-per-line text (newlines escaped) for C++ tools.

  python bench/export_prompts.py <prompts.json> <out.txt>
The order is kept; the warmup prompt goes first.
"""
import json
import sys

prompts = json.load(open(sys.argv[1], encoding="utf-8"))
with open(sys.argv[2], "w", encoding="utf-8", newline="\n") as f:
    for p in prompts:
        f.write(p["prompt"].replace("\\", "\\\\").replace("\r", "").replace("\n", "\\n") + "\n")
print(f"{len(prompts)} prompts -> {sys.argv[2]}")
