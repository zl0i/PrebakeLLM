"""Compares generated token ids of two serverbench logs (greedy runs).

  python bench/compare_outputs.py <reference.jsonl> <candidate.jsonl> [...more candidates]

For each prompt, reports how many of the generated tokens match the reference
and where the first divergence is. Greedy decoding is deterministic in math,
but GPU/CPU floating point sums can differ in the last bits between kernels
or batch sizes; when two logits are nearly tied, the argmax may flip and the
texts diverge from that point. So "first divergence" plus a look at the two
continuations is the honest way to judge correctness.
"""
import json
import pathlib
import sys

LOGS = pathlib.Path(__file__).resolve().parent / "logs"


def load(path: str) -> dict:
    p = pathlib.Path(path)
    if not p.exists():
        p = LOGS / path
    runs = {}
    for line in open(p, encoding="utf-8"):
        r = json.loads(line)
        if r["kind"] == "run" and r["rep"] == 0:
            runs[r["id"]] = r
    return runs


def main() -> None:
    ref = load(sys.argv[1])
    for cand_path in sys.argv[2:]:
        cand = load(cand_path)
        print(f"== {pathlib.Path(cand_path).name} vs {pathlib.Path(sys.argv[1]).name}")
        n_same = 0
        for pid, r in ref.items():
            if pid not in cand:
                continue
            a, b = r["tokens"], cand[pid]["tokens"]
            div = next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), None)
            if div is None and len(a) == len(b):
                n_same += 1
                print(f"  {pid:8s} identical ({len(a)} tokens)")
            else:
                note = ""
                top = r.get("top")
                if top and div is not None and div < len(top) and len(top[div]) >= 2:
                    # how close the reference itself was to choosing differently at that position
                    (t1, lp1), (t2, lp2) = top[div][0], top[div][1]
                    note = f"  ref top1-top2 logprob gap = {lp1 - lp2:.3f}"
                    if b[div] == t2:
                        note += " (candidate took the ref's #2 token)"
                print(f"  {pid:8s} diverges at token {div} of {len(a)}{note}")
        print(f"  identical: {n_same}/{len(ref)}")


if __name__ == "__main__":
    main()
