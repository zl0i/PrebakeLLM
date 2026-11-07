"""Milestone 1 measurement: our engine (saved aic build) vs the best llama.cpp settings.

  python bench/milestone.py [--rounds 3]

Each round runs every contender once over all prompts of bench/prompts.json
(8 short + 1 long ~17K tokens), contenders interleaved so that drift of the
machine (temperature, background load) affects all of them alike. Every run
is a fresh server process via bench/serverbench.py; raw logs in bench/logs/.
Prints a markdown table (median over rounds of the per-run mean speed).
"""
import argparse
import json
import os
import pathlib
import statistics
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import aic  # noqa: E402

MODEL = ROOT / "models" / "Qwen3.6-35B-A3B-UD-Q4_K_M.gguf"
MTP = "models/mtp-Qwen3.6-35B-A3B-Q4_0.gguf"
LLAMA = ["-ngl", "99", "-cmoe", "-lm", "none", "-t", "8"]
CONTENDERS = {
    # the three best llama.cpp configurations from bench/sweep_results.csv
    "llama.cpp cache6500+mtp2": (None, LLAMA + ["--moe-cache-mib", "6500", "-md", MTP, "--spec-type", "draft-mtp", "--spec-draft-n-max", "2"], {}),
    "llama.cpp cache6500+mtp2 t6": (None, LLAMA[:-1] + ["6", "--moe-cache-mib", "6500", "-md", MTP, "--spec-type", "draft-mtp", "--spec-draft-n-max", "2"], {}),
    "llama.cpp cache6500+mtp3": (None, LLAMA + ["--moe-cache-mib", "6500", "-md", MTP, "--spec-type", "draft-mtp", "--spec-draft-n-max", "3"], {}),
}


def ours():
    prof = aic.load_profile(MODEL)
    cfg = prof["config"]
    args = aic.server_args(MODEL, cfg)
    # serverbench adds -m and -c itself
    i = args.index("-m"); del args[i:i + 2]
    i = args.index("-c"); del args[i:i + 2]
    i = args.index("-np"); del args[i:i + 2]
    env = {"LLAMA_MOE_ZC_MIB": str(cfg["cache_mib"]), "LLAMA_MOE_ZC_THREADS": str(cfg.get("cpu_threads", 8))}
    return str(aic.ENGINE), args, env


def run(tag, server, args, env):
    cmd = [sys.executable, str(ROOT / "bench" / "serverbench.py"), "--tag", tag, "--reps", "1"]
    if server:
        cmd += ["--server", server]
    cmd += ["--"] + args
    e = dict(os.environ)
    e["PATH"] = aic.CUDA_BIN + os.pathsep + e["PATH"]
    e["PYTHONIOENCODING"] = "utf-8"
    e.update(env)
    p = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace", env=e)
    log = next(l.split("log: ", 1)[1].strip() for l in p.stdout.splitlines() if l.startswith("log: "))
    short, long_, pp_long, acc = [], [], None, []
    for line in open(log, encoding="utf-8"):
        r = json.loads(line)
        if r["kind"] != "run":
            continue
        t = r["timings"]
        (long_ if r["id"] == "long16k" else short).append(t["predicted_per_second"])
        if r["id"] == "long16k":
            pp_long = t["prompt_per_second"]
        if t.get("draft_n"):
            acc.append(t["draft_n_accepted"] / t["draft_n"])
    return {"short": statistics.mean(short), "long": long_[0], "pp_long": pp_long,
            "acc": statistics.mean(acc) if acc else None, "log": pathlib.Path(log).name}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=3)
    a = ap.parse_args()
    contenders = dict(CONTENDERS)
    contenders["ours (aic build)"] = ours()
    results = {k: [] for k in contenders}
    for rnd in range(a.rounds):
        for name, (server, args, env) in contenders.items():
            tag = "ms_" + name.replace(" ", "_").replace("(", "").replace(")", "").replace("+", "_").replace(".", "")
            r = run(f"{tag}_r{rnd}", server, args, env)
            results[name].append(r)
            print(f"round {rnd} {name:30s} short {r['short']:6.1f}  long {r['long']:6.1f}  pp_long {r['pp_long']:6.1f}  {r['log']}", flush=True)
    print("\n| Contender | short (mean of 8 prompts), tok/s | long 17K, tok/s | prompt 17K, tok/s | runs |")
    print("|---|---|---|---|---|")
    for name, rs in results.items():
        med = lambda k: statistics.median(r[k] for r in rs)
        print(f"| {name} | {med('short'):.1f} | {med('long'):.1f} | {med('pp_long'):.0f} | "
              + ", ".join(f"{r['short']:.1f}/{r['long']:.1f}" for r in rs) + " |")
    out = ROOT / "bench" / "milestone_results.json"
    out.write_text(json.dumps(results, indent=1), encoding="utf-8")
    print(f"\nraw: {out}")


if __name__ == "__main__":
    main()
