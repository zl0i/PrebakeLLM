"""One command for the three-engine comparison (Strata, llama.cpp, our engine) on one model.

  python bench/strata_compare.py qwen38 [--rounds 2] [--tail-skip 0] [--skip-strata] [--skip-llama-ours] [--dry-run]
  python bench/strata_compare.py qwen36 ...

Steps:
  1. Strata: `strata launch` (terminal build in third_party/strata/ctl2) with the settings of the first
     comparison, the server on port 8091; bench/strata_bench.py on our prompts; `strata launch --stop`.
     --tail-skip 0 passes STRATA_ROUTE_TAIL_SKIP=0 (via `--set`): Strata then computes every routed expert,
     as llama.cpp and our engine do. Its default (7) skips missed experts routed at rank >= 7 and changes
     the output slightly (Strata's docs/DETAILS.md).
  2. llama.cpp and our engine: bench/milestone.py with bench/contenders/<model>.json plus our saved aic build
     (cache/<model>__<hw>.json; run `python aic.py build --model ...` first if it is missing).
  3. Prints the summary of both (median per contender; Strata: mean of short prompts and the long one per round).

Needs nothing else running on the GPU. Takes ~25 min per model with 2 rounds.
"""
import argparse
import json
import pathlib
import statistics
import subprocess
import sys
import time
import urllib.request

ROOT = pathlib.Path(__file__).resolve().parent.parent
STRATA = ROOT / "third_party" / "strata" / "ctl2" / "bin" / "strata.cmd"
DATA = ROOT / "third_party" / "strata" / "data"
PORT = 8091

MODELS = {
    # model file of our engine / llama.cpp, Strata launch flags (the same files; Strata checks their SHA-256)
    "qwen38": {
        "model": ROOT / "models" / "qwen38" / "Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00002.gguf",
        "strata": ["--family", "qwen", "--model", "q2_0", "--model-path", str(ROOT / "models" / "qwen38"), "--drafting", "mtp"],
    },
    "qwen36": {
        "model": ROOT / "models" / "qwen36mtp" / "Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf",
        "strata": ["--family", "qwen36", "--model", "ud-q4_k_xl",
                   "--model-path", str(ROOT / "models" / "qwen36mtp" / "Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf")],
    },
}


def sh(cmd, dry, **kw):
    print("$ " + " ".join(str(c) for c in cmd), flush=True)
    if dry:
        return subprocess.CompletedProcess(cmd, 0, "", "")
    return subprocess.run([str(c) for c in cmd], cwd=ROOT, text=True, encoding="utf-8", errors="replace", **kw)


def strata_summary(log):
    rs = [json.loads(l) for l in open(log, encoding="utf-8")][1:]
    for rep in sorted({r["rep"] for r in rs}):
        sh_ = [r["timings"] for r in rs if r["rep"] == rep and r["id"] != "long16k"]
        lo = [r["timings"] for r in rs if r["rep"] == rep and r["id"] == "long16k"][0]
        dn = sum(t.get("draft_n", 0) for t in sh_)
        da = sum(t.get("draft_n_accepted", 0) for t in sh_)
        print(f"  Strata round {rep}: short {statistics.mean(t['predicted_per_second'] for t in sh_):.1f}  long {lo['predicted_per_second']:.1f}"
              f"  prompt 17K {lo['prompt_per_second']:.0f}  drafts accepted (short) {da}/{dn}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("which", choices=list(MODELS))
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--tail-skip", default="", help="STRATA_ROUTE_TAIL_SKIP for Strata (0 = compute every expert)")
    ap.add_argument("--skip-strata", action="store_true")
    ap.add_argument("--skip-llama-ours", action="store_true")
    ap.add_argument("--dry-run", action="store_true", help="print the commands only")
    a = ap.parse_args()
    m = MODELS[a.which]
    tag = f"strata_{a.which}" + (f"_tail{a.tail_skip}" if a.tail_skip != "" else "")
    strata_log = None

    if not a.skip_strata:
        extra = ["--set", f"STRATA_ROUTE_TAIL_SKIP={a.tail_skip}"] if a.tail_skip != "" else []
        sh(["cmd", "/c", STRATA, "launch", "--stop"], a.dry_run, capture_output=True)
        r = sh(["cmd", "/c", STRATA, "launch"] + m["strata"] + extra +
               ["--data-dir", DATA, "--context", "32k", "--vision", "off", "--tune", "skip", "--port", PORT,
                "--no-api-key", "--no-tui", "--yes", "--restart"], a.dry_run, capture_output=True)
        if r.returncode:
            sys.exit(f"strata launch failed:\n{r.stdout[-2000:]}\n{r.stderr[-2000:]}")
        if not a.dry_run:
            for _ in range(600):
                try:
                    if json.load(urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=5)).get("loaded"):
                        break
                except Exception:
                    pass
                time.sleep(1)
        r = sh([sys.executable, ROOT / "bench" / "strata_bench.py", "--model", m["model"], "--url", f"http://127.0.0.1:{PORT}",
                "--tag", tag, "--rounds", a.rounds], a.dry_run, capture_output=True)
        print(r.stdout[-3000:])
        strata_log = next((l.split("log: ", 1)[1].strip() for l in r.stdout.splitlines() if l.startswith("log: ")), None)
        sh(["cmd", "/c", STRATA, "launch", "--stop"], a.dry_run, capture_output=True)
        time.sleep(0 if a.dry_run else 5)

    if not a.skip_llama_ours:
        r = sh([sys.executable, ROOT / "bench" / "milestone.py", "--model", m["model"], "--rounds", a.rounds], a.dry_run,
               capture_output=True)
        print(r.stdout[-2500:])

    if strata_log:
        print(f"\nStrata ({strata_log}):")
        strata_summary(strata_log)


if __name__ == "__main__":
    main()
