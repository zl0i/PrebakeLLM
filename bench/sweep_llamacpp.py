"""Search for the best llama.cpp settings on this machine (the baseline to beat).

Each stage is a list of configs; each config is a full llama-server run via
serverbench.py. Results (median generation tok/s per prompt) are appended to
bench/sweep_results.csv, which is committed, so the baseline is reproducible.

  python bench/sweep_llamacpp.py placement
  python bench/sweep_llamacpp.py threads --base "-ngl 99 -ncmoe 25 -lm none"

Logical CPU layout on i7-13700KF under Windows: 0..15 = 8 P-cores x 2
hyperthreads (0,1 = core 0 ...), 16..23 = 8 E-cores. -C takes a hex mask.
"""
import argparse
import csv
import datetime
import json
import pathlib
import shlex
import statistics
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
MTP = "models/mtp-Qwen3.6-35B-A3B-Q4_0.gguf"
DFLASH = "models/dflash-Qwen3.6-35B-A3B-Q8_0.gguf"

STAGES = {
    # where expert weights live: auto-fit vs explicit layer count, mmap vs plain load
    "placement": [
        ("fit", ""),
        ("fit_lmnone", "-lm none"),
        ("ncmoe24_lmnone", "-ngl 99 -ncmoe 24 -lm none"),
        ("ncmoe25_lmnone", "-ngl 99 -ncmoe 25 -lm none"),
        ("ncmoe26_lmnone", "-ngl 99 -ncmoe 26 -lm none"),
    ],
    # CPU threads for the experts that stay in RAM
    "threads": [
        ("t16", "-t 16"),
        ("t8_pin_p1", "-t 8 -C 0x5555 --cpu-strict 1"),
        ("t8", "-t 8"),
        ("t12", "-t 12"),
        ("t16_pin_pall", "-t 16 -C 0xFFFF --cpu-strict 1"),
        ("t24", "-t 24"),
    ],
    # llama.cpp's own VRAM LRU cache of experts (all experts in RAM, hot ones cached)
    "moecache": [
        ("cmoe_cache5000", "-ngl 99 -cmoe --moe-cache-mib 5000 -lm none"),
        ("cmoe_cache6500", "-ngl 99 -cmoe --moe-cache-mib 6500 -lm none"),
        ("cmoe_cache7500", "-ngl 99 -cmoe --moe-cache-mib 7500 -lm none"),
    ],
    # speculative decoding: exact for greedy, only speed changes
    "spec": [
        ("mtp_n1", f"-md {MTP} --spec-type draft-mtp --spec-draft-n-max 1"),
        ("mtp_n2", f"-md {MTP} --spec-type draft-mtp --spec-draft-n-max 2"),
        ("mtp_n3", f"-md {MTP} --spec-type draft-mtp --spec-draft-n-max 3"),
        ("mtp_n4", f"-md {MTP} --spec-type draft-mtp --spec-draft-n-max 4"),
        ("dflash_n8", f"-md {DFLASH} --spec-type draft-dflash --spec-draft-n-max 8"),
        ("dflash_n15", f"-md {DFLASH} --spec-type draft-dflash --spec-draft-n-max 15"),
    ],
    # combination of the two winners: VRAM expert cache + MTP (MTP needs ~1 GB VRAM, so smaller cache)
    "combo": [
        ("cache5500_mtp2", f"-ngl 99 -cmoe --moe-cache-mib 5500 -lm none -md {MTP} --spec-type draft-mtp --spec-draft-n-max 2"),
        ("cache6000_mtp2", f"-ngl 99 -cmoe --moe-cache-mib 6000 -lm none -md {MTP} --spec-type draft-mtp --spec-draft-n-max 2"),
        ("cache6500_mtp2", f"-ngl 99 -cmoe --moe-cache-mib 6500 -lm none -md {MTP} --spec-type draft-mtp --spec-draft-n-max 2"),
        ("cache6000_mtp3", f"-ngl 99 -cmoe --moe-cache-mib 6000 -lm none -md {MTP} --spec-type draft-mtp --spec-draft-n-max 3"),
        ("cache7500_nomtp", "-ngl 99 -cmoe --moe-cache-mib 7500 -lm none"),
        ("ncmoe25_mtp2", f"-ngl 99 -ncmoe 25 -lm none -md {MTP} --spec-type draft-mtp --spec-draft-n-max 2"),
    ],
    # around the best combo: a bit more cache (smaller compute buffer via -ub), threads
    "combo2": [
        ("cache7000_mtp2", f"-ngl 99 -cmoe --moe-cache-mib 7000 -lm none -md {MTP} --spec-type draft-mtp --spec-draft-n-max 2"),
        ("cache7000_mtp2_ub256", f"-ngl 99 -cmoe --moe-cache-mib 7000 -lm none -ub 256 -md {MTP} --spec-type draft-mtp --spec-draft-n-max 2"),
        ("cache7400_mtp2_ub128", f"-ngl 99 -cmoe --moe-cache-mib 7400 -lm none -ub 128 -md {MTP} --spec-type draft-mtp --spec-draft-n-max 2"),
        ("cache6500_mtp3", f"-ngl 99 -cmoe --moe-cache-mib 6500 -lm none -md {MTP} --spec-type draft-mtp --spec-draft-n-max 3"),
        ("cache6500_mtp2_t16pin", f"-ngl 99 -cmoe --moe-cache-mib 6500 -lm none -md {MTP} --spec-type draft-mtp --spec-draft-n-max 2 -t 16 -C 0xFFFF --cpu-strict 1"),
        ("cache6500_mtp2_t6", f"-ngl 99 -cmoe --moe-cache-mib 6500 -lm none -md {MTP} --spec-type draft-mtp --spec-draft-n-max 2 -t 6"),
    ],
}


def run(tag: str, args: str, only: str, reps: int) -> dict:
    cmd = [sys.executable, str(ROOT / "bench" / "serverbench.py"), "--tag", tag, "--reps", str(reps),
           "--only", only, "--"] + shlex.split(args)
    p = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace")
    print(p.stdout[-1500:], p.stderr[-1500:] if p.returncode else "", flush=True)
    log = next((l.split("log: ", 1)[1].strip() for l in p.stdout.splitlines() if l.startswith("log: ")), None)
    if not log:
        return {"error": f"exit {p.returncode}"}
    tg, pp, acc, header = {}, {}, {}, {}
    for line in open(log, encoding="utf-8"):
        r = json.loads(line)
        if r["kind"] == "header":
            header = r
            continue
        t = r["timings"]
        tg.setdefault(r["id"], []).append(t["predicted_per_second"])
        pp.setdefault(r["id"], []).append(t["prompt_per_second"])
        if t.get("draft_n"):
            acc.setdefault(r["id"], []).append(t["draft_n_accepted"] / t["draft_n"])
    return {
        "tg": {k: round(statistics.median(v), 2) for k, v in tg.items()},
        "pp": {k: round(statistics.median(v), 1) for k, v in pp.items()},
        "acc": {k: round(statistics.median(v), 3) for k, v in acc.items()},
        "vram_free_loaded": header.get("vram_free_loaded_mib"), "load_s": round(header.get("load_s", 0), 1),
        "log": pathlib.Path(log).name,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=list(STAGES))
    ap.add_argument("--base", default="", help="args prepended to every config of the stage")
    ap.add_argument("--only", default="short0,short1")
    ap.add_argument("--reps", type=int, default=2)
    a = ap.parse_args()

    out = ROOT / "bench" / "sweep_results.csv"
    new = not out.exists()
    with open(out, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["time", "stage", "tag", "args", "prompts", "tg_median_by_prompt", "tg_mean",
                        "pp_by_prompt", "draft_accept", "vram_free_loaded_mib", "load_s", "log"])
        for tag, args in STAGES[a.stage]:
            full = (a.base + " " + args).strip()
            r = run(tag, full, a.only, a.reps)
            tg_mean = round(statistics.mean(r["tg"].values()), 2) if r.get("tg") else ""
            w.writerow([datetime.datetime.now().isoformat(timespec="seconds"), a.stage, tag, full, a.only,
                        json.dumps(r.get("tg")), tg_mean, json.dumps(r.get("pp")), json.dumps(r.get("acc")),
                        r.get("vram_free_loaded"), r.get("load_s"), r.get("log", r.get("error"))])
            f.flush()
            print(f"==> {tag:18s} tg_mean={tg_mean}  {r.get('tg')}  acc={r.get('acc')}", flush=True)


if __name__ == "__main__":
    main()
