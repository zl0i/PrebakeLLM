"""Head-to-head measurement: our engine (saved aic build) vs other engines' best settings.

  python bench/milestone.py [--model models/X.gguf] [--rounds 3] [--only name1,name2]

Contenders come from bench/contenders/<model>.json:
  [{"name": "...", "server": "path or empty for official llama.cpp", "args": [...], "env": {...}}, ...]
plus "ours (aic build)" taken from the saved aic build of that model.

Each round runs every contender once over all prompts of the model (8 short +
1 long ~17K tokens), contenders interleaved so that drift of the machine affects
all alike. Every run is a fresh server process via bench/serverbench.py; raw
logs in bench/logs/. Prints a markdown table: median over rounds of the mean
speed over the 8 short prompts, the long prompt, its prompt processing speed,
draft acceptance, CPU busy share and GPU power during generation.

Compare engines only within one column: the columns differ in how well the
draft guesses (e.g. ~77% on short chat prompts vs ~90% on the long summary).
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

DEFAULT_MODEL = ROOT / "models" / "Qwen3.6-35B-A3B-UD-Q4_K_M.gguf"


def ours(model, cfg=None, name="ours (aic build)"):
    """our engine with the saved aic build of the model, or with an explicit config (e.g. another model's choice)"""
    if cfg is None:
        cfg = aic.load_profile(model)["config"]
    args = aic.server_args(model, cfg)
    for flag in ("-m", "-c", "-np"):  # serverbench adds these itself
        i = args.index(flag)
        del args[i:i + 2]
    env = {}
    if cfg.get("mode", "moe-zc") == "moe-zc":
        env = {"LLAMA_MOE_ZC_MIB": str(cfg["cache_mib"]), "LLAMA_MOE_ZC_THREADS": str(cfg.get("cpu_threads", 8))}
    return {"name": name, "server": str(aic.ENGINE), "args": args, "env": env}


def run(tag, model, c, stream=False):
    cmd = [sys.executable, str(ROOT / "bench" / "serverbench.py"), "--tag", tag, "--reps", "1", "--model", str(model)]
    if stream:
        cmd += ["--stream"]
    if c.get("server"):
        cmd += ["--server", c["server"]]
    if c.get("plain"):
        cmd += ["--plain"]
    cmd += ["--"] + c["args"]
    e = dict(os.environ)
    e["PATH"] = aic.CUDA_BIN + os.pathsep + e["PATH"]
    e["PYTHONIOENCODING"] = "utf-8"
    e.update(c.get("env", {}))
    p = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace", env=e)
    log = next((l.split("log: ", 1)[1].strip() for l in p.stdout.splitlines() if l.startswith("log: ")), None)
    if not log:
        return {"error": (p.stdout + p.stderr)[-800:]}
    short, long_, pp_long, acc, cpu, gpu_w = [], None, None, [], [], []
    short_srv, long_srv = [], None
    for line in open(log, encoding="utf-8"):
        r = json.loads(line)
        if r["kind"] != "run":
            continue
        t = r["timings"]
        cl = r.get("client")  # streaming run: speeds by the client's clock (server timings kept as *_srv)
        tg = cl["tg"] if cl else t["predicted_per_second"]
        if r["id"] == "long16k":
            long_, long_srv = tg, t["predicted_per_second"]
            pp_long = cl["pp"] if cl else t["prompt_per_second"]
        else:
            short.append(tg)
            short_srv.append(t["predicted_per_second"])
            if r.get("load"):
                cpu.append(r["load"]["cpu_busy"] or 0)
                gpu_w.append(r["load"]["gpu_w"] or 0)
        if t.get("draft_n"):
            acc.append(t["draft_n_accepted"] / t["draft_n"])
    return {"short": statistics.mean(short), "long": long_, "pp_long": pp_long,
            "short_srv": statistics.mean(short_srv), "long_srv": long_srv, "client_timed": stream,
            "acc": statistics.mean(acc) if acc else None,
            "cpu_busy": statistics.mean(cpu) if cpu else None, "gpu_w": statistics.mean(gpu_w) if gpu_w else None,
            "log": pathlib.Path(log).name}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(DEFAULT_MODEL))
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--only", default="", help="comma-separated contender names")
    ap.add_argument("--contenders", default="", help="contenders file (default: bench/contenders/<model>.json)")
    ap.add_argument("--stream", action="store_true", help="client-timed streaming requests (speeds by the client's clock)")
    ap.add_argument("--no-ours", action="store_true", help="only the contenders from the file, not the saved aic build")
    ap.add_argument("--out", default="", help="results file (default: bench/milestone_results_<model>.json)")
    a = ap.parse_args()
    model = pathlib.Path(a.model).resolve()
    cfile = pathlib.Path(a.contenders) if a.contenders else ROOT / "bench" / "contenders" / f"{model.stem}.json"
    contenders = []
    for c in json.loads(cfile.read_text(encoding="utf-8")):
        # {"name": ..., "ours_cfg": {...}}: our engine with a fixed config (e.g. transplanted from another model)
        contenders.append(ours(model, dict(c["ours_cfg"]), c["name"]) if "ours_cfg" in c else c)
    if not a.no_ours:
        contenders.append(ours(model))
    if a.only:
        keep = set(a.only.split(","))
        contenders = [c for c in contenders if c["name"] in keep]
    results = {c["name"]: [] for c in contenders}
    for rnd in range(a.rounds):
        for c in contenders:
            tag = "ms_" + model.stem[:12] + "_" + "".join(ch if ch.isalnum() else "_" for ch in c["name"]) + f"_r{rnd}"
            r = run(tag, model, c, a.stream)
            if "error" in r:
                print(f"round {rnd} {c['name']}: FAILED {r['error']}", flush=True)
                continue
            results[c["name"]].append(r)
            print(f"round {rnd} {c['name']:34s} short {r['short']:6.1f}  long {r['long']:6.1f}  pp_long {r['pp_long']:6.1f}"
                  f"  acc {r['acc'] or 0:.2f}  CPU {100 * (r['cpu_busy'] or 0):4.1f}%  GPU {r['gpu_w'] or 0:5.1f} W  {r['log']}", flush=True)
    print(f"\nmodel: {model.name}")
    print("| Contender | short, tok/s | long 17K, tok/s | prompt 17K, tok/s | draft accepted | CPU busy | GPU power | GPU J/token (short) | short/long per round |")
    print("|---|---|---|---|---|---|---|---|---|")
    for name, rs in results.items():
        if not rs:
            continue
        med = lambda k: statistics.median(r[k] for r in rs if r[k] is not None) if any(r[k] is not None for r in rs) else float("nan")
        print(f"| {name} | {med('short'):.1f} | {med('long'):.1f} | {med('pp_long'):.0f} | {100 * med('acc'):.0f}% | "
              f"{100 * med('cpu_busy'):.0f}% | {med('gpu_w'):.0f} W | {med('gpu_w') / med('short'):.2f} | "
              + ", ".join(f"{r['short']:.1f}/{r['long']:.1f}" for r in rs) + " |")
    out = pathlib.Path(a.out) if a.out else ROOT / "bench" / f"milestone_results_{model.stem}.json"
    out.write_text(json.dumps(results, indent=1), encoding="utf-8")
    print(f"\nraw: {out}")


if __name__ == "__main__":
    main()
