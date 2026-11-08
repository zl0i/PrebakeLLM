"""Search for the best llama.cpp settings on this machine (the baseline to beat).

Each stage is a list of configs; each config is a full llama-server run via
serverbench.py. Results (median generation tok/s per prompt) are appended to
bench/sweep_results.csv, which is committed, so the baseline is reproducible.

  python bench/sweep_llamacpp.py placement
  python bench/sweep_llamacpp.py threads --base "-ngl 99 -ncmoe 25 -lm none"
  python bench/sweep_llamacpp.py generic --model models/OTHER.gguf --draft embedded   (another model)
  python bench/sweep_llamacpp.py ik --server build/ik/bin/llama-server.exe            (ik_llama.cpp)

--draft replaces the Qwen3.6 MTP sidecar in the configs: a file path, "embedded"
(MTP layer inside the model file), or "none". Results of other models go to
bench/sweep_results_<model>.csv.

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
    # compact search for a new model: placement, expert cache sizes, draft length (all on P-cores, no mmap)
    "generic": [
        ("fit", "-t 8"),
        ("fit_lmnone", "-lm none -t 8"),
        ("cache5500", "-ngl 99 -cmoe --moe-cache-mib 5500 -lm none -t 8"),
        ("cache6500", "-ngl 99 -cmoe --moe-cache-mib 6500 -lm none -t 8"),
        ("cache7000", "-ngl 99 -cmoe --moe-cache-mib 7000 -lm none -t 8"),
        ("cache6000_mtp2", f"-ngl 99 -cmoe --moe-cache-mib 6000 -lm none -t 8 -md {MTP} --spec-type draft-mtp --spec-draft-n-max 2"),
        ("cache6000_mtp3", f"-ngl 99 -cmoe --moe-cache-mib 6000 -lm none -t 8 -md {MTP} --spec-type draft-mtp --spec-draft-n-max 3"),
        ("cache6500_mtp2", f"-ngl 99 -cmoe --moe-cache-mib 6500 -lm none -t 8 -md {MTP} --spec-type draft-mtp --spec-draft-n-max 2"),
        ("cache6500_mtp3", f"-ngl 99 -cmoe --moe-cache-mib 6500 -lm none -t 8 -md {MTP} --spec-type draft-mtp --spec-draft-n-max 3"),
    ],
    # ik_llama.cpp (fork tuned for hybrid CPU+GPU MoE): its own expert cache, merged up/gate experts, layer offload
    # (MTP: ik loads its own way; spec options of ik are given via --spec-type mtp:n_max=N)
    "ik": [
        ("ik_ncmoe25", "-ngl 99 -ncmoe 25 -nmm -t 8"),
        ("ik_ncmoe25_muge", "-ngl 99 -ncmoe 25 -nmm -t 8 -muge"),
        ("ik_ncmoe25_mtp2", f"-ngl 99 -ncmoe 25 -nmm -t 8 -md {MTP} --spec-type mtp:n_max=2"),
        ("ik_cache6000", "-ngl 99 -cmoe --moe-cache-mib 6000 -nmm -t 8"),
        ("ik_cache6500", "-ngl 99 -cmoe --moe-cache-mib 6500 -nmm -t 8"),
        ("ik_cache7000", "-ngl 99 -cmoe --moe-cache-mib 7000 -nmm -t 8"),
        ("ik_cache6500_muge", "-ngl 99 -cmoe --moe-cache-mib 6500 -nmm -t 8 -muge"),
        ("ik_cache6500_mtp1", f"-ngl 99 -cmoe --moe-cache-mib 6500 -nmm -t 8 -md {MTP} --spec-type mtp:n_max=1"),
        ("ik_cache6500_mtp2", f"-ngl 99 -cmoe --moe-cache-mib 6500 -nmm -t 8 -md {MTP} --spec-type mtp:n_max=2"),
        ("ik_cache6500_mtp3", f"-ngl 99 -cmoe --moe-cache-mib 6500 -nmm -t 8 -md {MTP} --spec-type mtp:n_max=3"),
        ("ik_cache6000_mtp2", f"-ngl 99 -cmoe --moe-cache-mib 6000 -nmm -t 8 -md {MTP} --spec-type mtp:n_max=2"),
        ("ik_cache6500_mtp2_t6", f"-ngl 99 -cmoe --moe-cache-mib 6500 -nmm -t 6 -md {MTP} --spec-type mtp:n_max=2"),
    ],
    # GLM-4.7-Flash without an MTP layer (unsloth file): no draft possible, so placement x cache x threads
    "glm_nodraft": [
        ("fit_lmnone", "-lm none -t 8"),
        ("cache7000_t8", "-ngl 99 -cmoe --moe-cache-mib 7000 -lm none -t 8"),
        ("cache7500_t8", "-ngl 99 -cmoe --moe-cache-mib 7500 -lm none -t 8"),
        ("cache8000_t6", "-ngl 99 -cmoe --moe-cache-mib 8000 -lm none -t 6"),
        ("cache8000_t8", "-ngl 99 -cmoe --moe-cache-mib 8000 -lm none -t 8"),
        ("cache8500_t6", "-ngl 99 -cmoe --moe-cache-mib 8500 -lm none -t 6"),
    ],
    # the same for ik_llama.cpp (MLA 3, fused MoE and flash attention are its defaults)
    "ik_glm": [
        ("ik_ncmoe28", "-ngl 99 -ncmoe 28 -nmm -t 8"),
        ("ik_ncmoe32", "-ngl 99 -ncmoe 32 -nmm -t 8"),
        ("ik_cache7500", "-ngl 99 -cmoe --moe-cache-mib 7500 -nmm -t 8"),
        ("ik_cache8000", "-ngl 99 -cmoe --moe-cache-mib 8000 -nmm -t 8"),
        ("ik_cache8000_t6", "-ngl 99 -cmoe --moe-cache-mib 8000 -nmm -t 6"),
        ("ik_cache8000_muge", "-ngl 99 -cmoe --moe-cache-mib 8000 -nmm -t 8 -muge"),
        ("ik_cache8000_ger", "-ngl 99 -cmoe --moe-cache-mib 8000 -nmm -t 8 -ger"),
        ("ik_cache8000_rtr", "-ngl 99 -cmoe --moe-cache-mib 8000 -nmm -t 8 -rtr"),
        ("ik_cache8500", "-ngl 99 -cmoe --moe-cache-mib 8500 -nmm -t 8"),
        ("ik_cache9000", "-ngl 99 -cmoe --moe-cache-mib 9000 -nmm -t 8"),
        ("ik_cache9500", "-ngl 99 -cmoe --moe-cache-mib 9500 -nmm -t 8"),
        ("ik_cache9000_t6", "-ngl 99 -cmoe --moe-cache-mib 9000 -nmm -t 6"),
    ],
    # Qwen3.6 after closing background apps (more free VRAM): does llama.cpp also gain from a bigger cache?
    "qwen_morevram": [
        ("cache7000_mtp2", f"-ngl 99 -cmoe --moe-cache-mib 7000 -lm none -t 8 -md {MTP} --spec-type draft-mtp --spec-draft-n-max 2"),
        ("cache7000_mtp3", f"-ngl 99 -cmoe --moe-cache-mib 7000 -lm none -t 8 -md {MTP} --spec-type draft-mtp --spec-draft-n-max 3"),
        ("cache7500_mtp2", f"-ngl 99 -cmoe --moe-cache-mib 7500 -lm none -t 8 -md {MTP} --spec-type draft-mtp --spec-draft-n-max 2"),
        ("cache7500_mtp3", f"-ngl 99 -cmoe --moe-cache-mib 7500 -lm none -t 8 -md {MTP} --spec-type draft-mtp --spec-draft-n-max 3"),
        ("cache6500_mtp2", f"-ngl 99 -cmoe --moe-cache-mib 6500 -lm none -t 8 -md {MTP} --spec-type draft-mtp --spec-draft-n-max 2"),
    ],
    # Gemma 4 26B-A4B (16 GB, almost fits): bigger caches, MTP sidecar (gemma4-assistant) via draft-mtp
    "gemma": [
        ("cache8000", "-ngl 99 -cmoe --moe-cache-mib 8000 -lm none -t 8"),
        ("cache9000", "-ngl 99 -cmoe --moe-cache-mib 9000 -lm none -t 8"),
        ("cache7500_mtp2", f"-ngl 99 -cmoe --moe-cache-mib 7500 -lm none -t 8 -md {MTP} --spec-type draft-mtp --spec-draft-n-max 2"),
        ("cache8500_mtp2", f"-ngl 99 -cmoe --moe-cache-mib 8500 -lm none -t 8 -md {MTP} --spec-type draft-mtp --spec-draft-n-max 2"),
        ("cache8500_mtp3", f"-ngl 99 -cmoe --moe-cache-mib 8500 -lm none -t 8 -md {MTP} --spec-type draft-mtp --spec-draft-n-max 3"),
        ("cache8500_mtp4", f"-ngl 99 -cmoe --moe-cache-mib 8500 -lm none -t 8 -md {MTP} --spec-type draft-mtp --spec-draft-n-max 4"),
        ("fit_mtp3", f"-lm none -t 8 -md {MTP} --spec-type draft-mtp --spec-draft-n-max 3"),
    ],
    "gemma2": [
        ("cache7000_mtp3", f"-ngl 99 -cmoe --moe-cache-mib 7000 -lm none -t 8 -md {MTP} --spec-type draft-mtp --spec-draft-n-max 3"),
        ("cache7000_mtp2", f"-ngl 99 -cmoe --moe-cache-mib 7000 -lm none -t 8 -md {MTP} --spec-type draft-mtp --spec-draft-n-max 2"),
        ("cache6500_mtp4", f"-ngl 99 -cmoe --moe-cache-mib 6500 -lm none -t 8 -md {MTP} --spec-type draft-mtp --spec-draft-n-max 4"),
        ("cache6500_mtp3_t6", f"-ngl 99 -cmoe --moe-cache-mib 6500 -lm none -t 6 -md {MTP} --spec-type draft-mtp --spec-draft-n-max 3"),
    ],
    # Qwen3.8-Flash-Next (ISTA-DASLab Q2_0, 512 experts, PLE table read lazily from disk): cache sizes, the two
    # MTP heads of unsloth (full: the MTP layer with its 512 experts; shared: only its shared expert), draft lengths
    "qwen38": [
        ("cache5000", "-ngl 99 -cmoe --moe-cache-mib 5000 -lm none -t 8"),
        ("cache6500", "-ngl 99 -cmoe --moe-cache-mib 6500 -lm none -t 8"),
        ("cache6000_t6", "-ngl 99 -cmoe --moe-cache-mib 6000 -lm none -t 6"),
        ("shared_cache4500_mtp2", "-ngl 99 -cmoe --moe-cache-mib 4500 -lm none -t 8 -md models/qwen38/mtp-Qwen3.8-Flash-Next-GSQ-RCO-shared-Q4_K_M.gguf --spec-type draft-mtp --spec-draft-n-max 2"),
        ("shared_cache4500_mtp3", "-ngl 99 -cmoe --moe-cache-mib 4500 -lm none -t 8 -md models/qwen38/mtp-Qwen3.8-Flash-Next-GSQ-RCO-shared-Q4_K_M.gguf --spec-type draft-mtp --spec-draft-n-max 3"),
        ("full_cache3500_mtp2", "-ngl 99 -cmoe --moe-cache-mib 3500 -lm none -t 8 -md models/qwen38/mtp-Qwen3.8-Flash-Next-GSQ-RCO-Q4_K_M.gguf --spec-type draft-mtp --spec-draft-n-max 2"),
        ("full_cache3500_mtp3", "-ngl 99 -cmoe --moe-cache-mib 3500 -lm none -t 8 -md models/qwen38/mtp-Qwen3.8-Flash-Next-GSQ-RCO-Q4_K_M.gguf --spec-type draft-mtp --spec-draft-n-max 3"),
        ("full_cmoed_cache5000_mtp2", "-ngl 99 -cmoe --moe-cache-mib 5000 -lm none -t 8 -md models/qwen38/mtp-Qwen3.8-Flash-Next-GSQ-RCO-Q4_K_M.gguf -cmoed --spec-type draft-mtp --spec-draft-n-max 2"),
    ],
    "qwen38b": [
        ("full_cache4000_mtp3", "-ngl 99 -cmoe --moe-cache-mib 4000 -lm none -t 8 -md models/qwen38/mtp-Qwen3.8-Flash-Next-GSQ-RCO-Q4_K_M.gguf --spec-type draft-mtp --spec-draft-n-max 3"),
        ("full_cmoed_cache5000_mtp3", "-ngl 99 -cmoe --moe-cache-mib 5000 -lm none -t 8 -md models/qwen38/mtp-Qwen3.8-Flash-Next-GSQ-RCO-Q4_K_M.gguf -cmoed --spec-type draft-mtp --spec-draft-n-max 3"),
        ("full_cmoed_cache5500_mtp3", "-ngl 99 -cmoe --moe-cache-mib 5500 -lm none -t 8 -md models/qwen38/mtp-Qwen3.8-Flash-Next-GSQ-RCO-Q4_K_M.gguf -cmoed --spec-type draft-mtp --spec-draft-n-max 3"),
    ],
    # the larger caches the first passes pointed to (GLM unsloth file is 0.5 GB smaller than meshllm's)
    "glm_nodraft2": [
        ("cache9000_t6", "-ngl 99 -cmoe --moe-cache-mib 9000 -lm none -t 6"),
        ("cache9500_t6", "-ngl 99 -cmoe --moe-cache-mib 9500 -lm none -t 6"),
        ("cache9000_t8", "-ngl 99 -cmoe --moe-cache-mib 9000 -lm none -t 8"),
    ],
}


def run(tag: str, args: str, only: str, reps: int, model: str = "", server: str = "", plain: bool = False) -> dict:
    cmd = [sys.executable, str(ROOT / "bench" / "serverbench.py"), "--tag", tag, "--reps", str(reps), "--only", only]
    if plain:
        cmd += ["--plain"]
    if model:
        cmd += ["--model", model]
    if server:
        cmd += ["--server", server]
    cmd += ["--"] + shlex.split(args)
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
    ap.add_argument("--model", default="", help="model file (default: Qwen3.6-35B-A3B)")
    ap.add_argument("--server", default="", help="server binary (default: official llama.cpp)")
    ap.add_argument("--draft", default="", help='MTP draft for this model: file, "embedded" or "none"')
    ap.add_argument("--tag-prefix", default="")
    ap.add_argument("--plain", action="store_true", help="passed to serverbench (servers without --no-webui)")
    a = ap.parse_args()

    def adapt(args: str) -> str:
        if not a.draft or MTP not in args:
            return args
        if a.draft == "none":
            return ""   # config needs a draft the model does not have: skipped
        return args.replace(f"-md {MTP}", "" if a.draft == "embedded" else f"-md {a.draft}")

    out = ROOT / "bench" / ("sweep_results.csv" if not a.model else f"sweep_results_{pathlib.Path(a.model).stem}.csv")
    new = not out.exists()
    with open(out, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["time", "stage", "tag", "args", "prompts", "tg_median_by_prompt", "tg_mean",
                        "pp_by_prompt", "draft_accept", "vram_free_loaded_mib", "load_s", "log"])
        for tag, args in STAGES[a.stage]:
            args = adapt(args)
            if not args:
                continue
            tag = a.tag_prefix + tag
            full = (a.base + " " + args).strip()
            r = run(tag, full, a.only, a.reps, a.model, a.server, a.plain)
            tg_mean = round(statistics.mean(r["tg"].values()), 2) if r.get("tg") else ""
            w.writerow([datetime.datetime.now().isoformat(timespec="seconds"), a.stage, tag, full, a.only,
                        json.dumps(r.get("tg")), tg_mean, json.dumps(r.get("pp")), json.dumps(r.get("acc")),
                        r.get("vram_free_loaded"), r.get("load_s"), r.get("log", r.get("error"))])
            f.flush()
            print(f"==> {tag:18s} tg_mean={tg_mean}  {r.get('tg')}  acc={r.get('acc')}", flush=True)


if __name__ == "__main__":
    main()
