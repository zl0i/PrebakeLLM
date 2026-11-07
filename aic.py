"""aic - "compiles" the fastest way to run an LLM on this machine, then runs it.

  python aic.py build --model models/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf
  python aic.py run   --model models/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf [--port 8080]
  python aic.py show  --model ...

Like a game compiling shaders for the installed GPU: `build` measures this
machine with short real runs and saves the best settings ("the build") in
cache/<model>__<hardware>.json. `run` loads that file and starts the engine
right away, without any search. If the hardware, driver or engine changed,
the saved build no longer matches and `run` says so.

What a build decides (first milestone: MoE models larger than VRAM):
  * VRAM for the expert cache: everything the GPU does not need for other
    weights, KV cache and buffers, minus a safety margin for the desktop;
  * the speculative decoding draft length (MTP head of the model);
  * the number of CPU threads that compute experts not in VRAM.
The engine itself is our fork of llama.cpp (engine/patches, build/llama-fork).
"""
import argparse
import hashlib
import json
import os
import pathlib
import platform
import statistics
import subprocess
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "bench"))
import serverbench as sb  # noqa: E402  (server start/stop, HTTP helpers)

ENGINE = ROOT / "build" / "llama-fork" / "bin" / "llama-server.exe"
CACHE = ROOT / "cache"
CUDA_BIN = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.5\bin\x64"
MARGIN_MIB = 700          # VRAM left free for the desktop and its fluctuations
PROBE_CACHE_MIB = 1024    # cache size of the first run that measures fixed VRAM use


# ---------------------------------------------------------------- fingerprints

def hw_fingerprint() -> dict:
    q = lambda args: subprocess.run(args, capture_output=True, text=True).stdout.strip()
    gpu = q(["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader,nounits"]).split(", ")
    cpu = platform.processor()
    try:
        cpu = q(["powershell", "-NoProfile", "-Command", "(Get-CimInstance Win32_Processor).Name"]) or cpu
        ram = int(q(["powershell", "-NoProfile", "-Command",
                     "(Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory"])) // (1 << 30)
    except Exception:
        ram = 0
    return {"gpu": gpu[0], "vram_mib": int(gpu[1]), "driver": gpu[2], "cpu": cpu.strip(), "ram_gib": ram}


def engine_version() -> str:
    # the engine = pinned llama.cpp commit + our patches; hash the patch files
    h = hashlib.sha256()
    for p in sorted((ROOT / "engine" / "patches").glob("*.patch")):
        h.update(p.read_bytes())
    return h.hexdigest()[:12]


def model_fingerprint(path: pathlib.Path) -> str:
    # fast: size + first and last MiB (hashing 22 GB would take a minute)
    h = hashlib.sha256(str(path.stat().st_size).encode())
    with open(path, "rb") as f:
        h.update(f.read(1 << 20))
        f.seek(-(1 << 20), 2)
        h.update(f.read(1 << 20))
    return h.hexdigest()[:12]


def profile_path(model: pathlib.Path, hw: dict) -> pathlib.Path:
    hw_key = hashlib.sha256(json.dumps(hw, sort_keys=True).encode()).hexdigest()[:8]
    return CACHE / f"{model.stem}__{hw_key}.json"


# ---------------------------------------------------------------- measuring

def server_args(model: pathlib.Path, cfg: dict) -> list:
    args = ["-m", str(model), "-ngl", "99", "-cmoe", "-lm", "none", "-t", "1", "-c", str(cfg.get("ctx", 20480)), "-np", "1"]
    mtp = model.parent / cfg["mtp_file"] if cfg.get("mtp_file") else None
    if mtp and cfg.get("draft_n", 0) > 0:
        args += ["-md", str(mtp), "--spec-type", "draft-mtp", "--spec-draft-n-max", str(cfg["draft_n"])]
    return args


def server_env(cfg: dict) -> dict:
    env = dict(os.environ)
    env["PATH"] = CUDA_BIN + os.pathsep + env.get("PATH", "")
    env["LLAMA_MOE_ZC_MIB"] = str(cfg["cache_mib"])
    env["LLAMA_MOE_ZC_THREADS"] = str(cfg.get("cpu_threads", 8))
    return env


def measure(model: pathlib.Path, cfg: dict, prompts: list, n_predict: int = 160) -> dict:
    """Starts the engine with cfg, generates on the prompts, returns speed and free VRAM."""
    port = 8097
    log = open(ROOT / "bench" / "logs" / "aic_build.server.log", "w", encoding="utf-8")
    env_backup = dict(os.environ)
    os.environ.update(server_env(cfg))
    try:
        free0 = sb.vram_free_mib()
        t0 = time.time()
        proc = sb.start_server([str(ENGINE)] + server_args(model, cfg) + ["--port", str(port), "--no-webui"], log)
    finally:
        os.environ.clear()
        os.environ.update(env_backup)
    base = f"http://127.0.0.1:{port}"
    try:
        sb.wait_ready(base, proc)
        load_s = time.time() - t0
        free1 = sb.vram_free_mib()
        sb.post(base + "/completion", {"prompt": prompts[0], "n_predict": 96, "temperature": 0, "ignore_eos": True})
        tg = []
        for p in prompts:
            r = sb.post(base + "/completion", {"prompt": p, "n_predict": n_predict, "temperature": 0, "top_k": 1,
                                               "ignore_eos": True, "cache_prompt": False})
            tg.append(r["timings"]["predicted_per_second"])
        return {"tg": statistics.mean(tg), "tg_all": tg, "vram_free_before": free0, "vram_free_loaded": free1,
                "load_s": round(load_s, 1), "ok": True}
    except Exception as e:  # out of memory, crash...
        return {"ok": False, "error": str(e)}
    finally:
        sb.stop_server(proc)
        log.close()


# ---------------------------------------------------------------- commands

def cmd_build(a):
    model = pathlib.Path(a.model).resolve()
    hw = hw_fingerprint()
    print(f"hardware: {hw}")
    prompts = [p["prompt"] for p in json.loads((ROOT / "bench" / "calib_prompts.json").read_text(encoding="utf-8"))[:3]]
    mtp = next(iter(sorted(model.parent.glob("mtp-*.gguf"))), None)
    base = {"mtp_file": mtp.name if mtp else None, "draft_n": 2 if mtp else 0, "cpu_threads": 8, "ctx": a.ctx}
    t_start = time.time()
    trials = []

    # 1. how much VRAM does everything except the expert cache take?
    probe = measure(model, dict(base, cache_mib=PROBE_CACHE_MIB), prompts[:1], n_predict=32)
    if not probe["ok"]:
        raise SystemExit(f"engine failed to start: {probe['error']}")
    fixed = probe["vram_free_before"] - probe["vram_free_loaded"] - PROBE_CACHE_MIB
    max_cache = probe["vram_free_before"] - fixed - MARGIN_MIB
    print(f"fixed VRAM use (dense weights, KV, buffers, draft): {fixed} MiB -> expert cache up to {max_cache} MiB")

    # 2. small search over the decisions that matter most (each trial = real engine run)
    grid = []
    for cache in (max_cache, max_cache - 512):
        for dn in ([2, 3] if mtp else [0]):
            grid.append(dict(base, cache_mib=int(cache), draft_n=dn))
    def trial(cfg):
        r = measure(model, cfg, prompts)
        trials.append({"cfg": cfg, **r})
        print(f"  {cfg['cache_mib']:5d} MiB cache, draft {cfg['draft_n']}, {cfg['cpu_threads']} CPU threads -> "
              + (f"{r['tg']:.1f} tok/s, VRAM free {r['vram_free_loaded']} MiB" if r["ok"] else f"FAILED {r['error']}"), flush=True)

    def best_so_far():
        ok = [t for t in trials if t["ok"] and t["vram_free_loaded"] >= MARGIN_MIB // 2]
        return max(ok, key=lambda t: t["tg"])

    for cfg in grid:
        trial(cfg)
    # then the CPU threads that compute the experts not in VRAM, around the best so far
    for n in (4, 5, 6):
        trial(dict(best_so_far()["cfg"], cpu_threads=n))
    best = best_so_far()
    prof = {
        "model": str(model), "model_fingerprint": model_fingerprint(model), "hardware": hw,
        "engine": engine_version(), "built": time.strftime("%Y-%m-%d %H:%M:%S"),
        "build_seconds": round(time.time() - t_start), "fixed_vram_mib": fixed,
        "config": best["cfg"], "expected_tg": round(best["tg"], 1), "trials": trials,
    }
    CACHE.mkdir(exist_ok=True)
    out = profile_path(model, hw)
    out.write_text(json.dumps(prof, indent=1), encoding="utf-8")
    print(f"best: {best['cfg']} -> {best['tg']:.1f} tok/s; saved {out} (build took {prof['build_seconds']} s)")


def load_profile(model: pathlib.Path) -> dict:
    hw = hw_fingerprint()
    p = profile_path(model, hw)
    if not p.exists():
        raise SystemExit(f"no build for this model on this machine ({p.name}); run: python aic.py build --model {model}")
    prof = json.loads(p.read_text(encoding="utf-8"))
    if prof["engine"] != engine_version():
        print("warning: the engine changed since this build; consider rebuilding")
    if prof["model_fingerprint"] != model_fingerprint(model):
        raise SystemExit("the model file changed since this build; run build again")
    return prof


def cmd_run(a):
    model = pathlib.Path(a.model).resolve()
    t0 = time.time()
    prof = load_profile(model)
    cfg = prof["config"]
    print(f"using build {prof['built']}: {cfg} (expected ~{prof['expected_tg']} tok/s), found in {time.time() - t0:.2f} s")
    args = [str(ENGINE)] + server_args(model, cfg) + ["--port", str(a.port)] + a.extra
    os.execve(str(ENGINE), args, server_env(cfg)) if os.name != "nt" else sys.exit(subprocess.call(args, env=server_env(cfg)))


def cmd_show(a):
    print(json.dumps(load_profile(pathlib.Path(a.model).resolve()), indent=1))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--model", required=True)
    b.add_argument("--ctx", type=int, default=20480)
    r = sub.add_parser("run")
    r.add_argument("--model", required=True)
    r.add_argument("--port", type=int, default=8080)
    r.add_argument("extra", nargs=argparse.REMAINDER)
    s = sub.add_parser("show")
    s.add_argument("--model", required=True)
    a = ap.parse_args()
    {"build": cmd_build, "run": cmd_run, "show": cmd_show}[a.cmd](a)


if __name__ == "__main__":
    main()
