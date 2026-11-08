"""aic - "compiles" the fastest way to run an LLM on this machine, then runs it.

  python aic.py build --model models/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf
  python aic.py run   --model models/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf [--port 8080]
  python aic.py show  --model ...

Like a game compiling shaders for the installed GPU: `build` measures this
machine with short real runs and saves the best settings ("the build") in
cache/<model>__<hardware>.json. `run` loads that file and starts the engine
right away, without any search. If the hardware, driver or engine changed,
the saved build no longer matches and `run` says so.

What a build decides:
  * the mode: "moe-zc" (our hybrid expert engine, MoE models larger than VRAM),
    "gpu" (the whole model in VRAM, plain llama.cpp path) or "fit" (a dense model
    larger than VRAM, llama.cpp places the layers itself);
  * moe-zc: VRAM for the expert cache (everything the GPU does not need for other
    weights, KV cache and buffers, minus a safety margin for the desktop) and the
    number of CPU threads that compute experts not in VRAM;
  * speculative decoding: none, the model's own MTP head (draft length 1-3) or a
    small model of the same family with the same tokenizer (models/*.gguf).
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

def find_draft(model: pathlib.Path):
    """The model's own multi-token-prediction (MTP) head, if any:
    "embedded" when the GGUF contains the MTP layer, else a sidecar file
    mtp-<model base name>*.gguf next to it (as published by ggml-org), else None."""
    sys.path.insert(0, str(ROOT / "third_party" / "llama.cpp" / "gguf-py"))
    from gguf import GGUFReader
    r = GGUFReader(str(model))
    for k, f in r.fields.items():
        if k.endswith(".nextn_predict_layers") and int(f.contents()) > 0:
            return "embedded"
    base = model.stem.split("-UD-")[0].split("-Q")[0]
    side = sorted(model.parent.glob(f"mtp-{base}*.gguf"))
    return side[0].name if side else None


def find_mtp_heads(model: pathlib.Path) -> list:
    """All MTP heads of the model: "embedded", or every sidecar mtp-<base>*.gguf (some models ship several,
    e.g. the full MTP layer and one with only its shared expert, which costs less VRAM)."""
    first = find_draft(model)
    if first in (None, "embedded"):
        return [first] if first else []
    base = model.stem.split("-UD-")[0].split("-Q")[0]
    return [p.name for p in sorted(model.parent.glob(f"mtp-{base}*.gguf"))]


def model_info(model: pathlib.Path) -> dict:
    """What the build needs to know about a GGUF: MoE or dense, total size (all shards), tokenizer."""
    sys.path.insert(0, str(ROOT / "third_party" / "llama.cpp" / "gguf-py"))
    from gguf import GGUFReader
    r = GGUFReader(str(model))
    n_expert, pre, n_vocab = 0, "", 0
    tok = hashlib.sha256()
    for k, f in r.fields.items():
        if k.endswith(".expert_count"):
            n_expert = int(f.contents())
        elif k == "tokenizer.ggml.pre":
            pre = str(f.contents())
        elif k == "tokenizer.ggml.tokens":
            toks = f.contents()
            n_vocab = len(toks)
            for t in toks[:2000] + toks[-2000:]:
                tok.update(str(t).encode("utf-8", "replace"))
    shards = sorted(model.parent.glob(model.name.replace("-00001-of-", "-*-of-"))) if "-00001-of-" in model.name else [model]
    return {"n_expert": n_expert, "size_mib": sum(p.stat().st_size for p in shards) // 1048576,
            "tokenizer": f"{pre}:{n_vocab}:{tok.hexdigest()[:12]}"}


def find_small_drafts(model: pathlib.Path, info: dict) -> list:
    """Smaller models in the same folder with the same tokenizer (usable as a classic draft model),
    smallest first, at most two; a draft must be at most a third of the model."""
    out = []
    for p in sorted(model.parent.glob("*.gguf"), key=lambda p: p.stat().st_size):
        if p == model or p.name.startswith(("mtp-", "dflash-")) or "-MTP-" in p.name or "-of-0" in p.name and "-00001-of-" not in p.name:
            continue
        if p.stat().st_size // 1048576 > info["size_mib"] // 3:
            break
        try:
            di = model_info(p)
        except Exception:
            continue
        if di["tokenizer"] == info["tokenizer"]:
            out.append(p.name)
        if len(out) == 2:
            break
    return out


def power_plan() -> str:
    """Active Windows power plan: it changes CPU behavior, so a build records it (run warns on a change)."""
    try:
        import re
        out = subprocess.run(["powercfg", "/getactivescheme"], capture_output=True, text=True, errors="replace").stdout
        g = re.search(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}", out)
        return g.group(0) if g else ""
    except Exception:
        return ""


def draft_type(cfg: dict):
    """"mtp", "simple" or None (profiles made before draft_type existed: MTP when mtp_file is set)"""
    if cfg.get("draft_n", 0) <= 0:
        return None
    return cfg.get("draft_type") or ("mtp" if cfg.get("mtp_file") else None)


def server_args(model: pathlib.Path, cfg: dict) -> list:
    mode = cfg.get("mode", "moe-zc")
    args = ["-m", str(model), "-c", str(cfg.get("ctx", 20480)), "-np", "1"]
    if mode == "moe-zc":    # experts in RAM (our engine caches/computes them), the rest on the GPU
        args += ["-ngl", "99", "-cmoe", "-lm", "none", "-t", "1"]
    elif mode == "gpu":     # everything in VRAM
        args += ["-ngl", "99", "-lm", "none", "-t", "8"]
    else:                   # "fit": llama.cpp decides how many layers go to the GPU
        args += ["-lm", "none", "-t", "8"]
    dt = draft_type(cfg)
    if dt == "mtp":
        if cfg["mtp_file"] != "embedded":
            args += ["-md", str(model.parent / cfg["mtp_file"])]
        if cfg.get("draft_cpu_moe"):   # the MTP head's own experts stay in RAM (plain llama.cpp CPU path,
            # which needs its own threads: the main model runs with -t 1 in moe-zc mode)
            args += ["-cmoed", "-td", str(cfg.get("draft_threads", 8))]
        args += ["--spec-type", "draft-mtp", "--spec-draft-n-max", str(cfg["draft_n"])]
    elif dt == "simple":
        args += ["-md", str(model.parent / cfg["draft_model"]), "-ngld", "99",
                 "--spec-type", "draft-simple", "--spec-draft-n-max", str(cfg["draft_n"])]
    return args + sb.template_args(str(model))


def server_env(cfg: dict) -> dict:
    env = dict(os.environ)
    env["PATH"] = CUDA_BIN + os.pathsep + env.get("PATH", "")
    env.pop("LLAMA_MOE_ZC_MIB", None)
    if cfg.get("mode", "moe-zc") == "moe-zc":
        env["LLAMA_MOE_ZC_MIB"] = str(cfg["cache_mib"])
        env["LLAMA_MOE_ZC_THREADS"] = str(cfg.get("cpu_threads", 8))
    return env


def measure(model: pathlib.Path, cfg: dict, prompts: list, n_predict: int = 160, passes: int = 2) -> dict:
    """Starts the engine with cfg, warms it up, generates on the prompts `passes` times and
    returns the median speed (robust to a single slow or fast run) and free VRAM."""
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
        for _ in range(passes):
            for p in prompts:
                r = sb.post(base + "/completion", {"prompt": p, "n_predict": n_predict, "temperature": 0, "top_k": 1,
                                                   "ignore_eos": True, "cache_prompt": False})
                tg.append(r["timings"]["predicted_per_second"])
        return {"tg": statistics.median(tg), "tg_all": tg, "vram_free_before": free0, "vram_free_loaded": free1,
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
    pp = sb.prompts_path(str(model), calib=True)
    if pp.parent.name != "prompts":
        print(f"warning: no prompts made with this model's chat template, using {pp.name} (bench/make_model_prompts.py)")
    prompts = [p["prompt"] for p in json.loads(pp.read_text(encoding="utf-8"))[:3]]
    info = model_info(model)
    heads = find_mtp_heads(model)
    mtp = heads[0] if heads else None
    smalls = find_small_drafts(model, info)
    vram_free = sb.vram_free_mib()
    moe = info["n_expert"] > 0
    fits = info["size_mib"] + 1500 + MARGIN_MIB < vram_free   # weights + KV/buffers + desktop margin
    modes = (["moe-zc"] if moe else []) + (["gpu"] if fits else []) + (["fit"] if not moe and not fits else [])
    print(f"model: {info['size_mib']} MiB, {'MoE, ' + str(info['n_expert']) + ' experts' if moe else 'dense'}; "
          f"VRAM free {vram_free} MiB; modes to try: {modes}; MTP heads: {heads or 'none'}; small drafts: {smalls or 'none'}")
    t_start = time.time()
    trials = []
    fixed = None

    def trial(cfg):
        r = measure(model, cfg, prompts)
        trials.append({"cfg": cfg, **r})
        what = cfg["mode"] + (f", {cfg['cache_mib']:5d} MiB cache, {cfg['cpu_threads']} CPU threads" if cfg["mode"] == "moe-zc" else "")
        dt = draft_type(cfg)
        what += f", draft {dt + ' ' + str(cfg['draft_n']) if dt else 'none'}" + (f" ({cfg['draft_model']})" if dt == "simple" else "")             + (" (head experts in RAM)" if dt == "mtp" and cfg.get("draft_cpu_moe") else "")
        print(f"  {what} -> " + (f"{r['tg']:.1f} tok/s, VRAM free {r['vram_free_loaded']} MiB" if r["ok"] else f"FAILED {r['error']}"), flush=True)

    def key(cfg):
        return json.dumps(cfg, sort_keys=True)

    def ranked():
        """configs by the median over all their runs (a config may have been measured several times)"""
        runs = {}
        for t in trials:
            if t["ok"] and t["vram_free_loaded"] >= MARGIN_MIB // 2:
                runs.setdefault(key(t["cfg"]), (t["cfg"], []))[1].extend(t["tg_all"])
        return sorted(((statistics.median(v), c) for c, v in runs.values()), key=lambda x: -x[0])

    # draft options of the plain modes ("gpu", "fit"): none, the MTP head, small models of the family
    plain_drafts = [dict(draft_n=0)]
    for h in heads:
        plain_drafts += [dict(draft_type="mtp", mtp_file=h, draft_n=n) for n in (1, 2, 3)]
    for d in smalls:
        plain_drafts += [dict(draft_type="simple", draft_model=d, draft_n=n) for n in (4, 8)]

    if "moe-zc" in modes:
        base = {"mode": "moe-zc", "mtp_file": mtp, "draft_type": "mtp" if mtp else None, "draft_n": 2 if mtp else 0,
                "cpu_threads": 8, "ctx": a.ctx}

        # 1. how much VRAM does everything except the expert cache take? (with and without the draft head:
        #    without speculation the MTP layer is not loaded, which leaves more room for the expert cache)
        def probe_max_cache(draft_n, head=mtp, cpu_moe=False):
            pr = measure(model, dict(base, cache_mib=PROBE_CACHE_MIB, draft_n=draft_n, mtp_file=head, draft_cpu_moe=cpu_moe),
                         prompts[:1], n_predict=32, passes=1)
            if not pr["ok"]:
                raise SystemExit(f"engine failed to start: {pr['error']}")
            fx = pr["vram_free_before"] - pr["vram_free_loaded"] - PROBE_CACHE_MIB
            mc = pr["vram_free_before"] - fx - MARGIN_MIB
            print(f"fixed VRAM use (dense weights, KV, buffers{', draft' if draft_n else ''}): {fx} MiB -> expert cache up to {mc} MiB")
            return fx, mc

        fixed, max_cache_nodraft = probe_max_cache(0)
        max_cache = {h: probe_max_cache(2, h)[1] for h in heads}   # each head costs its own VRAM

        # 2. small search over the decisions that matter most (each trial = real engine run)
        for cache in (max_cache_nodraft, max_cache_nodraft - 512):
            trial(dict(base, cache_mib=int(cache), draft_n=0))
        for h in heads:
            for dn in (1, 2, 3):   # speculation can also hurt (a draft that guesses badly)
                for cache in (max_cache[h], max_cache[h] - 512):
                    trial(dict(base, cache_mib=int(cache), mtp_file=h, draft_type="mtp", draft_n=dn))
            # a head with its own routed experts (e.g. Qwen3.8's MTP layer, 512 experts, ~2 GB) can keep them in RAM:
            # slower drafting, but the VRAM goes to the main model's expert cache
            if h != "embedded" and model_info(model.parent / h)["n_expert"] > 0:
                mc = probe_max_cache(2, h, cpu_moe=True)[1]
                for dn in (2, 3):
                    for cache in (mc, mc - 512):
                        trial(dict(base, cache_mib=int(cache), mtp_file=h, draft_type="mtp", draft_n=dn, draft_cpu_moe=True))
        # a small model of the family as the draft instead of the MTP head (it costs VRAM: smaller cache)
        for d in smalls:
            d_mib = (model.parent / d).stat().st_size // 1048576
            # draft + 1 tokens are checked per step; the hybrid path takes at most 4 (ZC_MAX_TOKENS in moe-zc.cu),
            # longer drafts fall back to the prompt-processing path that copies experts every step (10x slower)
            for n in (2, 3):
                trial(dict(base, cache_mib=int(max_cache_nodraft - d_mib - 512), draft_type="simple", draft_model=d, draft_n=n))
        # then the CPU threads that compute the experts not in VRAM, around the best so far
        best_zc = [c for _, c in ranked() if c["mode"] == "moe-zc"]
        for n in (4, 6):
            if best_zc:
                cfg = dict(best_zc[0], cpu_threads=n)
                if key(cfg) not in {key(t["cfg"]) for t in trials}:
                    trial(cfg)

    for mode in ("gpu", "fit"):
        if mode in modes:
            for d in plain_drafts:
                trial(dict({"mode": mode, "ctx": a.ctx}, **d))

    if not ranked():
        raise SystemExit("no configuration worked")
    # the two finalists are measured once more (fresh engine start) and decided by the median of all their runs:
    # a single short trial is noisy (+-10%), a lucky run must not decide the build
    for _, cfg in ranked()[:2]:
        trial(cfg)
    score, best_cfg = ranked()[0]
    best = {"cfg": best_cfg, "tg": score}
    prof = {
        "model": str(model), "model_fingerprint": model_fingerprint(model), "hardware": hw,
        "power_plan": power_plan(), "model_info": info,
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
    if prof.get("power_plan") and prof["power_plan"] != power_plan():
        print(f"warning: the build was made with the power plan {prof['power_plan']}, now {power_plan()}; consider rebuilding")
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
