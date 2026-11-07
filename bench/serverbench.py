"""End-to-end speed + output capture through a real llama-server run.

Usage:
  python bench/serverbench.py --tag base_ncmoe30 -- -ngl 99 -ncmoe 30 -fa on -t 8

Everything after `--` goes to llama-server verbatim. For each prompt in
bench/prompts.json the script runs greedy generation of exactly --n-predict
tokens (ignore_eos), --reps times, and writes one JSON line per run to
bench/logs/<date>_<tag>.jsonl: server timings (prompt and generation tok/s,
draft acceptance), generated text and token ids, plus free VRAM. Token ids
let us compare outputs of different engines/configs exactly.

Works with any server speaking the llama-server /completion API, so later
our own engine is measured by exactly the same script.
"""
import argparse
import datetime
import json
import pathlib
import statistics
import subprocess
import sys
import time
import urllib.request

ROOT = pathlib.Path(__file__).resolve().parent.parent
DEFAULT_SERVER = ROOT / "third_party" / "llama-bin" / "llama-server.exe"
DEFAULT_MODEL = ROOT / "models" / "Qwen3.6-35B-A3B-UD-Q4_K_M.gguf"


def vram_free_mib() -> int:
    out = subprocess.run(["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
                         capture_output=True, text=True).stdout
    return int(out.strip().splitlines()[0])


def post(url: str, body: dict, timeout: float = 1800) -> dict:
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def wait_ready(base: str, proc: subprocess.Popen, timeout: float = 600) -> None:
    t0 = time.time()
    while time.time() - t0 < timeout:
        if proc.poll() is not None:
            raise RuntimeError(f"server exited with code {proc.returncode}")
        try:
            with urllib.request.urlopen(base + "/health", timeout=2) as r:
                if r.status == 200:
                    return
        except Exception:
            pass
        time.sleep(0.5)
    raise TimeoutError("server did not become ready")


# llama-server only handles Ctrl+C for a clean shutdown (which prints e.g. MoE
# cache hit statistics). On Windows a Ctrl+C can only be sent to a whole
# console, so the server runs in its own hidden console and a tiny helper
# process attaches to it and raises Ctrl+C there.
_CTRL_C_HELPER = (
    "import ctypes,sys;k=ctypes.windll.kernel32;k.FreeConsole();"
    "k.AttachConsole(int(sys.argv[1])) or sys.exit(1);"
    "k.SetConsoleCtrlHandler(None,True);k.GenerateConsoleCtrlEvent(0,0)"
)


# Processes inherit "ignore Ctrl+C" from their parent (e.g. a shell running us
# in the background). This wrapper re-enables Ctrl+C, then runs the server, so
# the server inherits the enabled state; the wrapper itself waits it out.
_CTRL_C_WRAPPER = (
    "import ctypes,subprocess,sys;ctypes.windll.kernel32.SetConsoleCtrlHandler(None,False);"
    "p=subprocess.Popen(sys.argv[1:])\n"
    "while True:\n"
    " try: sys.exit(p.wait())\n"
    " except KeyboardInterrupt: pass"
)


def start_server(cmd, log_file) -> subprocess.Popen:
    if sys.platform != "win32":
        return subprocess.Popen(cmd, stdout=log_file, stderr=subprocess.STDOUT)
    cmd = [sys.executable, "-c", _CTRL_C_WRAPPER] + list(cmd)
    si = subprocess.STARTUPINFO()
    si.dwFlags |= subprocess.STARTF_USESHOWWINDOW
    si.wShowWindow = 0  # SW_HIDE
    return subprocess.Popen(cmd, stdout=log_file, stderr=subprocess.STDOUT,
                            creationflags=subprocess.CREATE_NEW_CONSOLE, startupinfo=si)


def stop_server(proc: subprocess.Popen) -> None:
    if proc.poll() is not None:
        return
    if sys.platform == "win32":
        subprocess.run([sys.executable, "-c", _CTRL_C_HELPER, str(proc.pid)])
    else:
        proc.send_signal(2)
    try:
        proc.wait(timeout=60)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--server", default=str(DEFAULT_SERVER))
    ap.add_argument("--model", default=str(DEFAULT_MODEL))
    ap.add_argument("--n-predict", type=int, default=256)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--only", default="", help="comma-separated prompt ids")
    ap.add_argument("--port", type=int, default=8099)
    ap.add_argument("--ctx", type=int, default=20480)
    ap.add_argument("--n-probs", type=int, default=0,
                    help="store top-N token probabilities per generated token (for correctness analysis; slows generation)")
    ap.add_argument("server_args", nargs=argparse.REMAINDER)
    a = ap.parse_args()
    extra = [x for x in a.server_args if x != "--"]

    prompts = json.loads((ROOT / "bench" / "prompts.json").read_text(encoding="utf-8"))
    warmup = next(p for p in prompts if p["id"] == "warmup")
    prompts = [p for p in prompts if p["id"] != "warmup"]
    if a.only:
        keep = set(a.only.split(","))
        prompts = [p for p in prompts if p["id"] in keep]

    logdir = ROOT / "bench" / "logs"
    logdir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    log_path = logdir / f"{stamp}_{a.tag}.jsonl"
    srv_log = logdir / f"{stamp}_{a.tag}.server.log"

    cmd = [str(pathlib.Path(a.server).resolve()), "-m", str(pathlib.Path(a.model).resolve()), "--port", str(a.port), "-c", str(a.ctx), "-np", "1",
           "--no-webui", "--metrics"] + extra
    vram_before = vram_free_mib()
    t_start = time.time()
    with open(srv_log, "w", encoding="utf-8") as slog:
        proc = start_server(cmd, slog)
        base = f"http://127.0.0.1:{a.port}"
        try:
            wait_ready(base, proc)
            load_s = time.time() - t_start
            vram_loaded = vram_free_mib()
            print(f"[{a.tag}] loaded in {load_s:.1f}s, VRAM free {vram_before} -> {vram_loaded} MiB")
            # warmup: long enough to touch most experts, so mmap'ed weights are paged in
            post(base + "/completion", {"prompt": warmup["prompt"], "n_predict": 128, "temperature": 0,
                                        "ignore_eos": True, "cache_prompt": False})

            header = {"kind": "header", "tag": a.tag, "cmd": cmd, "load_s": load_s,
                      "vram_free_before_mib": vram_before, "vram_free_loaded_mib": vram_loaded,
                      "n_predict": a.n_predict, "time": stamp}
            results = {}
            with open(log_path, "w", encoding="utf-8") as log:
                log.write(json.dumps(header, ensure_ascii=False) + "\n")
                # round-robin over prompts, so a repeat never directly follows the same prompt
                for rep in range(a.reps):
                    for p in prompts:
                        body = {"prompt": p["prompt"], "n_predict": a.n_predict, "temperature": 0, "top_k": 1,
                                "ignore_eos": True, "cache_prompt": False, "return_tokens": True, "seed": 1}
                        if a.n_probs:
                            body["n_probs"] = a.n_probs
                        t0 = time.time()
                        r = post(base + "/completion", body)
                        wall = time.time() - t0
                        tm = r.get("timings", {})
                        rec = {"kind": "run", "id": p["id"], "rep": rep, "wall_s": wall, "timings": tm,
                               "tokens": r.get("tokens"), "content": r.get("content")}
                        if a.n_probs:
                            # per generated token: [(token_id, logprob), ...] top-N candidates
                            rec["top"] = [[(t["id"], t["logprob"]) for t in step.get("top_logprobs", [])]
                                          for step in r.get("completion_probabilities", [])]
                        log.write(json.dumps(rec, ensure_ascii=False) + "\n")
                        log.flush()
                        results.setdefault(p["id"], []).append(tm)
                        acc = ""
                        if tm.get("draft_n"):
                            acc = f"  draft {tm['draft_n_accepted']}/{tm['draft_n']}"
                        print(f"  {p['id']:8s} rep{rep}: pp {tm.get('prompt_per_second', 0):8.1f} t/s"
                              f" ({tm.get('prompt_n')} tok)  tg {tm.get('predicted_per_second', 0):6.2f} t/s{acc}")
            print(f"[{a.tag}] median tg t/s: " + ", ".join(
                f"{k}={statistics.median(x['predicted_per_second'] for x in v):.2f}" for k, v in results.items()))
            print(f"log: {log_path}")
        finally:
            stop_server(proc)
    # surface engine statistics printed at shutdown (e.g. MoE cache hit rate)
    for line in open(srv_log, encoding="utf-8", errors="replace"):
        if "llama_moe_cache:" in line:
            print("  " + line.strip())


if __name__ == "__main__":
    sys.exit(main())
