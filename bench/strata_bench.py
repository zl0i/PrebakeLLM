"""Measures a running Strata server (OpenAI chat API) on the same prompts as bench/serverbench.py.

  python bench/strata_bench.py --model models/qwen38/<shard1>.gguf --url http://127.0.0.1:8091 --tag strata [--rounds 2]

Strata (github.com/Niko1221/Strata, MIT) has no raw /completion endpoint, so the user message is taken back out of
our pre-rendered prompt (Qwen chat template, thinking off) and sent as a chat message with reasoning_effort=none.
Strata renders the same template: the prompt token counts match ours. Greedy (temperature 0), 256 tokens, EOS
ignored, as serverbench does. Speeds are Strata's own `timings` (llama.cpp names).

Differences that remain: Strata keeps a conversation/prompt cache between requests (the second round of the long
prompt may read part of it from the cache: prompt speed is only comparable in the first round), and it is started
by `strata launch` with its own defaults (MTP draft layer, int8 KV, tuning skipped).
Writes bench/logs/<date>_<tag>.jsonl like serverbench.
"""
import argparse
import datetime
import json
import pathlib
import sys
import time
import urllib.request

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "bench"))
import serverbench as sb  # noqa: E402


def chat(url, msg, n):
    body = {"model": "strata", "messages": [{"role": "user", "content": msg}], "max_tokens": n, "temperature": 0,
            "reasoning_effort": "none", "ignore_eos": True}
    r = urllib.request.Request(url + "/v1/chat/completions", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(r, timeout=3600) as f:
        return json.load(f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--url", default="http://127.0.0.1:8091")
    ap.add_argument("--tag", default="strata")
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--n-predict", type=int, default=256)
    a = ap.parse_args()
    prompts = json.loads(sb.prompts_path(a.model).read_text(encoding="utf-8"))
    user = lambda p: p.split("<|im_start|>user\n", 1)[1].split("<|im_end|>", 1)[0]
    out = ROOT / "bench" / "logs" / f"{datetime.datetime.now():%Y%m%d-%H%M%S}_{a.tag}.jsonl"
    with open(out, "w", encoding="utf-8") as log:
        log.write(json.dumps({"kind": "header", "url": a.url, "model": a.model, "engine": "strata"}) + "\n")
        warm = [p for p in prompts if p["id"] == "warmup"]
        if warm:
            chat(a.url, user(warm[0]["prompt"]), 96)
        for rnd in range(a.rounds):
            for p in prompts:
                if p["id"] == "warmup":
                    continue
                t0 = time.time()
                d = chat(a.url, user(p["prompt"]), a.n_predict)
                t = d["timings"]
                rec = {"kind": "run", "id": p["id"], "rep": rnd, "timings": t, "wall_s": round(time.time() - t0, 2),
                       "content": d["choices"][0]["message"].get("content", ""), "usage": d.get("usage")}
                log.write(json.dumps(rec, ensure_ascii=False) + "\n")
                log.flush()
                acc = f"  draft {t.get('draft_n_accepted')}/{t.get('draft_n')}" if t.get("draft_n") else ""
                print(f"  {p['id']:8s} rep{rnd}: pp {t['prompt_per_second']:8.1f} t/s ({t['prompt_n']} tok)  tg {t['predicted_per_second']:7.2f} t/s{acc}",
                      flush=True)
    print(f"log: {out}")


if __name__ == "__main__":
    main()
