"""Where does the GPU time go in a time window of an nsys profile (sqlite export)?

Unlike profsplit.py / graphsplit.py (per token or per graph launch), this takes the last WINDOW
seconds of kernel activity, e.g. the generation phase of the last request of a llama-server run
profiled with speculative decoding, where steps and graph launches do not map one to one.

  python tools/windowsplit.py bench/prof/x.sqlite --window 3.5 --skip-end 0.2 [--ms-per-step 45.2]

Prints the share of the window with any kernel running, the time kernels spin waiting for the CPU
(k_zc_recv: the GPU is "busy" but does no work), the idle time and the top kernels. With
--ms-per-step the numbers are also given per step.
"""
import argparse
import sqlite3


def union_len(iv):
    iv.sort()
    tot, cs, ce = 0, None, None
    for s, e in iv:
        if cs is None or s > ce:
            if cs is not None:
                tot += ce - cs
            cs, ce = s, e
        else:
            ce = max(ce, e)
    if cs is not None:
        tot += ce - cs
    return tot


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("db")
    ap.add_argument("--window", type=float, default=3.0, help="seconds")
    ap.add_argument("--skip-end", type=float, default=0.2, help="seconds cut off before the last kernel")
    ap.add_argument("--ms-per-step", type=float, default=0.0)
    ap.add_argument("--top", type=int, default=10)
    a = ap.parse_args()
    db = sqlite3.connect(a.db)
    k = db.execute("SELECT k.start, k.end, s.value FROM CUPTI_ACTIVITY_KIND_KERNEL k JOIN StringIds s ON k.shortName = s.id").fetchall()
    t_last = max(e for _, e, _ in k)
    t1 = t_last - int(a.skip_end * 1e9)
    t0 = t1 - int(a.window * 1e9)
    win = [(max(s, t0), min(e, t1), n) for s, e, n in k if e > t0 and s < t1]
    w = (t1 - t0) / 1e6
    busy = union_len([(s, e) for s, e, _ in win]) / 1e6
    per = {}
    for s, e, n in win:
        per.setdefault(n, [0.0, 0])
        per[n][0] += (e - s) / 1e6
        per[n][1] += 1
    try:
        mc = db.execute("SELECT start, end FROM CUPTI_ACTIVITY_KIND_MEMCPY").fetchall()
        cp = union_len([(max(s, t0), min(e, t1)) for s, e in mc if e > t0 and s < t1]) / 1e6
    except sqlite3.OperationalError:
        cp = 0.0
    spin = per.get("k_zc_recv", [0.0])[0]
    f = (lambda x: f"{x:8.1f} ms  ({100 * x / w:4.1f}%)" + (f"  {x * a.ms_per_step / w:6.2f} ms/step" if a.ms_per_step else ""))
    print(f"window {w:.0f} ms" + (f" = {w / a.ms_per_step:.0f} steps of {a.ms_per_step} ms" if a.ms_per_step else ""))
    print(f"  kernels running       {f(busy)}")
    print(f"    of it k_zc_recv     {f(spin)}   (waiting for the CPU experts)")
    print(f"  GPU idle              {f(w - busy)}")
    print(f"  copies (any)          {f(cp)}")
    print("  top kernels (sum of durations):")
    for n, (t, c) in sorted(per.items(), key=lambda x: -x[1][0])[: a.top]:
        print(f"    {f(t)}  x{c:6d}  {n}")


if __name__ == "__main__":
    main()
