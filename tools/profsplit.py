"""Breaks down where the time of one generated token goes, from an Nsight
Systems trace exported to SQLite (nsys profile --export=sqlite
--cuda-graph-trace=node ...).

Decode steps are found as the repeating cudaGraphLaunch bursts at the end of
the run; we take the last --tokens steady-state tokens by wall time and
report per token:
  - GPU kernel time (union of kernel intervals),
  - PCIe copy time (host->device and device->host),
  - GPU idle time (neither kernels nor copies running),
  - count of kernels, copies and CPU<->GPU synchronizations.

  python tools/profsplit.py bench/prof/cache7500.sqlite --tokens 200 --n-gen 300
"""
import argparse
import sqlite3


def union_len(intervals, lo, hi):
    """Total length covered by [s,e) intervals, clipped to [lo,hi)."""
    total, cur_s, cur_e = 0, None, None
    for s, e in sorted((max(s, lo), min(e, hi)) for s, e in intervals if e > lo and s < hi):
        if cur_e is None or s > cur_e:
            if cur_e is not None:
                total += cur_e - cur_s
            cur_s, cur_e = s, e
        else:
            cur_e = max(cur_e, e)
    if cur_e is not None:
        total += cur_e - cur_s
    return total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("db")
    ap.add_argument("--tokens", type=int, default=200, help="steady-state tokens to analyse (taken from the end)")
    ap.add_argument("--n-gen", type=int, default=300, help="tokens generated in the run")
    a = ap.parse_args()
    db = sqlite3.connect(a.db)
    q = lambda sql: db.execute(sql).fetchall()

    kern = q("SELECT start, end FROM CUPTI_ACTIVITY_KIND_KERNEL")
    names = dict(q("SELECT id, value FROM StringIds"))
    kern_named = q("SELECT start, end, shortName FROM CUPTI_ACTIVITY_KIND_KERNEL")
    mem = q("SELECT start, end, copyKind, bytes FROM CUPTI_ACTIVITY_KIND_MEMCPY")
    rt = q("SELECT r.start, r.end, s.value FROM CUPTI_ACTIVITY_KIND_RUNTIME r JOIN StringIds s ON r.nameId = s.id")

    # The end of the run = last kernel; decode ran over the last n_gen tokens.
    # Find the window: time of the last `tokens` tokens ~ proportional slice of
    # the decode phase, which starts after the last large prompt kernel burst.
    # Simplest robust approach: decode tokens are delimited by the D2H copy of
    # the sampled logits (one large-ish D2H per token). Use output-sized D2H copies.
    d2h = sorted((s, e, b) for s, e, k, b in mem if k == 2)
    # logits copy: vocab 248320 floats ~ 0.99 MB
    marks = [s for s, e, b in d2h if b >= 900_000]
    if len(marks) < a.tokens + 2:
        raise SystemExit(f"found only {len(marks)} logits copies")
    marks = marks[-(a.tokens + 1):]
    lo, hi = marks[0], marks[-1]
    n = len(marks) - 1
    wall = hi - lo

    k_iv = [(s, e) for s, e in kern if s < hi and e > lo]
    h2d = [(s, e, b) for s, e, k, b in mem if k == 1 and s < hi and e > lo]
    d2h_w = [(s, e, b) for s, e, k, b in mem if k == 2 and s < hi and e > lo]
    t_kern = union_len(k_iv, lo, hi)
    t_h2d = union_len([(s, e) for s, e, _ in h2d], lo, hi)
    t_d2h = union_len([(s, e) for s, e, _ in d2h_w], lo, hi)
    t_busy = union_len(k_iv + [(s, e) for s, e, _ in h2d + d2h_w], lo, hi)
    syncs = [r for r in rt if lo <= r[0] < hi and "Synchronize" in r[2]]
    t_sync = sum(e - s for s, e, _ in syncs)

    ms = lambda x: x / n / 1e6
    print(f"window: {n} tokens, {wall / 1e9:.3f} s  ->  {ms(wall):.2f} ms/token = {n / (wall / 1e9):.1f} tok/s")
    print(f"  GPU kernels running      {ms(t_kern):6.2f} ms/token  ({100 * t_kern / wall:4.1f}%)  {len(k_iv) / n:.0f} kernels/token")
    print(f"  PCIe host->device copies {ms(t_h2d):6.2f} ms/token  ({100 * t_h2d / wall:4.1f}%)  "
          f"{len(h2d) / n:.0f} copies/token, {sum(b for *_, b in h2d) / n / 2**20:.1f} MiB/token")
    print(f"  PCIe device->host copies {ms(t_d2h):6.2f} ms/token  ({100 * t_d2h / wall:4.1f}%)  {len(d2h_w) / n:.0f} copies/token")
    print(f"  GPU idle (no kernel/copy){ms(wall - t_busy):6.2f} ms/token  ({100 * (wall - t_busy) / wall:4.1f}%)")
    print(f"  CPU blocked in *Synchronize calls: {ms(t_sync):.2f} ms/token, {len(syncs) / n:.0f} calls/token")

    # top kernels in the window
    agg = {}
    for s, e, nm in kern_named:
        if lo <= s < hi:
            k = names.get(nm, str(nm))
            c = agg.setdefault(k, [0, 0])
            c[0] += e - s
            c[1] += 1
    print("  top kernels (ms/token, count/token):")
    for k, (t, c) in sorted(agg.items(), key=lambda x: -x[1][0])[:12]:
        print(f"    {ms(t):6.3f}  {c / n:6.1f}  {k[:90]}")


if __name__ == "__main__":
    main()
