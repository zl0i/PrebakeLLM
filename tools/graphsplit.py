"""Per-step GPU breakdown of the decode phase from an Nsight Systems SQLite export, for any engine that
launches one CUDA graph per decode step (llama.cpp / our engine: one per token; Strata: one per verify window).

  python tools/graphsplit.py bench/prof/<file>.sqlite [--last 400] [--top 12]

Takes the last N cudaGraphLaunch calls (steady-state decode), and over the interval they span reports per launch:
wall time, union of kernel intervals, GPU idle time, kernel count, and the kernels with the most time.
Used to compare our engine with Strata as a black box (DECISIONS D11).
"""
import argparse
import sqlite3


def union_len(iv):
    tot, cs, ce = 0, None, None
    for s, e in sorted(iv):
        if ce is None or s > ce:
            if ce is not None:
                tot += ce - cs
            cs, ce = s, e
        else:
            ce = max(ce, e)
    return tot + ((ce - cs) if ce is not None else 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("db")
    ap.add_argument("--last", type=int, default=400)
    ap.add_argument("--top", type=int, default=12)
    a = ap.parse_args()
    con = sqlite3.connect(a.db)
    q = lambda s: con.execute(s).fetchall()
    names = dict(q("SELECT id, value FROM StringIds"))
    launches = [r[0] for r in q("SELECT r.start FROM CUPTI_ACTIVITY_KIND_RUNTIME r JOIN StringIds s ON r.nameId = s.id "
                                "WHERE s.value LIKE 'cudaGraphLaunch%' ORDER BY r.start")]
    launches = launches[-(a.last + 1):]
    lo, hi = launches[0], launches[-1]
    n = len(launches) - 1
    kern = q(f"SELECT start, end, shortName FROM CUPTI_ACTIVITY_KIND_KERNEL WHERE end > {lo} AND start < {hi}")
    kern = [(max(s, lo), min(e, hi), k) for s, e, k in kern]
    copies = []
    try:
        copies = [(max(s, lo), min(e, hi)) for s, e in q(f"SELECT start, end FROM CUPTI_ACTIVITY_KIND_MEMCPY WHERE end > {lo} AND start < {hi}")]
    except sqlite3.OperationalError:
        pass
    wall = (hi - lo) / n / 1e6
    tk = union_len([(s, e) for s, e, _ in kern]) / n / 1e6
    tb = union_len([(s, e) for s, e, _ in kern] + copies) / n / 1e6
    print(f"{a.db}: last {n} graph launches, {wall:.2f} ms per launch")
    print(f"  kernels running {tk:6.2f} ms/launch ({100 * tk / wall:4.1f}%), {len(kern) / n:.0f} kernels/launch")
    print(f"  copies          {union_len(copies) / n / 1e6:6.2f} ms/launch")
    print(f"  GPU idle        {wall - tb:6.2f} ms/launch")
    agg = {}
    for s, e, k in kern:
        t = agg.setdefault(names.get(k, str(k)), [0, 0])
        t[0] += e - s
        t[1] += 1
    print("  top kernels (ms/launch, count/launch):")
    for k, (t, c) in sorted(agg.items(), key=lambda kv: -kv[1][0])[:a.top]:
        print(f"    {t / n / 1e6:7.3f} {c / n:7.1f}  {k[:90]}")


if __name__ == "__main__":
    main()
