"""Offline expert-cache simulator over traces from tools/trace_experts.

Question it answers: for a given VRAM budget (in experts), how many expert
weights per generated token must come from RAM under different policies?
That number times 1.81 MiB is PCIe traffic per token, the main cost we pay.

Policies:
  lru        global LRU, miss inserted right away (what llama.cpp's cache does; its real
             policy is a frequency-aware LRU variant, so this is an approximation)
  lru_lagK   LRU whose insertions become visible only K tokens later
             (a cache managed by the CPU in the background, off the critical path)
  static     fixed hot set = most frequently used experts on a calibration trace
  static+lru half the budget static hot set, half LRU
  *_mtp3     same, but tokens are processed in groups of 3 (MTP verification:
             the union of experts of 3 tokens is needed at once)

  python tools/cachesim.py bench/traces/eval_set.bin [--calib other.bin] [--mib 6500]
"""
import argparse
import collections
import struct

import numpy as np

EXPERT_MIB = (144 + 144 + 176) / 256  # gate+up (Q4_K) + down (Q5_K) of one expert, MiB


def load(path):
    with open(path, "rb") as f:
        magic, n_layer, n_used = struct.unpack("<3i", f.read(12))
        assert magic == 0x52545845, "bad trace file"
        prompts = []
        while True:
            h = f.read(8)
            if len(h) < 8:
                break
            pi, n = struct.unpack("<2i", h)
            a = np.frombuffer(f.read(4 * n * n_layer * n_used), dtype=np.int32).reshape(n, n_layer, n_used)
            prompts.append((pi, a))
    return n_layer, n_used, prompts


def steps(prompts, group):
    """Yields per decode step the set of (layer, expert) keys needed at once."""
    for _, a in prompts:
        n, n_layer, _ = a.shape
        for t in range(0, n, group):
            blk = a[t:t + group]
            keys = set()
            for il in range(n_layer):
                for e in np.unique(blk[:, il, :]):
                    if e >= 0:
                        keys.add((il, int(e)))
            yield keys


class LRU:
    def __init__(self, cap, lag=0):
        self.cap, self.lag = cap, lag
        self.od = collections.OrderedDict()
        self.pending = collections.deque()  # (step_index_visible, keys)

    def step(self, i, keys):
        while self.pending and self.pending[0][0] <= i:
            for k in self.pending.popleft()[1]:
                self.insert(k)
        miss = [k for k in keys if k not in self.od]
        for k in keys:
            if k in self.od:
                self.od.move_to_end(k)
        if self.lag == 0:
            for k in miss:
                self.insert(k)
        else:
            self.pending.append((i + self.lag, miss))
        return len(miss)

    def insert(self, k):
        if k in self.od:
            self.od.move_to_end(k)
            return
        self.od[k] = True
        if len(self.od) > self.cap:
            self.od.popitem(last=False)


class LRUAdmit:
    """Zero-copy engine: a miss is read over PCIe directly by the GPU kernel.
    Copying it into the VRAM cache costs PCIe traffic a second time, so only
    admit an expert after it missed `k` times within the last `window` tokens.
    Insertions become visible one token later (managed by the CPU in background).
    Reports traffic = misses read + insertions uploaded."""

    def __init__(self, cap, k=2, window=8, lag=1):
        self.lru = LRU(cap, lag)
        self.k, self.window = k, window
        self.recent = collections.defaultdict(collections.deque)
        self.inserted = 0

    def step(self, i, keys):
        lru = self.lru
        while lru.pending and lru.pending[0][0] <= i:
            for key in lru.pending.popleft()[1]:
                lru.insert(key)
        miss = [key for key in keys if key not in lru.od]
        for key in keys:
            if key in lru.od:
                lru.od.move_to_end(key)
        admit = []
        for key in miss:
            dq = self.recent[key]
            dq.append(i)
            while dq and dq[0] <= i - self.window:
                dq.popleft()
            if len(dq) >= self.k:
                admit.append(key)
                dq.clear()
        self.inserted += len(admit)
        lru.pending.append((i + lru.lag, admit))
        return len(miss)


class Static:
    def __init__(self, hot):
        self.hot = hot

    def step(self, i, keys):
        return sum(1 for k in keys if k not in self.hot)


class StaticPlusLRU:
    def __init__(self, hot, cap_lru, lag=0):
        self.hot, self.lru = hot, LRU(cap_lru, lag)

    def step(self, i, keys):
        rest = {k for k in keys if k not in self.hot}
        return self.lru.step(i, rest)


def hot_set(prompts, cap):
    cnt = collections.Counter()
    for _, a in prompts:
        n, n_layer, n_used = a.shape
        for il in range(n_layer):
            ids, c = np.unique(a[:, il, :], return_counts=True)
            for e, k in zip(ids, c):
                cnt[(il, int(e))] += int(k)
    return {k for k, _ in cnt.most_common(cap)}


def run(policy, prompts, group, warm=0):
    """Returns (misses per generated token, needed experts per token)."""
    misses, need, ntok = 0, 0, 0
    for i, keys in enumerate(steps(prompts, group)):
        m = policy.step(i, keys)
        if i >= warm:
            misses += m
            need += len(keys)
            ntok += group
    return misses / ntok, need / ntok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("trace")
    ap.add_argument("--calib", help="trace for choosing the static hot set (default: same trace = optimistic)")
    ap.add_argument("--mib", type=float, nargs="+", default=[5500, 6500, 7500])
    ap.add_argument("--skip-first", type=int, default=1, help="prompts used only to warm caches (warmup prompt)")
    a = ap.parse_args()
    n_layer, n_used, prompts = load(a.trace)
    calib = load(a.calib)[2] if a.calib else prompts
    total = n_layer * 256
    warm_steps = sum(p[1].shape[0] for p in prompts[:a.skip_first])
    print(f"trace: {len(prompts)} prompts, {sum(p[1].shape[0] for p in prompts)} tokens, {n_layer} layers x {n_used} experts")
    print(f"static hot set chosen on: {'calibration trace ' + a.calib if a.calib else 'the SAME trace (optimistic)'}")
    for mib in a.mib:
        cap = int(mib / EXPERT_MIB)
        print(f"\n== VRAM cache {mib:.0f} MiB = {cap} experts ({100 * cap / total:.0f}% of all {total})")
        hot = hot_set(calib, cap)
        hot_half = hot_set(calib, cap // 2)
        for group in (1, 3):
            sfx = "" if group == 1 else "_mtp3"
            ws = warm_steps // group
            res = [
                ("lru" + sfx, run(LRU(cap), prompts, group, ws)),
                ("lru_lag1" + sfx, run(LRU(cap, 1), prompts, group, ws)),
                ("lru_lag2" + sfx, run(LRU(cap, 2), prompts, group, ws)),
                ("static" + sfx, run(Static(hot), prompts, group, ws)),
                ("static+lru" + sfx, run(StaticPlusLRU(hot_half, cap - cap // 2), prompts, group, ws)),
            ]
            for name, (m, need) in res:
                print(f"  {name:16s} misses/token {m:6.1f} of {need:5.1f} needed  hit {100 * (1 - m / need):5.1f}%"
                      f"  -> {m * EXPERT_MIB:6.1f} MiB/token from RAM")
            # zero-copy engine: PCIe traffic = misses read by kernels + experts uploaded into the cache
            for k, w in ((1, 1), (2, 4), (2, 8), (2, 16), (3, 16)):
                p = LRUAdmit(cap, k, w)
                ntok = sum(pp[1].shape[0] for pp in prompts) - warm_steps
                m, need = run(p, prompts, group, ws)
                ins = p.inserted / (ntok + warm_steps)
                print(f"  zc_admit{k}of{w:<3d}{sfx:5s} misses/token {m:6.1f}  hit {100 * (1 - m / need):5.1f}%  inserts/token {ins:5.1f}"
                      f"  -> PCIe {(m + ins) * EXPERT_MIB:6.1f} MiB/token (lru for comparison {res[0][1][0] * EXPERT_MIB:5.1f})")


if __name__ == "__main__":
    main()
