"""Matched-cost paired comparisons (bootstrap over tasks) between regimes.

Per-task value = expected resolve with the pre-registered R-short selector, averaged
over all k-subsets of that regime's seeds.
Usage: python paired_tests.py <results_dir>
"""
import itertools, json, random, sys
from collections import defaultdict
from pathlib import Path

R = Path(sys.argv[1])


def load(*names):
    by = defaultdict(dict)
    for i, n in enumerate(names):
        for l in open(R / n / "features.jsonl"):
            r = json.loads(l)
            by[r["task"]][f"{i}:{r['seed']}"] = r
    return by


def valid(r):
    return r["n_src_files"] > 0 and not r["error"]


def rs(r):
    return (valid(r), r.get("last_run_ok") is True, -r["patch_size"])


def per_task(by, k, max_combos=200):
    seeds = sorted({s for v in by.values() for s in v})
    combos = list(itertools.combinations(seeds, k))
    if len(combos) > max_combos:
        combos = random.Random(0).sample(combos, max_combos)
    out = {}
    for t in by:
        tot = 0
        for c in combos:
            g = [by[t][s] for s in c]
            sc = [rs(r) for r in g]
            b = max(sc)
            w = [r for r, v in zip(g, sc) if v == b]
            tot += sum(r["resolved"] for r in w) / len(w)
        out[t] = tot / len(combos)
    return out


S = load("baseline", "baseline_holdout")
M = load("baseline_medium")
L = load("baseline_long")
T = load("long_thorough")


def cmp(a, b, name):
    d = [a[t] - b[t] for t in a]
    rng = random.Random(0)
    xs = sorted(sum(rng.choice(d) for _ in d) for _ in range(4000))
    print(f"{name:52s} {sum(d):+5.1f}  [{xs[100]:+.1f}, {xs[3899]:+.1f}]")


cmp(per_task(L, 1), per_task(S, 2), "long x1 - short x2+R (~7.5 min)")
cmp(per_task(T, 1), per_task(S, 2), "thorough x1 - short x2+R (~7.3 min)")
cmp(per_task(M, 1), per_task(S, 2), "medium x1 - short x2+R (medium cheaper)")
cmp(per_task(L, 2), per_task(S, 4), "long x2+R - short x4+R (~15 min)")
cmp(per_task(T, 2), per_task(S, 4), "thorough x2+R - short x4+R (~15 min)")
cmp(per_task(L, 1), per_task(S, 5), "long x1 - short x5+R (short 2.6x cost)")
cmp(per_task(T, 1), per_task(L, 1), "thorough x1 - long x1 (same cost)")
cmp(per_task(M, 1), per_task(S, 1), "medium x1 - short x1")
# Pooled: average of the matched-cost long-vs-short and thorough-vs-short gaps at ~7.5 and ~15 min
a = {t: (per_task(L, 1)[t] + per_task(T, 1)[t]) / 2 for t in S}
cmp(a, per_task(S, 2), "budget (long & thorough, x1) - short x2+R, pooled")
