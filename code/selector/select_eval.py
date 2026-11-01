"""Evaluate test-free selectors over k runs per task.

A selector scores each candidate run of a task using only run artifacts (patch, trace);
the highest score wins and ties are broken uniformly at random, so we report the exact
expected number of resolved tasks. Labels (`resolved`) are used only for scoring.
Results are averaged over every k-subset of the 5 seeds.

Usage: python select_eval.py features.jsonl [extra_scores.jsonl ...]
"""
import itertools, json, random, statistics as st, sys
from collections import defaultdict

R = [json.loads(l) for l in open(sys.argv[1])]
EXTRA = {}  # (task, seed) -> {name: score} from model-judge files
for p in sys.argv[2:]:
    for l in open(p):
        e = json.loads(l)
        EXTRA.setdefault((e["task"], e["seed"]), {}).update(e["scores"])
by_task = defaultdict(dict)
for r in R:
    r.update(EXTRA.get((r["task"], r["seed"]), {}))
    by_task[r["task"]][r["seed"]] = r
TASKS = sorted(by_task)
SEEDS = sorted({r["seed"] for r in R})


def valid(r):
    return r["n_src_files"] > 0 and not r["error"]


def jacc(a, b):
    a, b = set(a), set(b)
    return len(a & b) / len(a | b) if a | b else 0.0


def consensus(r, group, key="src_lines"):
    """Mean similarity of r's changed lines to the other valid candidates."""
    others = [o for o in group if o is not r and valid(o)]
    return st.mean(jacc(r[key], o[key]) for o in others) if others else 0.0


def file_consensus(r, group):
    others = [o for o in group if o is not r and valid(o)]
    return st.mean(jacc(r["src_files"], o["src_files"]) for o in others) if others else 0.0


# Each selector: (group of candidate runs) -> list of scores. Higher is better.
def S(f):
    return lambda g: [f(r, g) for r in g]


SELECTORS = {
    "random": S(lambda r, g: 0),
    "valid": S(lambda r, g: valid(r)),
    "valid+tested_ok": S(lambda r, g: (valid(r), r.get("last_run_ok") is True)),
    "valid+consensus": S(lambda r, g: (valid(r), consensus(r, g))),
    "valid+file_consensus": S(lambda r, g: (valid(r), file_consensus(r, g))),
    "valid+small": S(lambda r, g: (valid(r), -r["patch_size"])),
    "valid+few_calls": S(lambda r, g: (valid(r), -(r.get("n_calls") or 99))),
    "valid+ran_ok+consensus": S(lambda r, g: (valid(r), r.get("last_run_ok") is True, consensus(r, g))),
}
if EXTRA:
    SELECTORS["valid+ran_ok+judge_list"] = S(lambda r, g: (valid(r), r.get("last_run_ok") is True, r.get("judge_list", 0)))
    SELECTORS["valid+ran_ok+small"] = S(lambda r, g: (valid(r), r.get("last_run_ok") is True, -r["patch_size"]))
for name in sorted({k for e in EXTRA.values() for k in e}):
    SELECTORS[f"valid+{name}"] = S(lambda r, g, n=name: (valid(r), r.get(n, 0)))
    SELECTORS[f"valid+{name}+consensus"] = S(lambda r, g, n=name: (valid(r), r.get(n, 0) + consensus(r, g)))


def expected(sel, seeds):
    tot, per = 0.0, {}
    for t in TASKS:
        g = [by_task[t][s] for s in seeds]
        sc = sel(g)
        best = max(sc)
        win = [r for r, s in zip(g, sc) if s == best]
        per[t] = sum(r["resolved"] for r in win) / len(win)
        tot += per[t]
    return tot, per


def boot_ci(per_a, per_b, n=2000, seed=0):
    rng = random.Random(seed)
    d = [per_a[t] - per_b[t] for t in TASKS]
    xs = sorted(sum(rng.choice(d) for _ in d) for _ in range(n))
    return xs[int(.025 * n)], xs[int(.975 * n)]


print(f"{len(TASKS)} tasks x {len(SEEDS)} seeds")
for k in range(2, len(SEEDS) + 1):
    combos = list(itertools.combinations(SEEDS, k))
    oracle = st.mean(sum(any(by_task[t][s]["resolved"] for s in c) for t in TASKS) for c in combos)
    single = st.mean(sum(by_task[t][s]["resolved"] for t in TASKS) for s in SEEDS)
    print(f"\nk={k}: single run {single:.1f}, oracle {oracle:.1f}, headroom {oracle - single:.1f}")
    print(f"  {'selector':32s} {'solved':>7s} {'gain':>6s} {'%headroom':>9s}")
    res = {}
    for name, sel in SELECTORS.items():
        vals = [expected(sel, c) for c in combos]
        res[name] = vals
        m = st.mean(v[0] for v in vals)
        print(f"  {name:32s} {m:7.1f} {m - single:+6.1f} {(m - single) / (oracle - single):9.0%}")
    if k == len(SEEDS):
        base = res["random"][0][1]
        print("  95% bootstrap CI of gain over random (k=5):")
        for name, vals in res.items():
            lo, hi = boot_ci(vals[0][1], base)
            print(f"    {name:30s} [{lo:+.1f}, {hi:+.1f}]")
