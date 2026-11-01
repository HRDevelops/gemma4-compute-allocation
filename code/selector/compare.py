"""Fixed-compute comparison: k short runs + test-free selector vs long-budget runs.

Cost per task is measured from the runs themselves (wall seconds, LLM tokens), so
"matched compute" uses what was actually spent, not the nominal budgets.

Usage: python compare.py short_features.jsonl long_features.jsonl
"""
import itertools, json, random, statistics as st, sys
from collections import defaultdict


def load(p):
    by = defaultdict(dict)
    for l in open(p):
        r = json.loads(l)
        by[r["task"]][r["seed"]] = r
    return by


S, L = load(sys.argv[1]), load(sys.argv[2])
TASKS = sorted(t for t in S if t in L and len(L[t]) == max(len(v) for v in L.values()))
SS = sorted({s for t in TASKS for s in S[t]}); LS = sorted({s for t in TASKS for s in L[t]})


def valid(r):
    return r["n_src_files"] > 0 and not r["error"]


def rule(r):  # the selector fixed after the selector gate
    return (valid(r), r.get("last_run_ok") is True, -r["patch_size"])


def pick(group, sel):
    sc = [sel(r) for r in group]
    b = max(sc)
    win = [r for r, s in zip(group, sc) if s == b]
    return sum(r["resolved"] for r in win) / len(win)


def cost(rs):
    return dict(wall=sum(r["duration"] for r in rs), gen=sum(r.get("completion_tokens", 0) for r in rs),
                tok=sum(r.get("tokens", 0) for r in rs), calls=sum(r.get("n_calls", 0) or 0 for r in rs))


def row(name, by, seeds, k, sel):
    combos = list(itertools.combinations(seeds, k))
    solved, costs, per_task = [], defaultdict(list), defaultdict(float)
    for c in combos:
        tot = 0
        for t in TASKS:
            g = [by[t][s] for s in c]
            v = pick(g, sel) if sel else g[0]["resolved"]
            tot += v; per_task[t] += v / len(combos)
        solved.append(tot)
        cc = cost([by[t][s] for t in TASKS for s in c])
        for kk, vv in cc.items():
            costs[kk].append(vv / len(TASKS))
    m = {kk: st.mean(v) for kk, v in costs.items()}
    print(f"{name:34s} {st.mean(solved):6.1f} {min(solved):4.0f}-{max(solved):<4.0f} "
          f"{m['wall'] / 60:8.1f} {m['gen'] / 1e3:8.1f} {m['tok'] / 1e3:8.0f} {m['calls']:7.1f}")
    return per_task


print(f"{len(TASKS)} tasks; short seeds {SS}; long seeds {LS}")
print(f"{'condition':34s} {'solved':>6s} {'range':>9s} {'min/task':>8s} {'kgen/tsk':>8s} {'ktok/tsk':>8s} {'calls':>7s}")
res = {}
for k in range(1, len(SS) + 1):
    res[f"short x{k}"] = row(f"short x{k} + selector" if k > 1 else "short x1", S, SS, k, rule if k > 1 else None)
for k in range(1, len(LS) + 1):
    res[f"long x{k}"] = row(f"long x{k} + selector" if k > 1 else "long x1", L, LS, k, rule if k > 1 else None)
oracle = {t: float(any(S[t][s]["resolved"] for s in SS)) for t in TASKS}
print(f"{'short oracle any-of-' + str(len(SS)):34s} {sum(oracle.values()):6.1f}")
print(f"{'long oracle any-of-' + str(len(LS)):34s} {sum(any(L[t][s]['resolved'] for s in LS) for t in TASKS):6.1f}")

# Paired bootstrap: best-of-5 short + selector minus a single long run.
a, b = res[f"short x{len(SS)}"], res["long x1"]
rng = random.Random(0)
d = [a[t] - b[t] for t in TASKS]
xs = sorted(sum(rng.choice(d) for _ in d) for _ in range(4000))
print(f"\nshort x{len(SS)}+selector minus long x1: {sum(d):+.1f} tasks, 95% CI [{xs[100]:+.1f}, {xs[3899]:+.1f}]")
# Where does the long budget get spent?
lr = [L[t][s] for t in TASKS for s in LS]
print("long runs: hit limit", sum(bool(r["error"]) for r in lr), "/", len(lr),
      "| duration p50/p90 %.0f/%.0f s" % (st.median(r["duration"] for r in lr), sorted(r["duration"] for r in lr)[int(.9 * len(lr))]),
      "| tool calls p50/p90 %d/%d" % (st.median(r["n_calls"] or 0 for r in lr), sorted(r["n_calls"] or 0 for r in lr)[int(.9 * len(lr))]),
      "| runs >28 calls", sum((r["n_calls"] or 0) > 28 for r in lr), "| >8 min", sum(r["duration"] > 480 for r in lr))
