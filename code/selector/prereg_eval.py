"""Holdout evaluation of the rules frozen in paper/preregistration.md.

Usage: python prereg_eval.py short_holdout_features.jsonl long_features.jsonl
"""
import itertools, json, random, statistics as st, sys
from collections import defaultdict


def load(p):
    by = defaultdict(dict)
    for l in open(p):
        r = json.loads(l)
        by[r["task"]][r["seed"]] = r
    return by


def valid(r):
    return r["n_src_files"] > 0 and not r["error"]


def jacc(a, b):
    a, b = set(a), set(b)
    return len(a & b) / len(a | b) if a | b else 0.0


def r_short(r, g):
    return (valid(r), r.get("last_run_ok") is True, -r["patch_size"])


def r_long(r, g):
    others = [o for o in g if o is not r and valid(o)]
    cons = st.mean(jacc(r["src_lines"], o["src_lines"]) for o in others) if others else 0.0
    return (valid(r), r.get("last_run_ok") is True, cons)


def evaluate(by, combos, rule):
    """Expected solved per combo (ties uniform), mean single-run, oracle, per-task gain over random."""
    tasks = sorted(by)
    sel, rnd, orc = [], [], []
    per = defaultdict(float)
    for c in combos:
        s = r_ = o = 0.0
        for t in tasks:
            g = [by[t][x] for x in c]
            sc = [rule(r, g) for r in g]
            b = max(sc)
            win = [r for r, v in zip(g, sc) if v == b]
            a = sum(r["resolved"] for r in win) / len(win)
            m = sum(r["resolved"] for r in g) / len(g)
            s += a; r_ += m; o += any(r["resolved"] for r in g)
            per[t] += (a - m) / len(combos)
        sel.append(s); rnd.append(r_); orc.append(o)
    rng = random.Random(0)
    d = list(per.values())
    xs = sorted(sum(rng.choice(d) for _ in d) for _ in range(4000))
    sm, rm, om = st.mean(sel), st.mean(rnd), st.mean(orc)
    return f"single {rm:5.1f}  selected {sm:5.1f}  gain {sm - rm:+5.1f} [{xs[100]:+.1f}, {xs[3899]:+.1f}]  oracle {om:5.1f}  share {(sm - rm) / (om - rm):4.0%}"


S, L = load(sys.argv[1]), load(sys.argv[2])
ss = sorted({s for v in S.values() for s in v})
print(f"R-short on holdout short seeds {ss}")
for k in range(2, len(ss) + 1):
    print(f"  k={k}: " + evaluate(S, list(itertools.combinations(ss, k)), r_short))
ls = sorted({s for v in L.values() for s in v})
hold = {"seed3", "seed4"}
print(f"R-long on long seeds {ls} (holdout = {sorted(hold)})")
print("  k=2 {3,4}:            " + evaluate(L, [("seed3", "seed4")], r_long))
for k in (2, 3):
    cs = [c for c in itertools.combinations(ls, k) if hold & set(c)]
    print(f"  k={k} with holdout ({len(cs)} sets): " + evaluate(L, cs, r_long))
print("For reference, R-short applied to long runs:")
for k in (2, 3):
    cs = [c for c in itertools.combinations(ls, k) if hold & set(c)]
    print(f"  k={k} with holdout: " + evaluate(L, cs, r_short))
