"""Seed noise and how many seeds are needed to tell two agent configs apart.

Model: each task t has resolve probability p_t (estimated from the seeds). A single run's
score is a sum of independent Bernoullis, so Var(score) = sum p_t (1 - p_t). For two configs
compared on the same tasks with n seeds each, Var(mean diff) = (VarA + VarB) / n.

Usage: python noise.py name=features.jsonl [name=features.jsonl ...]
"""
import json, math, statistics as st, sys
from collections import defaultdict

V = {}
for arg in sys.argv[1:]:
    name, path = arg.split("=")
    by = defaultdict(list)
    for l in open(path):
        r = json.loads(l)
        by[r["task"]].append(r["resolved"])
    k = len(next(iter(by.values())))
    p = {t: sum(v) / len(v) for t, v in by.items()}
    # unbiased per-task Bernoulli variance: p(1-p) * k/(k-1)
    var = sum(q * (1 - q) for q in p.values()) * k / (k - 1)
    per_seed = [sum(v[i] for v in by.values()) for i in range(k)]
    V[name] = var
    flip = sum(0 < q < 1 for q in p.values())
    print(f"{name}: {k} seeds, mean {st.mean(per_seed):.1f}, observed SD {st.stdev(per_seed):.2f}, "
          f"model SD {math.sqrt(var):.2f} tasks ({math.sqrt(var) / len(p):.1%}); flipping tasks {flip}")

print("\nMinimum detectable difference (95% two-sided, 80% power) between two configs, tasks out of 118:")
names = list(V)
for n in (1, 2, 3, 5, 10):
    row = []
    for a in names:
        for b in names[names.index(a):]:
            sd = math.sqrt((V[a] + V[b]) / n)
            row.append(f"{a}-vs-{b} {2.8 * sd:4.1f}")
    print(f"  n={n:2d} seeds/arm: " + " | ".join(row))
ints = ", ".join(f"{a} {1.96 * math.sqrt(V[a]):.1f}" for a in names)
print(f"\nSingle leaderboard run: 95% interval of one config's score = +/- {ints} tasks")
