"""Cost-vs-solved frontier across budget regimes and sample counts.

For each regime and k, average over all k-subsets of that regime's seeds:
  solved with (a) one run, (b) validity filter + random, (c) pre-registered R-short;
  cost = measured wall-minutes, generated tokens, tool calls per task, summed over the k runs.
Error bars: SD of the per-subset solved count (spread across seed draws).

Usage: python frontier.py name=f1.jsonl[,f2.jsonl] ... --csv out.csv
"""
import csv, itertools, json, random, statistics as st, sys
from collections import defaultdict

args = [a for a in sys.argv[1:] if not a.startswith("--")]
out_csv = sys.argv[sys.argv.index("--csv") + 1] if "--csv" in sys.argv else None
args = [a for a in args if a != out_csv]


def valid(r):
    return r["n_src_files"] > 0 and not r["error"]


RULES = {
    "valid": lambda r: valid(r),
    "R-short": lambda r: (valid(r), r.get("last_run_ok") is True, -r["patch_size"]),
}
rows = []
for a in args:
    name, paths = a.split("=")
    by = defaultdict(dict)
    for i, p in enumerate(paths.split(",")):
        for l in open(p):
            r = json.loads(l)
            by[r["task"]][f"{i}:{r['seed']}"] = r
    tasks = sorted(by)
    seeds = sorted({s for v in by.values() for s in v})
    for k in range(1, min(5, len(seeds)) + 1):
        combos = list(itertools.combinations(seeds, k))
        if len(combos) > 300:
            combos = random.Random(0).sample(combos, 300)
        res = defaultdict(list)
        for c in combos:
            runs = [by[t][s] for t in tasks for s in c]
            res["minutes"].append(sum(r["duration"] for r in runs) / 60 / len(tasks))
            res["kgen"].append(sum(r.get("completion_tokens", 0) for r in runs) / 1e3 / len(tasks))
            res["calls"].append(sum(r.get("n_calls") or 0 for r in runs) / len(tasks))
            res["oracle"].append(sum(any(by[t][s]["resolved"] for s in c) for t in tasks))
            res["single"].append(sum(sum(by[t][s]["resolved"] for s in c) / k for t in tasks))
            for rn, f in RULES.items():
                tot = 0.0
                for t in tasks:
                    g = [by[t][s] for s in c]
                    sc = [f(r) for r in g]
                    b = max(sc)
                    w = [r for r, v in zip(g, sc) if v == b]
                    tot += sum(r["resolved"] for r in w) / len(w)
                res[rn].append(tot)
        row = {"regime": name, "k": k, "n_seeds": len(seeds)}
        for kk, v in res.items():
            row[kk] = round(st.mean(v), 2)
            if kk in ("single", "valid", "R-short", "oracle"):
                row[kk + "_sd"] = round(st.pstdev(v), 2) if k > 1 else round(st.pstdev(res["single"]) if False else 0, 2)
        if k == 1:  # seed-to-seed SD of a single run
            row["single_sd"] = round(st.pstdev(res["single"]), 2)
        rows.append(row)

hdr = ["regime", "k", "n_seeds", "minutes", "kgen", "calls", "single", "single_sd", "valid", "R-short", "R-short_sd", "oracle"]
print(" ".join(f"{h:>9s}" for h in hdr))
for r in rows:
    print(" ".join(f"{str(r.get(h, '')):>9s}" for h in hdr))
if out_csv:
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=sorted({k for r in rows for k in r}, key=lambda x: (x not in hdr, hdr.index(x) if x in hdr else 0, x)))
        w.writeheader(); w.writerows(rows)
