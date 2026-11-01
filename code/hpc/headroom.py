"""Headroom gate: how much a perfect selector over k runs could gain over one run."""
import itertools, json, statistics as st, sys, csv
from pathlib import Path

run = Path(sys.argv[1])
seeds = sorted(p for p in run.glob("seed*") if (p / "task_results.jsonl").exists())
R = {}  # task -> {seed: result}
for s in seeds:
    for l in open(s / "task_results.jsonl"):
        r = json.loads(l)
        R.setdefault(r["instance_id"], {})[s.name] = r
tasks = sorted(R)
names = [s.name for s in seeds]
complete = [t for t in tasks if all(n in R[t] for n in names)]
print(f"seeds: {names}; tasks with all seeds: {len(complete)}/{len(tasks)}")
M = {t: [bool(R[t][n]["resolved"]) for n in names] for t in complete}
n, k_max = len(complete), len(names)

per_seed = [sum(M[t][i] for t in complete) for i in range(k_max)]
print("solved per seed:", per_seed, f"mean {st.mean(per_seed):.1f} sd {st.pstdev(per_seed):.1f}",
      f"-> pass@1 {st.mean(per_seed)/n:.3f}")
for k in range(1, k_max + 1):
    vals = [sum(any(M[t][i] for i in c) for t in complete) for c in itertools.combinations(range(k_max), k)]
    print(f"oracle any-of-{k}: {st.mean(vals):.1f} tasks ({st.mean(vals)/n:.3f})  range {min(vals)}-{max(vals)}")
always = sum(all(v) for v in M.values()); never = sum(not any(v) for v in M.values())
flaky = n - always - never
print(f"always solved {always}, never {never}, sometimes {flaky} ({flaky/n:.1%} of tasks flip across seeds)")
gain = st.mean([sum(any(M[t]) for t in complete)]) - st.mean(per_seed)
print(f"headroom (oracle any-of-{k_max} minus mean single run): +{gain:.1f} tasks")

by_repo = {}
for t in complete:
    repo = R[t][names[0]]["repo"]
    a = by_repo.setdefault(repo, [0, 0, 0])
    a[0] += 1; a[1] += sum(M[t]) / k_max; a[2] += any(M[t])
for repo, (c, m, o) in sorted(by_repo.items()):
    print(f"  {repo:18s} n={c:3d} mean solved {m:5.1f}  any-of-{k_max} {o:3d}")
dur = [float(R[t][s]["duration_seconds"]) for t in complete for s in names]
calls = [int(R[t][s]["tool_calls"] or 0) for t in complete for s in names]
print(f"time/task: mean {st.mean(dur):.0f}s, p95 {sorted(dur)[int(.95*len(dur))]:.0f}s; tool calls mean {st.mean(calls):.1f}")
errs = [(t, s, R[t][s]["error"]) for t in complete for s in names if R[t][s].get("error")]
print(f"runs with harness errors: {len(errs)}", errs[:5])
with open(run / "matrix.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["task"] + names)
    for t in complete:
        w.writerow([t] + [int(x) for x in M[t]])
