"""Combine per-repo Gate 1 runs into one per-task soundness table and compare with community audits."""
import json, sys, glob, os, csv, collections
runs = sys.argv[1]; out = sys.argv[2]; comm = sys.argv[3]; comm2 = sys.argv[4]
def latest(mode):
    res = {}
    for d in sorted(glob.glob(f"{runs}/gate1-{mode}-*"), key=os.path.getmtime):
        p = f"{d}/task_results.jsonl"
        if os.path.exists(p):
            for l in open(p):
                r = json.loads(l); res[r["instance_id"]] = r
    return res
g, e = latest("gold"), latest("empty")
C = {r["task_id"]: r for r in csv.DictReader(open(comm))}
Z = {r["instance_id"]: r for r in csv.DictReader(open(comm2))}
rows = []
for tid in sorted(set(g) | set(e)):
    gr, er = g.get(tid, {}), e.get(tid, {})
    gold_ok, empty_ok = bool(gr.get("resolved")), bool(er.get("resolved"))
    status = "sound" if gold_ok and not empty_ok else ("passes_without_fix" if empty_ok else "gold_fails")
    rows.append(dict(task_id=tid, repo=gr.get("repo") or er.get("repo"), status=status,
                     gold_resolved=int(gold_ok), empty_resolved=int(empty_ok),
                     gold_exit=gr.get("test_exit_code"), gold_error=(gr.get("error") or "")[:150],
                     community_status=C.get(tid, {}).get("status", "?"),
                     community2_gold=Z.get(tid, {}).get("gold_passes_scorer_like", "?")))
with open(out, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
print("tasks:", len(rows), collections.Counter(r["status"] for r in rows))
print("by repo:", collections.Counter((r["repo"], r["status"]) for r in rows))
agree = sum((r["status"] == "sound") == (r["community_status"] == "sound") for r in rows)
print(f"agreement with community soundness: {agree}/{len(rows)}")
for r in rows:
    if (r["status"] == "sound") != (r["community_status"] == "sound"):
        print("  DIFF", r["task_id"], "ours=", r["status"], "theirs=", r["community_status"], "| gold_err:", r["gold_error"][:90])
