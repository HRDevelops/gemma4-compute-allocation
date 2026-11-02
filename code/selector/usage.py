"""Per-regime usage: duration, tool calls, limit hits, invalid runs, generated tokens, per-repo solved.

Usage: python usage.py name=f1.jsonl[,f2.jsonl] ...
"""
import json, sys
from collections import Counter, defaultdict


def pct(xs, q):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(q * len(xs)))]


def limit(err):
    e = err.lower()
    if not e:
        return "none"
    if "timeout" in e or "time" in e:
        return "time"
    if "call" in e:
        return "calls"
    if "turn" in e:
        return "turns"
    return "other"


for a in sys.argv[1:]:
    name, paths = a.split("=")
    rs = [json.loads(l) for p in paths.split(",") for l in open(p)]
    inv = [r for r in rs if not (r["n_src_files"] > 0 and not r["error"])]
    nseed = len({(p, r["seed"]) for p in paths.split(",") for r in map(json.loads, open(p))})
    rep = defaultdict(float)
    for r in rs:
        rep[r["repo"].split("/")[1]] += r["resolved"] / nseed
    d = [r["duration"] for r in rs]
    c = [r.get("n_calls") or 0 for r in rs]
    print(f"{name:9s} runs={len(rs)} seeds={nseed} dur p50/p90 {pct(d, .5):.0f}/{pct(d, .9):.0f}s "
          f"calls p50/p90 {pct(c, .5)}/{pct(c, .9)} limits {dict(Counter(limit(r['error']) for r in rs))} "
          f"invalid {len(inv)} ({100 * len(inv) / len(rs):.0f}%, resolved {sum(r['resolved'] for r in inv)}) "
          f"gen_tok/run {sum(r.get('completion_tokens', 0) for r in rs) / len(rs):.0f} "
          + " ".join(f"{k}:{v:.1f}/{sum(1 for r in rs if r['repo'].endswith(k)) // nseed}" for k, v in sorted(rep.items())))
