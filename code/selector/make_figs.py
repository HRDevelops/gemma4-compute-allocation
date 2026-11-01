"""Figures 1-3 for the paper, computed directly from the per-run feature files.

Usage: python make_figs.py <results_dir> <out_dir>
"""
import itertools, json, statistics as st, sys
from collections import defaultdict
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

R, OUT = Path(sys.argv[1]), Path(sys.argv[2])
OUT.mkdir(parents=True, exist_ok=True)
# Reference categorical palette, fixed slot order; validated (light).
C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10, "axes.edgecolor": INK2, "axes.labelcolor": INK,
    "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF, "lines.linewidth": 2,
})


def load(*paths):
    by = defaultdict(dict)
    for i, p in enumerate(paths):
        for l in open(p):
            r = json.loads(l)
            by[r["task"]][f"{i}:{r['seed']}"] = r
    return by


def valid(r):
    return r["n_src_files"] > 0 and not r["error"]


def r_short(r):
    return (valid(r), r.get("last_run_ok") is True, -r["patch_size"])


def stats(by, k, rule=r_short, max_combos=300):
    import random
    seeds = sorted({s for v in by.values() for s in v})
    combos = list(itertools.combinations(seeds, k))
    if len(combos) > max_combos:
        combos = random.Random(0).sample(combos, max_combos)
    sel, single, orc, mins = [], [], [], []
    for c in combos:
        s = m = o = 0.0
        for t, runs in by.items():
            g = [runs[x] for x in c]
            sc = [rule(r) for r in g]
            b = max(sc)
            w = [r for r, v in zip(g, sc) if v == b]
            s += sum(r["resolved"] for r in w) / len(w)
            m += sum(r["resolved"] for r in g) / k
            o += any(r["resolved"] for r in g)
        sel.append(s); single.append(m); orc.append(o)
        mins.append(sum(by[t][x]["duration"] for t in by for x in c) / 60 / len(by))
    return dict(sel=st.mean(sel), sel_sd=st.pstdev(sel), single=st.mean(single), oracle=st.mean(orc),
                minutes=st.mean(mins))


f = lambda name: R / name / "features.jsonl"
REG = {
    "short (8 min)": load(f("baseline"), f("baseline_holdout")),
    "medium (16 min)": load(f("baseline_medium")),
    "long (40 min)": load(f("baseline_long")),
    "long + budget-aware prompt": load(f("long_thorough")),
}
MARK = ["o", "s", "^", "D"]

# ---- Figure 1: per-task resolve rate, short (10 seeds) vs long (5 seeds)
fig, ax = plt.subplots(figsize=(6.4, 3.4))
cats = ["never", "1–49% of seeds", "50–99% of seeds", "always"]
for i, (name, by) in enumerate([("short, 10 seeds", REG["short (8 min)"]), ("long, 5 seeds", REG["long (40 min)"])]):
    p = [sum(r["resolved"] for r in v.values()) / len(v) for v in by.values()]
    cnt = [sum(q == 0 for q in p), sum(0 < q < .5 for q in p), sum(.5 <= q < 1 for q in p), sum(q == 1 for q in p)]
    xs = [j + (i - .5) * 0.38 for j in range(4)]
    bars = ax.bar(xs, cnt, width=0.36, color=C[i], label=name, edgecolor=SURF, linewidth=2)
    for x, v in zip(xs, cnt):
        ax.text(x, v + 1, str(v), ha="center", va="bottom", fontsize=9, color=INK)
ax.set_xticks(range(4), cats)
ax.set_ylabel("tasks (of 118)")
ax.set_title("Fig. 1  How often each task is solved across seeds", loc="left", fontsize=11, color=INK)
ax.legend(frameon=False, loc="upper center")
ax.grid(axis="x", visible=False)
fig.tight_layout(); fig.savefig(OUT / "fig1_flip.png", dpi=200); plt.close(fig)

# ---- Figure 2: cost vs solved frontier (pre-registered R-short selector, k = 1..5)
fig, ax = plt.subplots(figsize=(6.8, 4.2))
rows = []
for i, (name, by) in enumerate(REG.items()):
    nseed = len({s for v in by.values() for s in v})
    pts = [stats(by, k) for k in range(1, min(5, nseed) + 1)]
    xs, ys = [p["minutes"] for p in pts], [p["sel"] for p in pts]
    sd = [p["sel_sd"] if k > 0 else 0 for k, p in enumerate(pts)]
    ax.errorbar(xs, ys, yerr=sd, color=C[i], marker=MARK[i], markersize=7, capsize=0, elinewidth=1, alpha=0.95,
                markeredgecolor=SURF, markeredgewidth=1.5, label=name)
    if i == 0:  # k labels on one line only, as a guide
        for k, (x, y) in enumerate(zip(xs, ys), 1):
            ax.annotate(f"k={k}", (x, y), textcoords="offset points", xytext=(6, -12), fontsize=7.5, color=INK2)
    off = {0: (8, -4), 1: (8, -14), 2: (-4, -18), 3: (-30, 9)}[i]
    ax.annotate(name, (xs[-1], ys[-1]), textcoords="offset points", xytext=off, fontsize=8.5, color=INK,
                ha="center" if i == 2 else "left")
    for k, p in enumerate(pts, 1):
        rows.append((name, k, round(p["minutes"], 2), round(p["sel"], 1), round(p["sel_sd"], 2), round(p["oracle"], 1)))
ax.set_xlabel("measured agent wall-time per task, summed over k runs (minutes, 2×H100)")
ax.set_ylabel("tasks solved (of 118)")
ax.set_title("Fig. 2  Budget per run vs. number of runs, at measured cost", loc="left", fontsize=11, color=INK)
ax.set_xlim(0, 46)
ax.legend(frameon=False, loc="lower right", fontsize=8.5)
fig.tight_layout(); fig.savefig(OUT / "fig2_frontier.png", dpi=200); plt.close(fig)
with open(OUT / "fig2_data.csv", "w", encoding="utf-8") as fh:
    fh.write("regime,k,minutes,solved_selected,sd,oracle\n")
    for r in rows:
        fh.write(",".join(map(str, r)) + "\n")

# ---- Figure 3: share of oracle headroom recovered, in-sample vs holdout seeds (short regime)
ins, hold = load(f("baseline")), load(f("baseline_holdout"))
fig, ax = plt.subplots(figsize=(6.4, 3.6))
data = []
for i, (name, by, rule) in enumerate([
        ("R-short, seeds 0–4 (rule chosen here)", ins, r_short),
        ("R-short, fresh seeds 5–9 (holdout)", hold, r_short),
        ("validity filter only, seeds 5–9", hold, valid)]):
    ks, sh = [], []
    for k in range(2, 6):
        p = stats(by, k, rule)
        ks.append(k); sh.append(100 * (p["sel"] - p["single"]) / (p["oracle"] - p["single"]))
        data.append((name, k, round(p["single"], 1), round(p["sel"], 1), round(p["oracle"], 1), round(sh[-1], 1)))
    ax.plot(ks, sh, color=C[i], marker=MARK[i], markersize=7, markeredgecolor=SURF, markeredgewidth=1.5)
    ax.annotate(name, (ks[-1], sh[-1]), textcoords="offset points", xytext=(8, [-3, 6, -14][i]), fontsize=8.5, color=INK)
ax.set_xticks([2, 3, 4, 5])
ax.set_xlim(1.8, 8.4)
ax.set_ylim(0, 100)
ax.set_xlabel("k (short runs per task)")
ax.set_ylabel("% of oracle headroom recovered")
ax.set_title("Fig. 3  The test-free selector's gain halves on fresh seeds", loc="left", fontsize=11, color=INK)
fig.tight_layout(); fig.savefig(OUT / "fig3_holdout.png", dpi=200); plt.close(fig)
with open(OUT / "fig3_data.csv", "w", encoding="utf-8") as fh:
    fh.write("selector,k,single,selected,oracle,share_pct\n")
    for r in data:
        fh.write(",".join(map(str, r)) + "\n")
print("figures written to", OUT)
