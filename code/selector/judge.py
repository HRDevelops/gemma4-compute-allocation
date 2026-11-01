"""Gemma-as-judge scores for candidate patches (no tests run, no hidden-test info).

Two judges, both reading only the issue text and the candidate patch(es):
  judge_point: each valid patch alone -> SCORE 0-10, mean of N samples, scaled to [0,1]
  judge_list:  all valid patches of a task together (shuffled) -> pick the best;
               score = fraction of N samples that picked this patch

Usage: python judge.py --features f.jsonl --tasks tasks.jsonl --raw raw_dir --port P --out scores.jsonl
"""
import argparse, asyncio, json, random, re
from collections import defaultdict
from pathlib import Path
from openai import AsyncOpenAI

ap = argparse.ArgumentParser()
ap.add_argument("--features", required=True); ap.add_argument("--tasks", required=True)
ap.add_argument("--raw", required=True); ap.add_argument("--port", type=int, required=True)
ap.add_argument("--out", required=True); ap.add_argument("--n", type=int, default=3)
ap.add_argument("--conc", type=int, default=48)
ap.add_argument("--model", default="gemma")
a = ap.parse_args()

TEST_RE = re.compile(r"(^|/)(tests?/|test_[^/]*\.py$|[^/]*_test\.py$|conftest\.py$)")
cli = AsyncOpenAI(base_url=f"http://127.0.0.1:{a.port}/v1", api_key="EMPTY", timeout=900)
sem = asyncio.Semaphore(a.conc)
tasks = {json.loads(l)["instance_id"]: json.loads(l) for l in open(a.tasks)}
runs = [json.loads(l) for l in open(a.features)]
valid = [r for r in runs if r["n_src_files"] > 0 and not r["error"]]


def src_patch(r, limit):
    """The run's patch restricted to source files (drops repro scripts etc.)."""
    text = (Path(a.raw) / r["seed"] / "patches" / f"{r['task']}.patch").read_text(errors="replace")
    parts = re.split(r"(?=^diff --git )", text, flags=re.M)
    keep = [p for p in parts if p.startswith("diff --git") and p.split("\n")[0].split(" b/")[-1] in r["src_files"]]
    s = "".join(keep)
    return s if len(s) <= limit else s[:limit] + "\n... [truncated]\n"


def issue(tid):
    t = tasks[tid]
    ps = t["problem_statement"]
    return f"Repository: {t['repo']}\n\nIssue:\n{ps[:8000]}"


SYS = ("You are a meticulous senior reviewer of Python pull requests. You judge whether a code change "
       "fully and correctly resolves an issue, as the project's maintainers' hidden tests would check it. "
       "Consider correctness, edge cases named in the issue, and whether the change touches the right place.")


async def ask(prompt, max_tokens=4096, temp=0.7):
    async with sem:
        for attempt in range(3):
            try:
                r = await cli.chat.completions.create(
                    model=a.model, messages=[{"role": "system", "content": SYS}, {"role": "user", "content": prompt}],
                    temperature=temp, top_p=0.95, max_tokens=max_tokens)
                return r.choices[0].message.content or ""
            except Exception as e:
                err = e
                await asyncio.sleep(2 + attempt * 5)
        return f"ERROR {err}"


async def point(r):
    p = (f"{issue(r['task'])}\n\nCandidate patch:\n```diff\n{src_patch(r, 14000)}```\n\n"
         "Does this patch fully and correctly resolve the issue? Reason briefly, then finish with exactly one "
         "line `SCORE: <integer 0-10>` where 10 means certainly correct.")
    outs = await asyncio.gather(*[ask(p) for _ in range(a.n)])
    vals = [int(m.group(1)) for o in outs if (m := re.search(r"SCORE:\s*\**\s*(\d+)", o))]
    return [min(v, 10) / 10 for v in vals]


async def listwise(tid, cands):
    picks = defaultdict(float)
    if len(cands) == 1:
        return {cands[0]["seed"]: 1.0}

    async def one(i):
        order = cands[:]
        random.Random(hash((tid, i))).shuffle(order)
        body = "\n\n".join(f"### Candidate {j + 1}\n```diff\n{src_patch(c, 5000)}```" for j, c in enumerate(order))
        p = (f"{issue(tid)}\n\n{len(order)} candidate patches were written independently for this issue.\n\n{body}\n\n"
             "Which candidate most likely resolves the issue correctly? Compare them, then finish with exactly one "
             "line `BEST: <candidate number>`.")
        o = await ask(p, max_tokens=6144)
        m = re.search(r"BEST:\s*\**\s*(?:Candidate\s*)?(\d+)", o)
        return order[int(m.group(1)) - 1]["seed"] if m and 1 <= int(m.group(1)) <= len(order) else None

    got = await asyncio.gather(*[one(i) for i in range(a.n)])
    for s in got:
        if s:
            picks[s] += 1 / a.n
    return dict(picks)


async def main():
    by_task = defaultdict(list)
    for r in valid:
        by_task[r["task"]].append(r)
    pts = await asyncio.gather(*[point(r) for r in valid])
    lists = dict(zip(by_task, await asyncio.gather(*[listwise(t, c) for t, c in by_task.items()])))
    n_parse = sum(len(v) for v in pts)
    with open(a.out, "w") as f:
        for r, v in zip(valid, pts):
            f.write(json.dumps({"task": r["task"], "seed": r["seed"], "scores": {
                "judge_point": sum(v) / len(v) if v else 0.5,
                "judge_list": lists[r["task"]].get(r["seed"], 0.0)}}) + "\n")
    print(f"judged {len(valid)} patches; parsed {n_parse}/{len(valid) * a.n} point scores; {len(by_task)} tasks listwise")


asyncio.run(main())
