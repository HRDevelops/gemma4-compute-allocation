"""Per-run features for test-free patch selection (no hidden-test information used).

Usage: python features.py <baseline_raw_dir> <results_dir_with_seedN_task_results.jsonl> <out.jsonl>
"""
import json, re, sys
from pathlib import Path

raw, resd, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
TEST_RE = re.compile(r"(^|/)(tests?/|test_[^/]*\.py$|[^/]*_test\.py$|conftest\.py$)")
PYTEST_RE = re.compile(r"\bpytest\b|-m\s+unittest")


def parse_patch(text):
    """Split a unified diff into files and normalized changed lines (src vs test vs junk)."""
    files, cur = {}, None
    for line in text.splitlines():
        if line.startswith("diff --git"):
            cur = line.split(" b/")[-1]
            files[cur] = []
        elif cur and (line.startswith("+") or line.startswith("-")) and not line.startswith(("+++", "---")):
            s = line[0] + " ".join(line[1:].split())
            if len(s) > 1:
                files[cur].append(s)
    src = {f: l for f, l in files.items() if not TEST_RE.search(f) and f.endswith(".py") and "/" in f}
    junk = [f for f in files if f not in src and not TEST_RE.search(f)]
    tests = [f for f in files if TEST_RE.search(f)]
    return files, src, junk, tests


def trace_features(path):
    t = json.load(open(path))
    calls = []  # (name, args, obs dict)
    for s in t["steps"]:
        for tc in s.get("tool_calls") or []:
            obs = s.get("observation") or {}
            try:
                o = json.loads(obs.get("content") or "{}")
            except Exception:
                o = {}
            calls.append((tc["function_name"], tc.get("arguments") or {}, o if isinstance(o, dict) else {}))
    edit_idx = [i for i, c in enumerate(calls) if c[0] in ("edit_file", "write_file")]
    last_edit = edit_idx[-1] if edit_idx else -1
    tests = [(i, c) for i, c in enumerate(calls) if c[0] == "run_command" and PYTEST_RE.search(str(c[1].get("command", "")))]
    runs = [(i, c) for i, c in enumerate(calls) if c[0] == "run_command"]
    after = [c for i, c in tests if i > last_edit]
    runs_after = [c for i, c in runs if i > last_edit]
    edit_fail = sum(1 for c in calls if c[0] == "edit_file" and c[2].get("status") not in ("ok", None))
    final = [s for s in t["steps"] if s["extra"].get("event_type") == "final"]
    return dict(
        n_calls=len(calls),
        n_edits=len(edit_idx),
        n_edit_fail=edit_fail,
        n_tests=len(tests),
        test_after_edit=bool(after),
        last_test_ok=(after[-1][2].get("exit_code") == 0) if after else None,
        any_test_ok_after=any(c[2].get("exit_code") == 0 for c in after) if after else None,
        last_run_ok=(runs_after[-1][2].get("exit_code") == 0) if runs_after else None,
        submitted=any(c[0] == "submit_patch" for c in calls),
        final_len=len(final[0]["message"]) if final else 0,
        tokens=t.get("final_metrics", {}).get("total_tokens", 0),
        prompt_tokens=t.get("final_metrics", {}).get("total_prompt_tokens", 0),
        completion_tokens=t.get("final_metrics", {}).get("total_completion_tokens", 0),
    )


rows = []
for rf in sorted(resd.glob("seed*_task_results.jsonl")):
    seed = rf.name.split("_")[0]
    for l in open(rf):
        r = json.loads(l)
        tid = r["instance_id"]
        pp = raw / seed / "patches" / f"{tid}.patch"
        text = pp.read_text(errors="replace") if pp.exists() else ""
        files, src, junk, tests = parse_patch(text)
        tp = raw / seed / "traces" / f"trace_{tid}.json"
        tf = trace_features(tp) if tp.exists() else {}
        rows.append(dict(
            task=tid, seed=seed, repo=r["repo"], resolved=bool(r["resolved"]),  # label: evaluation only
            error=r.get("error") or "", duration=r["duration_seconds"],
            patch_size=len(text), n_files=len(files), n_src_files=len(src), junk_files=junk, test_files=tests,
            src_lines=sorted({x for v in src.values() for x in v}),
            src_files=sorted(src), **tf))
with open(out, "w") as f:
    for r in rows:
        f.write(json.dumps(r) + "\n")
print(f"wrote {len(rows)} runs to {out}")
