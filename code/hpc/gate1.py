"""Gate 1: check the official verifier on our HPC.

mode=gold  -> agent step replaced by the task's reference patch; expect resolved.
mode=empty -> no patch (skip_agent_patch); expect NOT resolved.

Runs swegemma's own Evaluator (Docker code path) on an Apptainer-backed sandbox
(apptainer_sandbox.py) with a per-task, time-consistent wheel set (task_env.py).
"""
import argparse, asyncio, json, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from adk_submission import ModelRegistry
from adk_eval_core.tracing import SessionTrace
from swegemma.config import EvalConfig, build_submission_limits
from swegemma.evaluate import Evaluator
import task_env
from apptainer_sandbox import ApptainerManager

ap = argparse.ArgumentParser()
ap.add_argument("--mode", choices=["gold", "empty"], required=True)
ap.add_argument("--data", type=Path, required=True)
ap.add_argument("--out", type=Path, required=True)
ap.add_argument("--concurrency", type=int, default=8)
ap.add_argument("--task-ids", nargs="*")
ap.add_argument("--sif", required=True)
ap.add_argument("--work", type=Path, required=True)
ap.add_argument("--seed", type=Path, required=True)
ap.add_argument("--wheels", type=Path, required=True)
a = ap.parse_args()

task_env.install()
tasks = {json.loads(l)["instance_id"]: json.loads(l) for l in open(a.data / "tasks.jsonl")}
dates = json.load(open(a.data / "pypi_release_dates.json"))
wheel_dirs, manifest = {}, {}
for tid, t in tasks.items():
    if a.task_ids and tid not in a.task_ids:
        continue
    wheel_dirs[tid], manifest[tid] = task_env.build_task_wheels(
        t, a.wheels, a.data / "snapshots", dates, a.work / "task_wheels")
a.out.mkdir(parents=True, exist_ok=True)
json.dump(manifest, open(a.out / "task_env_manifest.json", "w"), indent=1)


class TaskEnvEvaluator(Evaluator):
    async def evaluate_task(self, task, *args, **kwargs):
        task_env.CURRENT_WHEELS.set(wheel_dirs[task.instance_id])
        return await super().evaluate_task(task, *args, **kwargs)


class GoldEvaluator(TaskEnvEvaluator):
    async def _run_agent_sandbox(self, task, *args, **kwargs):
        return task.patch, None, SessionTrace()


limits, gen = build_submission_limits()
cfg = EvalConfig(
    tasks_path=a.data / "tasks.jsonl",
    snapshots_dir=a.data / "snapshots",
    results_dir=a.out,
    submission_dir=a.data / "sample_submission",
    models=ModelRegistry(),
    sandbox="docker",
    wheels_dir=a.wheels,
    graph_dir=str(a.data / "graphs"),
    embeddings_dir=str(a.data / "embeddings"),
    verification_timeout_seconds=300,
    skip_agent_patch=(a.mode == "empty"),
    concurrency=a.concurrency,
    task_ids=a.task_ids or None,
    display_mode="quiet",
    limits=limits,
    generation_constraints=gen,
)
ev = (GoldEvaluator if a.mode == "gold" else TaskEnvEvaluator)(cfg)
ev.sandbox = ev.docker = ApptainerManager(a.sif, a.work / "sbx", a.seed)
res = asyncio.run(ev.run())
rs = res.task_results
print(f"MODE={a.mode} tasks={len(rs)} resolved={sum(r.resolved for r in rs)}")
for r in rs:
    if (a.mode == "gold") != bool(r.resolved):
        print("UNEXPECTED", r.instance_id, "resolved=", r.resolved, "exit=", r.test_exit_code, "err=", (r.error or "")[:200])
