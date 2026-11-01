"""Run a submission (agent) on tasks with the official swegemma Evaluator.

Mirrors the starter notebook: same EvalConfig fields, context caching and
event compaction, LiteLLM routing to an OpenAI-compatible vLLM endpoint.
Differences: Apptainer sandbox (apptainer_sandbox.py) and per-task
time-consistent wheels (task_env.py), as validated in Gate 1.
"""
import argparse, asyncio, csv, json, os, shutil, sys
from pathlib import Path

for k, v in {"LITELLM_LOCAL_MODEL_COST_MAP": "True", "LITELLM_TELEMETRY": "False", "LITELLM_LOG": "WARNING",
             "OPENAI_API_KEY": "EMPTY", "LOCAL_API_KEY": "EMPTY", "OTEL_SDK_DISABLED": "true"}.items():
    os.environ.setdefault(k, v)
sys.path.insert(0, str(Path(__file__).parent))

import litellm, yaml
from adk_submission import VllmConfig, VllmServer
from google.adk.agents.context_cache_config import ContextCacheConfig
from google.adk.apps._configs import EventsCompactionConfig
from swegemma.config import EvalConfig, build_submission_limits
from swegemma.evaluate import Evaluator
import task_env
from apptainer_sandbox import ApptainerManager

litellm.drop_params = True
ap = argparse.ArgumentParser()
ap.add_argument("--submission", type=Path, required=True)
ap.add_argument("--seed", type=int, required=True)
ap.add_argument("--data", type=Path, required=True)
ap.add_argument("--out", type=Path, required=True)
ap.add_argument("--model-path", required=True)
ap.add_argument("--port", type=int, required=True)
ap.add_argument("--concurrency", type=int, default=12)
ap.add_argument("--tasks-csv", type=Path, help="gate1_soundness.csv; only status==sound tasks run")
ap.add_argument("--task-ids", nargs="*")
ap.add_argument("--sif", required=True)
ap.add_argument("--work", type=Path, required=True)
ap.add_argument("--seed-usr-local", type=Path, required=True)
ap.add_argument("--wheels", type=Path, required=True)
a = ap.parse_args()

# Per-seed copy of the submission with the sampling seed set.
a.out.mkdir(parents=True, exist_ok=True)
sub = a.out / "submission"
if sub.exists():
    shutil.rmtree(sub)
shutil.copytree(a.submission, sub)
samp = sub / "configs" / "sampling.yaml"
cfg_s = yaml.safe_load(samp.read_text()) or {}
cfg_s["seed"] = a.seed
samp.write_text(yaml.safe_dump(cfg_s, sort_keys=False))

ev_cfg = (yaml.safe_load((sub / "eval_config.yaml").read_text()) or {}) if (sub / "eval_config.yaml").exists() else {}
ev_cfg = ev_cfg.get("evaluation", ev_cfg)

# Task selection.
ids = a.task_ids
if not ids and a.tasks_csv:
    ids = [r["task_id"] for r in csv.DictReader(open(a.tasks_csv)) if r["status"] == "sound"]
tasks = {json.loads(l)["instance_id"]: json.loads(l) for l in open(a.data / "tasks.jsonl")}
ids = ids or list(tasks)

task_env.install()
dates = json.load(open(a.data / "pypi_release_dates.json"))
wheel_dirs = {tid: task_env.build_task_wheels(tasks[tid], a.wheels, a.data / "snapshots", dates,
                                              a.work / "task_wheels")[0] for tid in ids}

# Model registry pointing at the already-running vLLM server (same aliases as the notebook).
server = VllmServer(VllmConfig(model=a.model_path, port=a.port, host="127.0.0.1"))
models = server.create_model_registry(aliases=["gemma-4-31b-it-qat-w4a16-ct"], model_prefix="openai/",
                                      api_key="EMPTY")


class TaskEnvEvaluator(Evaluator):
    async def evaluate_task(self, task, *args, **kwargs):
        task_env.CURRENT_WHEELS.set(wheel_dirs[task.instance_id])
        return await super().evaluate_task(task, *args, **kwargs)


limits, gen = build_submission_limits()
turns = ev_cfg.get("max_turns", ev_cfg.get("max_llm_calls"))
cfg = EvalConfig(
    tasks_path=a.data / "tasks.jsonl",
    snapshots_dir=a.data / "snapshots",
    results_dir=a.out,
    submission_dir=sub,
    models=models,
    sandbox="docker",
    timeout_seconds=int(ev_cfg.get("timeout_seconds", 300)),
    verification_timeout_seconds=int(ev_cfg.get("verification_timeout_seconds", 300)),
    max_time_minutes=float(ev_cfg.get("max_time_minutes", 60.0)),
    max_tool_calls=int(ev_cfg.get("max_tool_calls", 100)),
    max_turns=int(turns) if turns is not None else None,
    limits=limits,
    generation_constraints=gen,
    context_cache_config=ContextCacheConfig(min_tokens=2048, ttl_seconds=1800, cache_intervals=10),
    events_compaction_config=EventsCompactionConfig(compaction_interval=15, overlap_size=2,
                                                    token_threshold=14336, event_retention_size=5),
    graph_dir=str(a.data / "graphs"),
    embeddings_dir=str(a.data / "embeddings"),
    wheels_dir=a.wheels,
    task_ids=ids,
    concurrency=a.concurrency,
    display_mode="quiet",
)
ev = TaskEnvEvaluator(cfg)
ev.sandbox = ev.docker = ApptainerManager(a.sif, a.work / "sbx", a.seed_usr_local)
res = asyncio.run(ev.run())
rs = res.task_results
print(f"SEED={a.seed} tasks={len(rs)} resolved={sum(r.resolved for r in rs)}")
