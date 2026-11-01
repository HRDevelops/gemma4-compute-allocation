"""Set up one sandbox per task with the official setup path, then report where key modules import from."""
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from apptainer_sandbox import ApptainerManager
from adk_submission import ModelRegistry
from swegemma.config import EvalConfig
from swegemma.harness.container_setup import extract_snapshot, setup_container_workspace
sif, work, seed, data, wheels = sys.argv[1:6]; ids = sys.argv[6:]
data = Path(data)
cfg = EvalConfig(tasks_path=data/"tasks.jsonl", snapshots_dir=data/"snapshots", results_dir=Path(work)/"res",
                 submission_dir=data/"sample_submission", models=ModelRegistry(), sandbox="docker", wheels_dir=Path(wheels))
m = ApptainerManager(sif, Path(work)/"sbx", seed)
T = {json.loads(l)["instance_id"]: json.loads(l) for l in open(data/"tasks.jsonl")}
for tid in ids:
    t = T[tid]; cid = m.start()
    extract_snapshot(m, cid, data/"snapshots"/f"{tid}.tgz")
    setup_container_workspace(m, cid, repo=t["repo"], config=cfg)
    mod = t["repo"].split("/")[1]
    r = m.exec(cid, f"cd /workspace && PYTHONSAFEPATH=1 PYTHONNOUSERSITE=1 python3 -s -c 'import {mod},pygments,typing_inspection; print({mod}.__file__, pygments.__version__)'; ls /usr/local/lib/python3.13/site-packages | grep -iE '^({mod}|__editable__|workspace_paths)' ")
    print(tid, "|", r.stdout.strip().replace("\n", " ; "), "|", r.stderr.strip()[-200:])
    m.stop(cid)
