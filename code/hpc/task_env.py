"""Per-task, time-consistent test environment for the swegemma Docker code path.

The public wheel folder holds several versions of some packages (56 of starlette,
7 of pygments, ...), but swegemma injects only the highest version of each, and
it also injects released wheels of the repositories under test, which then
shadow /workspace. For each task we therefore build a private wheel folder:

  * the task's own distribution (e.g. `requests` for psf/requests) is left out;
  * for every package with several versions we keep exactly one: the highest
    version that satisfies the repository's own declared constraint and was
    released on PyPI before the task's creation date (if none predates it, the
    lowest satisfying version);
  * single-version packages are kept unchanged.

`install()` patches swegemma so each running task sees its own folder (through
a ContextVar, safe under asyncio concurrency) and so the unpacked-wheel tar
cache is keyed by the selected wheel set instead of a fixed name.
"""
from __future__ import annotations

import contextvars
import hashlib
import io
import json
import os
import re
import tarfile
import tomllib
from datetime import datetime
from pathlib import Path

from packaging.specifiers import InvalidSpecifier, SpecifierSet
from packaging.version import InvalidVersion, Version

CURRENT_WHEELS: contextvars.ContextVar[Path | None] = contextvars.ContextVar("wheels", default=None)
norm = lambda n: re.sub(r"[-_.]+", "-", n).lower()


def _poetry_spec(s: str) -> str:
    s = s.strip()
    m = re.fullmatch(r"\^(\d+)(?:\.(\d+))?(?:\.(\d+))?", s)
    if m:
        a, b, c = (int(x) if x else 0 for x in m.groups())
        upper = f"{a + 1}.0.0" if a > 0 else (f"0.{b + 1}.0" if b > 0 else f"0.0.{c + 1}")
        return f">={a}.{b}.{c},<{upper}"
    m = re.fullmatch(r"~(\d+)(?:\.(\d+))?(?:\.(\d+))?", s)
    if m:
        a, b, c = (int(x) if x else 0 for x in m.groups())
        return f">={a}.{b}.{c},<{a}.{b + 1}.0" if m.group(2) else f">={a}.0.0,<{a + 1}.0.0"
    return "" if s in ("*", "") else s


def _req_lines_from_snapshot(snapshot: Path) -> list[str]:
    """Requirement strings from pyproject.toml and requirements*.txt inside a snapshot .tgz."""
    reqs: list[str] = []
    with tarfile.open(snapshot, "r:gz") as tf:
        for m in tf.getmembers():
            name = m.name.lstrip("./")
            if not m.isfile() or name.count("/") > 1:
                continue
            base = os.path.basename(name)
            if base == "pyproject.toml" and "/" not in name:
                data = tomllib.load(io.BytesIO(tf.extractfile(m).read()))
                proj = data.get("project", {})
                reqs += proj.get("dependencies", [])
                for v in proj.get("optional-dependencies", {}).values():
                    reqs += v
                for v in data.get("dependency-groups", {}).values():
                    reqs += [x for x in v if isinstance(x, str)]
                poetry = data.get("tool", {}).get("poetry", {})
                groups = [poetry.get("dependencies", {}), poetry.get("dev-dependencies", {})]
                groups += [g.get("dependencies", {}) for g in poetry.get("group", {}).values()]
                for g in groups:
                    for k, v in g.items():
                        spec = v if isinstance(v, str) else (v.get("version", "") if isinstance(v, dict) else "")
                        reqs.append(f"{k}{_poetry_spec(spec)}")
            elif re.fullmatch(r"(test-)?requirements.*\.txt", base):
                reqs += tf.extractfile(m).read().decode("utf-8", "replace").splitlines()
    return reqs


def _constraints(reqs: list[str]) -> dict[str, SpecifierSet]:
    out: dict[str, SpecifierSet] = {}
    for line in reqs:
        line = line.split("#", 1)[0].split(";", 1)[0].strip()
        m = re.match(r"^([A-Za-z0-9_.\-]+)\s*(?:\[[^\]]*\])?\s*(.*)$", line)
        if not m or line.startswith("-"):
            continue
        name, spec = norm(m.group(1)), m.group(2).strip().replace(" ", "")
        try:
            ss = SpecifierSet(spec)
        except InvalidSpecifier:
            continue
        out[name] = out[name] & ss if name in out else ss
    return out


def _v(s: str) -> Version:
    try:
        return Version(s)
    except InvalidVersion:
        return Version("0")


def build_task_wheels(task: dict, wheels_dir: Path, snapshots_dir: Path, dates: dict,
                      out_root: Path) -> tuple[Path, dict]:
    """Create <out_root>/<instance_id>/ with symlinks to the selected wheels; return (dir, choices)."""
    own = norm(task["repo"].split("/")[1])
    cutoff = datetime.fromisoformat(task["created_at"].replace("Z", "+00:00"))
    cons = _constraints(_req_lines_from_snapshot(snapshots_dir / f"{task['instance_id']}.tgz"))
    by_dist: dict[str, list[str]] = {}
    for f in sorted(os.listdir(wheels_dir)):
        if f.endswith(".whl"):
            by_dist.setdefault(norm(f.split("-")[0]), []).append(f)
    keep: list[str] = []
    choices: dict = {}
    for dist, files in by_dist.items():
        if dist == own:
            choices[dist] = "excluded (repository under test)"
            continue
        versions = sorted({f.split("-")[1] for f in files}, key=_v)
        if len(versions) == 1:
            keep += files
            continue
        spec = cons.get(dist)
        sat = [v for v in versions if spec is None or spec.contains(_v(v), prereleases=True)] or versions
        released = {v: dates.get(dist, {}).get(v) for v in sat}
        before = [v for v in sat if released[v] and datetime.fromisoformat(released[v].replace("Z", "+00:00")) <= cutoff]
        pick = max(before, key=_v) if before else min(sat, key=_v)
        keep += [f for f in files if f.split("-")[1] == pick]
        choices[dist] = f"{pick} (spec {spec or 'none'})"
    d = out_root / task["instance_id"]
    d.mkdir(parents=True, exist_ok=True)
    for f in keep:
        link = d / f
        if not link.exists():
            link.symlink_to(wheels_dir / f)
    return d, choices


def install() -> None:
    """Patch swegemma to use CURRENT_WHEELS and a content-keyed wheel-tar cache."""
    import swegemma.harness.container_setup as cs

    orig_resolve = cs.resolve_wheels_dir
    orig_build = cs._build_unpacked_wheels_tar

    def resolve(config=None):
        return CURRENT_WHEELS.get() or orig_resolve(config)

    def build(selected, cache_name, marker_name):
        h = hashlib.sha1("\n".join(sorted(p.name for p in selected)).encode()).hexdigest()[:12]
        return orig_build(selected, f"{cache_name}_{h}", marker_name)

    cs.resolve_wheels_dir = resolve
    cs._build_unpacked_wheels_tar = build
