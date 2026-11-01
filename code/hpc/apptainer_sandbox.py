"""Docker-free stand-in for swegemma's ContainerManager, backed by Apptainer.

Each sandbox is a host directory holding the writable parts of a container
(/workspace, /tmp, /var/tmp, /root, /sandbox and a private copy of /usr/local).
Every command runs as `apptainer exec` of the official sandbox image
(Dockerfile.public: python:3.13-slim + git/patch/pigz + pytest + /wheels) with
those directories bound in, so container paths match the Docker scorer exactly.

Because the class is not named SubprocessManager, swegemma takes its Docker
code path: snapshot extraction, wheel injection into site-packages and
/sandbox/setup.py run exactly as in official scoring.

Known differences from the Docker scorer: no network isolation, no 4 GiB /
2 vCPU limits, commands run as the calling user instead of root.
"""
from __future__ import annotations

import io
import logging
import os
import shutil
import stat
import subprocess
import threading
import time
import uuid
from pathlib import Path
from typing import Any

from adk_eval_core.sandbox import ExecutionResult
from swegemma.sandbox.docker import ContainerConfig, ContainerManager

logger = logging.getLogger(__name__)

# Container directories that live on the host, one copy per sandbox.
BOUND = ("/workspace", "/tmp", "/var/tmp", "/root", "/sandbox", "/usr/local")
PATH_ENV = "/usr/local/bin:/usr/local/sbin:/usr/sbin:/usr/bin:/sbin:/bin"


def _rmtree(path: Path) -> None:
    def onerror(func, p, _exc):
        try:
            os.chmod(p, stat.S_IRWXU)
            func(p)
        except Exception:
            pass
    shutil.rmtree(path, onerror=onerror)


class ApptainerManager(ContainerManager):
    def __init__(self, sif: str | Path, root: str | Path, usr_local_seed: str | Path,
                 config: ContainerConfig | None = None, apptainer: str = "apptainer"):
        super().__init__(config or ContainerConfig())
        self.sif = str(sif)
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.usr_local_seed = Path(usr_local_seed)  # extracted /usr/local of the image
        self.apptainer = apptainer
        self._roots: dict[str, Path] = {}
        self._rlock = threading.RLock()

    # ----- lifecycle -------------------------------------------------
    def start(self) -> str:
        cid = uuid.uuid4().hex[:12]
        r = self.root / f"sbx_{cid}"
        for d in ("workspace", "tmp", "var_tmp", "root", "sandbox"):
            (r / d).mkdir(parents=True, exist_ok=True)
        subprocess.run(["cp", "-a", f"{self.usr_local_seed}/.", str(r / "usr_local")], check=True)
        with self._rlock:
            self._roots[cid] = r
        with self._lock:
            self._active_containers.add(cid)
            if self._default_container_id is None:
                self._default_container_id = cid
        return cid

    def stop(self, container_id: str) -> None:
        with self._rlock:
            r = self._roots.pop(container_id, None)
        with self._lock:
            self._active_containers.discard(container_id)
            if self._default_container_id == container_id:
                self._default_container_id = next(iter(self._active_containers), None)
        if r is not None:
            _rmtree(r)

    def cleanup_all(self) -> None:
        for cid in list(self._roots):
            self.stop(cid)

    # ----- path mapping ----------------------------------------------
    def _host_dir(self, cid: str, prefix: str) -> Path:
        return self._roots[cid] / prefix.strip("/").replace("/", "_")

    def _map(self, cid: str, cpath: str) -> Path | None:
        cpath = os.path.normpath(cpath)
        for prefix in sorted(BOUND, key=len, reverse=True):
            if cpath == prefix or cpath.startswith(prefix + "/"):
                rel = cpath[len(prefix):].lstrip("/")
                return self._host_dir(cid, prefix) / rel
        return None

    # ----- exec ------------------------------------------------------
    def exec(self, container_id: str, command: str, *, timeout: int | None = None) -> ExecutionResult:
        timeout = timeout if timeout is not None else self.config.timeout_seconds
        binds = ",".join(f"{self._host_dir(container_id, p)}:{p}" for p in BOUND)
        env = {"TEST_TMPDIR": "/tmp", "HOME": "/root", "PATH": PATH_ENV, "LANG": "C.UTF-8"}
        env.update(self.config.environment or {})
        cmd = [self.apptainer, "-s", "exec", "--containall", "--no-home", "--cleanenv",
               "--bind", binds, "--pwd", self.config.working_dir]
        for k, v in env.items():
            cmd += ["--env", f"{k}={v}"]
        cmd += [self.sif, "timeout", "-k", "5s", str(timeout), "/bin/bash", "-c", command]
        t0 = time.perf_counter()
        try:
            p = subprocess.run(cmd, capture_output=True, timeout=timeout + 60)
            out, err, code = p.stdout, p.stderr, p.returncode
        except subprocess.TimeoutExpired as e:
            out, err, code = e.stdout or b"", e.stderr or b"", 124
        dur = time.perf_counter() - t0
        timed_out = code == 124 and dur >= max(0.0, timeout - 1.0)
        lim = 10 * 1024 * 1024
        so = out.decode("utf-8", "replace")[:lim]
        se = err.decode("utf-8", "replace")[:lim]
        status = "timeout" if timed_out else ("ok" if code == 0 else "error")
        return ExecutionResult(status=status, stdout=so, stderr=se, exit_code=code, duration_seconds=dur)

    # ----- file transfer (host side, no container needed) ------------
    def _copy_to_container(self, container_id: str, src: Path, dst: str) -> None:
        src = Path(src)
        target_dir, arcname, _ = self._resolve_copy_destination(src, dst)
        hdir = self._map(container_id, target_dir)
        if hdir is None:
            raise ValueError(f"copy_to: {target_dir} is outside the writable sandbox dirs")
        hdir.mkdir(parents=True, exist_ok=True)
        if src.is_file():
            shutil.copy2(src, hdir / (arcname or src.name))
        else:
            shutil.copytree(src, hdir, dirs_exist_ok=True, symlinks=True)

    def _copy_from_container(self, container_id: str, src: str, dst: Path, **_: Any) -> None:
        h = self._map(container_id, src)
        if h is None or not h.exists():
            raise FileNotFoundError(f"copy_from: {src} not found in sandbox {container_id}")
        dst = Path(dst)
        dst.parent.mkdir(parents=True, exist_ok=True)
        if h.is_dir():
            shutil.copytree(h, dst, dirs_exist_ok=True, symlinks=True)
        else:
            shutil.copy2(h, dst)

    def _path_exists_in_container(self, container_id: str, path: str) -> bool:
        h = self._map(container_id, path)
        if h is not None:
            return h.exists()
        return self.exec(container_id, f"test -e '{path}'").exit_code == 0

    def _host_tar(self, container_id: str, dst_dir: str, args: list[str], stdin: bytes | None = None) -> bool:
        h = self._map(container_id, dst_dir)
        if h is None:
            return False
        h.mkdir(parents=True, exist_ok=True)
        p = subprocess.run(["tar", "--no-same-owner", *args, "-C", str(h)],
                           input=stdin, capture_output=True)
        if p.returncode != 0:
            logger.warning("host tar into %s failed: %s", dst_dir, p.stderr.decode()[:300])
        return p.returncode == 0

    def extract_archive_to_container(self, container_id: str, archive_path: Path, dst_dir: str = "/workspace") -> bool:
        archive_path = Path(archive_path)
        if not archive_path.exists():
            raise FileNotFoundError(f"Archive path {archive_path} does not exist")
        name = archive_path.name
        args = ["-I", "pigz", "-xf", str(archive_path)] if name.endswith((".tgz", ".tar.gz")) else ["-xf", str(archive_path)]
        return self._host_tar(container_id, dst_dir, args)

    def _stream_tar_via_exec_socket(self, container_id: str, tar_stream: Any, target_dir: str) -> bool:
        tar_stream.seek(0)
        data = tar_stream.read() if isinstance(tar_stream, io.IOBase) or hasattr(tar_stream, "read") else bytes(tar_stream)
        return self._host_tar(container_id, target_dir, ["-xf", "-"], stdin=data)
