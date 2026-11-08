"""Small shared helpers: logging, JSON IO, ffmpeg discovery."""
from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Iterable

log = logging.getLogger("gsu")


def setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname).1s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stderr,
        force=True,
    )


def read_json(path: os.PathLike | str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: os.PathLike | str, data: Any, indent: int | None = 2) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent, default=_json_default)
    os.replace(tmp, path)


def _json_default(o: Any) -> Any:
    import numpy as np

    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, Path):
        return str(o)
    raise TypeError(f"not JSON serialisable: {type(o)}")


def find_ffmpeg() -> str:
    """Return a path to an ffmpeg binary (PATH first, then imageio-ffmpeg)."""
    exe = os.environ.get("GSU_FFMPEG") or shutil.which("ffmpeg")
    if exe:
        return exe
    try:
        import imageio_ffmpeg  # type: ignore

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception as exc:  # pragma: no cover - depends on environment
        raise RuntimeError(
            "ffmpeg not found. Install ffmpeg or `pip install imageio-ffmpeg`."
        ) from exc


def find_ffprobe() -> str | None:
    exe = os.environ.get("GSU_FFPROBE") or shutil.which("ffprobe")
    if exe:
        return exe
    ff = find_ffmpeg()
    cand = Path(ff).with_name("ffprobe" + Path(ff).suffix)
    return str(cand) if cand.exists() else None


def run(cmd: Iterable[str], **kw: Any) -> subprocess.CompletedProcess:
    cmd = [str(c) for c in cmd]
    log.debug("$ %s", " ".join(cmd))
    return subprocess.run(cmd, check=True, **kw)


class Timer:
    def __init__(self, label: str):
        self.label = label

    def __enter__(self):
        self.t0 = time.time()
        log.info("%s ...", self.label)
        return self

    def __exit__(self, *exc):
        log.info("%s done in %.1fs", self.label, time.time() - self.t0)
