"""Build the static WebGPU site: template + model file + viewer-config.json.

Everything in viewer-config.json is expressed in the *model file's* frame, so the
JavaScript never needs to know about COLMAP vs aligned coordinates:
  * `.splat` files carry a sidecar `<file>.json` {"frame": "aligned"|"colmap", "up": [...]}
    written by `gsu export`; aligned models get poses/route converted with scene.json.
  * `.ply` fallbacks (trained splat or SfM points) are in the COLMAP frame.
"""
from __future__ import annotations

import http.server
import os
import shutil
import socketserver
import subprocess
import sys
from functools import partial
from pathlib import Path

import numpy as np

from ..common import log, read_json, write_json
from ..paths import Workspace
from .poses import DEFAULT_TRANSITION, load_poses, matrix_from_quat_xyzw, pose_from_camera, quat_xyzw_from_matrix

TEMPLATE_DIR = Path(__file__).parent / "template"
N_AUTO_POSES = 5


def _pick_model(ws: Workspace, splat: Path | None) -> Path:
    cands = [splat] if splat else [ws.splat_web, ws.splat_ply, ws.points_ply]
    for c in cands:
        if c and Path(c).exists():
            return Path(c)
    raise SystemExit(
        f"no model to ship: looked for {', '.join(str(c) for c in cands if c)}. "
        "Run `gsu export` (after training) or at least `gsu sfm`."
    )


def _sidecar(model: Path) -> dict:
    sc = model.with_name(model.name + ".json")
    if sc.exists():
        try:
            return read_json(sc)
        except Exception as exc:  # pragma: no cover
            log.warning("ignoring unreadable sidecar %s: %s", sc, exc)
    return {}


class Similarity:
    """x' = s R x + t, and its inverse."""

    def __init__(self, R: np.ndarray, t: np.ndarray, s: float):
        self.R, self.t, self.s = np.asarray(R, float), np.asarray(t, float), float(s)

    @classmethod
    def identity(cls) -> "Similarity":
        return cls(np.eye(3), np.zeros(3), 1.0)

    def point(self, x) -> np.ndarray:
        return (self.R @ np.asarray(x, float)) * self.s + self.t

    def points(self, xs) -> np.ndarray:
        xs = np.asarray(xs, float).reshape(-1, 3)
        return (xs @ self.R.T) * self.s + self.t

    def quat(self, q_xyzw) -> list[float]:
        return quat_xyzw_from_matrix(self.R @ matrix_from_quat_xyzw(q_xyzw))

    def inverse(self) -> "Similarity":
        Rt = self.R.T
        return Similarity(Rt, -(Rt @ self.t) / self.s, 1.0 / self.s)

    def to_json(self) -> dict:
        return {"rotation": self.R.tolist(), "translation": self.t.tolist(), "scale": self.s}


def _alignment(ws: Workspace) -> Similarity | None:
    if not ws.scene_json.exists():
        return None
    d = read_json(ws.scene_json)
    return Similarity(np.array(d["rotation"]), np.array(d["translation"]), float(d["scale"]))


def _auto_poses(ws: Workspace) -> list[dict]:
    """A handful of evenly spaced viewpoints along the walk, so the site works before
    the user names their own poses with `gsu poses`."""
    if not ws.cameras_json.exists():
        return []
    cams = read_json(ws.cameras_json)["images"]
    if not cams:
        return []
    k = min(N_AUTO_POSES, len(cams))
    idx = np.unique(np.round(np.linspace(0, len(cams) - 1, k)).astype(int))
    poses = []
    for j, i in enumerate(idx):
        p = pose_from_camera(cams[i])
        pid = "start" if j == 0 else ("end" if j == len(idx) - 1 else f"point-{j + 1}")
        p.update({"id": pid, "label": pid.replace("-", " ").title(), "auto": True})
        poses.append(p)
    return poses


def _camera_up(ws: Workspace) -> list[float]:
    if ws.scene_json.exists():
        d = read_json(ws.scene_json)
        if "up_model" in d:
            return [float(v) for v in d["up_model"]]
    if ws.cameras_json.exists():
        cams = read_json(ws.cameras_json)["images"]
        if cams:
            down = np.array([np.asarray(c["camera_to_world"])[:3, 1] for c in cams])
            m = -down.mean(axis=0)
            n = np.linalg.norm(m)
            if n > 1e-9:
                return (m / n).tolist()
    return [0.0, 0.0, 1.0]


def build_config(ws: Workspace, model_file: Path, model_name: str, title: str) -> dict:
    side = _sidecar(model_file)
    model_frame = side.get("frame") or "colmap"
    align = _alignment(ws)
    if model_frame == "aligned" and align is None:
        log.warning("model is in the aligned frame but %s is missing; poses may be misplaced", ws.scene_json)
    colmap_to_model = align if (model_frame == "aligned" and align is not None) else Similarity.identity()

    poses_doc = load_poses(ws.poses_json)
    poses = poses_doc["poses"]
    auto = False
    if not poses:
        poses = _auto_poses(ws)
        auto = bool(poses)
        if auto:
            log.info("no poses.json: using %d automatic viewpoints along the walk (name your own with `gsu poses`)", len(poses))
    poses_frame = poses_doc.get("frame", "colmap")
    out_poses = []
    for p in poses:
        q = dict(p)
        if poses_frame == "colmap" and model_frame == "aligned":
            q["position"] = colmap_to_model.point(p["position"]).tolist()
            q["quaternion"] = colmap_to_model.quat(p["quaternion"])
            if "via" in p and p["via"]:
                q["via"] = [
                    ({"position": colmap_to_model.point(v["position"]).tolist(),
                      **({"quaternion": colmap_to_model.quat(v["quaternion"])} if "quaternion" in v else {})}
                     if isinstance(v, dict) else colmap_to_model.point(v).tolist())
                    for v in p["via"]
                ]
        elif poses_frame != model_frame and not (poses_frame == "aligned" and model_frame == "aligned"):
            log.warning("poses are in frame %r but the model is %r; shipping them unchanged", poses_frame, model_frame)
        out_poses.append(q)

    # walk path for the `route` transition mode (geometry stage writes it in the aligned frame)
    route = None
    path_json = ws.geometry / "path.json"
    if path_json.exists():
        try:
            pd = read_json(path_json)
            pts = np.asarray(pd.get("points", []), float).reshape(-1, 3)
            if len(pts) >= 2:
                if pd.get("frame", "aligned") == "aligned" and model_frame == "colmap":
                    if align is None:
                        log.warning("path.json is aligned but scene.json is missing; route disabled")
                        pts = None
                    else:
                        pts = align.inverse().points(pts)
                if pts is not None:
                    route = np.round(pts, 5).tolist()
        except Exception as exc:  # pragma: no cover
            log.warning("could not read %s: %s", path_json, exc)

    if model_frame == "aligned":
        up = [0.0, 0.0, 1.0]
    else:
        up = side.get("up") or _camera_up(ws)

    transitions = dict(poses_doc.get("transitions") or {})
    transitions.setdefault("default", dict(DEFAULT_TRANSITION))
    return {
        "title": title,
        "model": model_name,
        "model_kind": side.get("kind") or ("splat" if model_name.endswith(".splat") else "ply"),
        "frame": model_frame,
        "up": [float(v) for v in up],
        "poses": out_poses,
        "poses_auto": auto,
        "transitions": transitions,
        "route": route,
        "fallback_point_size": 0.02,
        # lets the in-browser pose author (press P) emit poses.json-compatible COLMAP-frame poses
        "colmap_to_model": colmap_to_model.to_json(),
        "n_splats": side.get("n"),
    }


README = """gsplat-utils web viewer
=======================

This folder is a complete static website. Upload it to any static host
(GitHub Pages, Netlify, Vercel, S3, nginx, ...) or run locally:

    python -m http.server 8000      # then open http://localhost:8000/

Requirements for visitors: a WebGPU browser (Chrome/Edge 113+, Safari 18+,
recent Firefox). No special HTTP headers are needed (no SharedArrayBuffer).
Serve the model file with gzip/brotli if your host supports it.

Files
  index.html, style.css, *.js   the viewer (no build step, no external scripts)
  model.splat | model.ply       the Gaussian splat (or SfM points as a preview)
  viewer-config.json            title, named poses, transition defaults, walk route

Editing poses / transitions
  gsu poses --work <work> --set studio 42.5 --set shop frame_000400.jpg
  gsu web   --work <work>
  ...or press P in the viewer to copy the current viewpoint as JSON.
  Transition defaults live under "transitions.default" in viewer-config.json
  (mode, easing, duration_ms, effect, effect_strength, up_lock, ...); any pose
  can override them with a "transition": {...} object.
"""


def run_web(ws: Workspace, out: Path | None = None, splat: Path | None = None, title: str = "gsplat",
            serve: bool = False, port: int = 8000) -> Path:
    out = Path(out) if out else ws.web
    out.mkdir(parents=True, exist_ok=True)
    model_src = _pick_model(ws, splat)
    for f in sorted(TEMPLATE_DIR.iterdir()):
        if f.is_file():
            shutil.copy2(f, out / f.name)
    model_name = "model.splat" if model_src.suffix.lower() == ".splat" else "model.ply"
    dst = out / model_name
    # remove the other variant so the site has exactly one model file
    for other in ("model.splat", "model.ply"):
        if other != model_name and (out / other).exists():
            (out / other).unlink()
    if not (dst.exists() and os.path.samefile(dst, model_src)):
        shutil.copy2(model_src, dst)
    config = build_config(ws, model_src, model_name, title)
    write_json(out / "viewer-config.json", config)
    (out / "README.txt").write_text(README)
    log.info("site written to %s (model %s, %d poses%s, frame=%s, route=%s)", out, model_src.name,
             len(config["poses"]), " auto" if config["poses_auto"] else "", config["frame"],
             "yes" if config["route"] else "no")
    if serve:
        serve_dir(out, port)
    return out


def serve_dir(folder: Path, port: int) -> None:
    handler = partial(http.server.SimpleHTTPRequestHandler, directory=str(folder))
    socketserver.TCPServer.allow_reuse_address = True
    with socketserver.TCPServer(("", port), handler) as httpd:
        print(f"serving {folder} at http://localhost:{port}/  (Ctrl-C to stop)", flush=True)
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            pass
