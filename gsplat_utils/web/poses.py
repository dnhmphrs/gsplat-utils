"""Author `poses.json`: the named viewpoints ("studio", "about", "shop", ...) the
website transitions between. Each pose is copied from a registered camera of
the reconstruction, chosen by image name, frame index or video time.

poses.json (poses stay in the COLMAP/model frame; `gsu web` converts):
{
  "frame": "colmap",
  "poses": [{"id": "studio", "label": "Studio", "position": [x,y,z],
             "quaternion": [x,y,z,w], "fov_y_deg": 55.0,
             "source_image": "frame_000123.jpg"}],
  "transitions": {"default": {"mode": "spline", "easing": "easeInOutCubic",
                              "duration_ms": 1800, "effect": "none", "up_lock": true}}
}
"""
from __future__ import annotations

import math
import re
from pathlib import Path

import numpy as np

from ..common import log, read_json, write_json
from ..paths import Workspace

DEFAULT_TRANSITION = {
    "mode": "spline",
    "easing": "easeInOutCubic",
    "duration_ms": 1800,
    "effect": "none",
    "effect_strength": 1.0,
    "up_lock": True,
}


def quat_xyzw_from_matrix(R: np.ndarray) -> list[float]:
    """Rotation matrix -> unit quaternion [x, y, z, w] (Shepperd's method)."""
    R = np.asarray(R, dtype=np.float64)
    m00, m01, m02 = R[0]
    m10, m11, m12 = R[1]
    m20, m21, m22 = R[2]
    tr = m00 + m11 + m22
    if tr > 0:
        s = math.sqrt(tr + 1.0) * 2
        w, x, y, z = 0.25 * s, (m21 - m12) / s, (m02 - m20) / s, (m10 - m01) / s
    elif m00 > m11 and m00 > m22:
        s = math.sqrt(1.0 + m00 - m11 - m22) * 2
        w, x, y, z = (m21 - m12) / s, 0.25 * s, (m01 + m10) / s, (m02 + m20) / s
    elif m11 > m22:
        s = math.sqrt(1.0 + m11 - m00 - m22) * 2
        w, x, y, z = (m02 - m20) / s, (m01 + m10) / s, 0.25 * s, (m12 + m21) / s
    else:
        s = math.sqrt(1.0 + m22 - m00 - m11) * 2
        w, x, y, z = (m10 - m01) / s, (m02 + m20) / s, (m12 + m21) / s, 0.25 * s
    q = np.array([x, y, z, w])
    q /= np.linalg.norm(q)
    if q[3] < 0:
        q = -q
    return [float(v) for v in q]


def matrix_from_quat_xyzw(q) -> np.ndarray:
    x, y, z, w = (float(v) for v in q)
    n = math.sqrt(x * x + y * y + z * z + w * w) or 1.0
    x, y, z, w = x / n, y / n, z / n, w / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def pose_from_camera(cam: dict, fov: float | None = None) -> dict:
    c2w = np.asarray(cam["camera_to_world"], dtype=np.float64)
    R = c2w[:3, :3]
    # re-orthonormalise (JSON rounding) so the quaternion is a clean rotation
    u, _, vt = np.linalg.svd(R)
    R = u @ vt
    if np.linalg.det(R) < 0:
        u[:, -1] *= -1
        R = u @ vt
    if fov is None:
        fov = math.degrees(2 * math.atan(0.5 * float(cam["height"]) / float(cam["fy"])))
    return {
        "position": [float(v) for v in c2w[:3, 3]],
        "quaternion": quat_xyzw_from_matrix(R),
        "fov_y_deg": round(float(fov), 3),
        "source_image": cam["name"],
    }


def resolve_camera(cameras: list[dict], value: str) -> dict:
    """Find a camera by image name, frame index, or video time (seconds)."""
    v = str(value).strip()
    by_name = {c["name"]: c for c in cameras}
    if v in by_name:
        return by_name[v]
    stem_hits = [c for c in cameras if Path(c["name"]).stem == Path(v).stem]
    if len(stem_hits) == 1:
        return stem_hits[0]
    if re.fullmatch(r"\d+", v):
        idx = int(v)
        have = [c for c in cameras if c.get("frame_index", -1) >= 0]
        if have:
            return min(have, key=lambda c: abs(int(c["frame_index"]) - idx))
        return cameras[min(idx, len(cameras) - 1)]
    if re.fullmatch(r"\d*\.\d+|\d+\.\d*|\d+(\.\d+)?s", v):
        t = float(v.rstrip("s"))
        have = [c for c in cameras if c.get("time_s") is not None]
        if have:
            return min(have, key=lambda c: abs(float(c["time_s"]) - t))
        raise SystemExit(f"cameras.json has no time_s; give a frame index or image name instead of {v!r}")
    raise SystemExit(f"cannot resolve {v!r}: use an image name (frame_000123.jpg), a frame index (123) or a time (42.5)")


def load_poses(path: Path) -> dict:
    if Path(path).exists():
        d = read_json(path)
        d.setdefault("frame", "colmap")
        d.setdefault("poses", [])
        d.setdefault("transitions", {"default": dict(DEFAULT_TRANSITION)})
        return d
    return {"frame": "colmap", "poses": [], "transitions": {"default": dict(DEFAULT_TRANSITION)}}


def _default_label(pid: str) -> str:
    return re.sub(r"[-_]+", " ", pid).strip().title() or pid


def run_poses(ws: Workspace, sets: list[list[str]], labels: list[list[str]], fov: float | None,
              do_list: bool) -> None:
    data = load_poses(ws.poses_json)
    if sets:
        if not ws.cameras_json.exists():
            raise SystemExit(f"{ws.cameras_json} not found; run `gsu sfm` first")
        cameras = read_json(ws.cameras_json)["images"]
        order = {p["id"]: i for i, p in enumerate(data["poses"])}
        for pid, value in sets:
            cam = resolve_camera(cameras, value)
            pose = pose_from_camera(cam, fov)
            pose["id"] = pid
            if pid in order:
                old = data["poses"][order[pid]]
                pose["label"] = old.get("label", _default_label(pid))
                for k, v in old.items():  # keep user extras (transition overrides, via, ...)
                    pose.setdefault(k, v)
                data["poses"][order[pid]] = pose
            else:
                pose["label"] = _default_label(pid)
                order[pid] = len(data["poses"])
                data["poses"].append(pose)
            log.info("pose %-12s <- %s (t=%s s, frame %s)", pid, cam["name"], cam.get("time_s"), cam.get("frame_index"))
    if labels:
        by_id = {p["id"]: p for p in data["poses"]}
        for pid, label in labels:
            if pid not in by_id:
                raise SystemExit(f"no pose {pid!r}; define it first with --set {pid} <image|frame|time>")
            by_id[pid]["label"] = label
    if sets or labels:
        # stable key order for readability
        data["poses"] = [
            {k: p[k] for k in ("id", "label", "position", "quaternion", "fov_y_deg", "source_image") if k in p}
            | {k: v for k, v in p.items() if k not in ("id", "label", "position", "quaternion", "fov_y_deg", "source_image")}
            for p in data["poses"]
        ]
        write_json(ws.poses_json, data)
        log.info("wrote %s (%d poses)", ws.poses_json, len(data["poses"]))
    if do_list or not (sets or labels):
        if not data["poses"]:
            print("no poses yet. Example: gsu poses --work W --set studio 12.5 --set shop frame_000400.jpg")
        for p in data["poses"]:
            pos = ", ".join(f"{v:.2f}" for v in p["position"])
            print(f"{p['id']:<14} {p.get('label', ''):<16} pos=({pos}) fov={p.get('fov_y_deg')} from {p.get('source_image')}")
