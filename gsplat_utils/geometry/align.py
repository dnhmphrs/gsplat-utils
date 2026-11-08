"""Gravity-align and scale the reconstruction.

Idea: during a handheld walk the phone is held roughly upright and at a roughly
constant height. So
  * the mean image-down vector of all cameras is a good first guess of gravity;
  * the heights of floor points *relative to the nearest camera* cluster at
    -camera_height, whatever floor of the building they are on; the direction
    that makes that cluster tightest is the refined up axis, and the cluster
    position gives metres-per-model-unit once a camera height is assumed.
Outputs scene.json with a 4x4 similarity (aligned <- model): +Z up, z=0 on the
floor under the first camera, +X along the walk's main horizontal axis.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, asdict

import numpy as np
from scipy.spatial import cKDTree

from ..common import log, read_json, write_json
from ..paths import Workspace
from ..sfm.model import SceneModel
from .points import load_scene


def _unit(v: np.ndarray) -> np.ndarray:
    return v / max(np.linalg.norm(v), 1e-12)


def up_from_cameras(model: SceneModel) -> np.ndarray:
    down = model.image_down
    m = _unit(down.mean(axis=0))
    for _ in range(3):  # trim outliers (looking at the floor/ceiling tilts the image y axis)
        cos = down @ m
        sel = cos > np.cos(np.deg2rad(35))
        if sel.sum() < 3:
            break
        m = _unit(down[sel].mean(axis=0))
    return -m


def _mode(values: np.ndarray, bins: int = 80) -> tuple[float, float]:
    """(centre, width) of the strongest histogram peak after light smoothing."""
    h, edges = np.histogram(values, bins=bins)
    k = np.array([1, 2, 3, 2, 1], float); k /= k.sum()
    hs = np.convolve(h, k, mode="same")
    i = int(np.argmax(hs))
    w = edges[1] - edges[0]
    return float(0.5 * (edges[i] + edges[i + 1])), float(w)


@dataclass
class FloorFit:
    up: np.ndarray
    cam_floor_dist: float      # model units, median camera height above floor
    ceiling_dist: float | None # model units, median ceiling height above camera (None if unseen)
    n_floor: int
    n_ceiling: int


def _plane_basis(up: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    a = np.array([1.0, 0, 0]) if abs(up[0]) < 0.9 else np.array([0, 1.0, 0])
    e1 = _unit(a - (a @ up) * up)
    e2 = np.cross(up, e1)
    return e1, e2


def _floor_band(h: np.ndarray, bins: int = 100) -> tuple[float, float] | None:
    """Locate the floor as the *lower edge* of the height distribution.

    Sparse SfM points sit mostly on textured things at eye level, so the floor is
    rarely the strongest peak; but nothing (except noise) lies below it. We take
    the lowest bin whose smoothed count reaches 30% of the maximum and the local
    maximum just above it. Returns (centre, half_width) or None."""
    if len(h) < 50:
        return None
    lo, hi = np.quantile(h, [0.01, 0.995])
    if hi - lo <= 0:
        return None
    cnt, edges = np.histogram(h, bins=bins, range=(lo, hi))
    k = np.array([1, 2, 3, 2, 1], float); k /= k.sum()
    hs = np.convolve(cnt, k, mode="same")
    thr = 0.3 * hs.max()
    i0 = int(np.argmax(hs >= thr))
    i1 = min(len(hs), i0 + 8)
    ip = i0 + int(np.argmax(hs[i0:i1]))
    w = edges[1] - edges[0]
    return float(0.5 * (edges[ip] + edges[ip + 1])), float(2.0 * w)


def fit_floor(points: np.ndarray, cams: np.ndarray, up0: np.ndarray, refine_up: bool = True) -> FloorFit:
    """Heights of points relative to the *horizontally* nearest camera (camera
    track smoothed along the sequence to remove hand-held bobbing). The floor is
    the lower edge of that distribution; the ceiling the upper edge."""
    from scipy.ndimage import gaussian_filter1d

    if len(cams) >= 7:
        cams_s = np.stack([gaussian_filter1d(cams[:, i], 3.0, mode="nearest") for i in range(3)], 1)
    else:
        cams_s = cams.copy()
    up = up0.copy()
    floor_mask = np.zeros(len(points), bool)
    h = np.zeros(len(points))
    d = points - cams_s[0]
    near_ok = np.ones(len(points), bool)
    for it in range(3):
        e1, e2 = _plane_basis(up)
        B = np.stack([e1, e2], 1)
        dist, near = cKDTree(cams_s @ B).query(points @ B, k=1)
        # only points reasonably close to the track: far points may belong to other floors
        near_ok = dist <= np.quantile(dist, 0.85)
        d = points - cams_s[near]
        h = d @ up
        band = _floor_band(h[near_ok & (h < 0)])
        if band is None:
            break
        peak, tol = band
        floor_mask = near_ok & (np.abs(h - peak) < tol)
        if not refine_up or floor_mask.sum() < 200:
            break
        dd = d[floor_mask]
        r = np.linalg.norm(dd, axis=1)
        dd = dd[r < np.quantile(r, 0.8)]
        w_, v_ = np.linalg.eigh(np.cov(dd.T))
        cand = _unit(v_[:, 0])
        if cand @ up < 0:
            cand = -cand
        ang0 = np.degrees(np.arccos(np.clip(cand @ up0, -1, 1)))
        # the floor normal must agree with the camera-based estimate; a planar
        # cluster of eye-level wall points would otherwise hijack the fit
        if ang0 > 10 or w_[0] > 0.05 * w_[1]:
            log.debug("floor refinement rejected (%.1f deg, eig ratio %.3f)", ang0, w_[0] / max(w_[1], 1e-12))
            break
        if np.degrees(np.arccos(np.clip(cand @ up, -1, 1))) < 0.05:
            up = cand
            break
        up = cand
    floor_vals = h[floor_mask]
    cam_floor = float(-np.median(floor_vals)) if len(floor_vals) else float("nan")
    # ceiling: upper edge of the distribution above the cameras
    ceil = None
    n_ceil = 0
    above = near_ok & (h > 0)
    if above.sum() > 50:
        band = _floor_band(-h[above])
        if band is not None:
            pc, tol = -band[0], band[1]
            cm = above & (np.abs(h - pc) < tol)
            if cm.sum() > 30 and np.isfinite(cam_floor) and pc > 0.15 * cam_floor:
                ceil = float(np.median(h[cm]))
                n_ceil = int(cm.sum())
    return FloorFit(up, cam_floor, ceil, int(floor_mask.sum()), n_ceil)


def scale_from_walking_speed(model: SceneModel, speed_m_s: float = 1.0) -> float | None:
    """Fallback metric scale: a person filming a walk-through moves at roughly
    1 m/s. Uses the video timestamps of the registered frames."""
    if model.time_s is None or np.all(np.isnan(model.time_s)) or model.n_images < 10:
        return None
    from scipy.ndimage import gaussian_filter1d

    t = model.time_s
    ok = ~np.isnan(t)
    pos = np.stack([gaussian_filter1d(model.positions[ok][:, i], 2.0, mode="nearest") for i in range(3)], 1)
    tt = t[ok]
    # speeds over ~1 s windows
    step = max(1, int(round(1.0 / max(np.median(np.diff(tt)), 1e-3))))
    dp = np.linalg.norm(pos[step:] - pos[:-step], axis=1)
    dt = tt[step:] - tt[:-step]
    good = dt > 1e-3
    if good.sum() < 5:
        return None
    v = dp[good] / dt[good]                    # model units per second
    v = v[v > np.quantile(v, 0.3)]             # ignore pauses
    if len(v) == 0 or np.median(v) <= 0:
        return None
    return float(speed_m_s / np.median(v))


def rotation_from_up_and_heading(up: np.ndarray, heading: np.ndarray) -> np.ndarray:
    """Rows of R are the new x, y, z axes expressed in the old frame."""
    z = _unit(up)
    x = heading - (heading @ z) * z
    if np.linalg.norm(x) < 1e-6:
        x = np.array([1.0, 0, 0]) if abs(z[0]) < 0.9 else np.array([0, 1.0, 0])
        x = x - (x @ z) * z
    x = _unit(x)
    y = np.cross(z, x)
    return np.stack([x, y, z])


def principal_heading(positions: np.ndarray, up: np.ndarray) -> np.ndarray:
    """Dominant horizontal direction of the walk modulo 90 degrees (so the
    aligned +X axis follows the building's corridor grid), pointed along the
    overall start->end displacement."""
    from scipy.ndimage import gaussian_filter1d

    from .path import dominant_angle

    p = positions - positions.mean(axis=0)
    p = p - np.outer(p @ up, up)
    e1 = _unit(p[np.argmax(np.linalg.norm(p, axis=1))]) if len(p) > 1 else np.array([1.0, 0, 0])
    if np.linalg.norm(e1) < 1e-9 or abs(e1 @ up) > 0.99:
        e1 = _unit(np.cross(up, np.array([0, 0, 1.0]) if abs(up[2]) < 0.9 else np.array([1.0, 0, 0])))
    e2 = np.cross(up, e1)
    q = np.stack([p @ e1, p @ e2], 1)
    if len(q) >= 7:
        q = np.stack([gaussian_filter1d(q[:, i], 2.0, mode="nearest") for i in range(2)], 1)
    t = np.diff(q, axis=0)
    L = np.linalg.norm(t, axis=1)
    ok = L > 1e-6
    if ok.sum() < 2:
        return e1
    theta = dominant_angle(t[ok] / L[ok, None], weights=L[ok])
    h2 = np.array([np.cos(theta), np.sin(theta)])
    disp = q[-1] - q[0]
    # choose among the 4 axis directions the one closest to the overall displacement
    cands = [h2, -h2, np.array([-h2[1], h2[0]]), np.array([h2[1], -h2[0]])]
    h2 = max(cands, key=lambda c: c @ disp) if np.linalg.norm(disp) > 1e-6 else h2
    return _unit(h2[0] * e1 + h2[1] * e2)


@dataclass
class SceneAlignment:
    transform: np.ndarray        # 4x4 aligned <- model  (x' = s R x + t)
    rotation: np.ndarray         # 3x3
    translation: np.ndarray      # 3
    scale: float                 # metres per model unit
    up_model: np.ndarray         # up axis in model frame
    camera_height_m: float
    ceiling_height_m: float | None
    source_points: str
    method: str

    def apply(self, model: SceneModel) -> SceneModel:
        return model.transformed(self.rotation, self.translation, self.scale)

    def apply_points(self, xyz: np.ndarray) -> np.ndarray:
        return (xyz @ self.rotation.T) * self.scale + self.translation

    def to_json(self) -> dict:
        d = asdict(self)
        d["frame"] = "aligned"
        d["convention"] = "+Z up, metres, z=0 floor under first camera, +X main walk axis; x_aligned = scale*R*x_model + t"
        return d

    @classmethod
    def from_json(cls, d: dict) -> "SceneAlignment":
        return cls(
            np.array(d["transform"]), np.array(d["rotation"]), np.array(d["translation"]), float(d["scale"]),
            np.array(d["up_model"]), float(d["camera_height_m"]), d.get("ceiling_height_m"),
            d.get("source_points", "sfm"), d.get("method", ""),
        )

    @classmethod
    def load(cls, path) -> "SceneAlignment":
        return cls.from_json(read_json(path))


def align_scene(model: SceneModel, camera_height: float = 1.5, scale: float | None = None,
                up_method: str = "auto", source_name: str = "sfm", walking_speed: float = 1.0) -> SceneAlignment:
    from .path import track_inliers

    keep = track_inliers(model.positions)
    if not keep.all():
        log.info("ignoring %d mis-registered cameras for alignment", int((~keep).sum()))
        model = model.subset(keep)
    up0 = up_from_cameras(model)
    method = "cameras"
    fit = None
    up = up0
    if up_method in ("auto", "floor") and len(model.points) > 200:
        fit = fit_floor(model.points, model.positions, up0, refine_up=(up_method != "cameras"))
        if np.isfinite(fit.cam_floor_dist) and fit.n_floor >= 50:
            up = fit.up
            method = "cameras+floor"
        else:
            log.warning("floor not found (%d floor pts); using camera up only", fit.n_floor if fit else 0)
    ang = np.degrees(np.arccos(np.clip(up @ up0, -1, 1)))
    log.info("up axis: %s (%.1f deg from camera estimate, method=%s)", np.round(up, 4), ang, method)

    speed_scale = scale_from_walking_speed(model, walking_speed)
    scale_method = "user"
    if scale is None:
        floor_scale = None
        if fit is not None and np.isfinite(fit.cam_floor_dist) and fit.cam_floor_dist > 0:
            floor_scale = camera_height / fit.cam_floor_dist
        plausible = True
        if floor_scale is not None and speed_scale is not None:
            ratio = floor_scale / speed_scale
            plausible = 0.1 < ratio < 10.0      # only catches gross failures (implied speed 0.1..10 m/s)
            if not plausible:
                log.warning("floor-based scale %.4f implies a walking speed of %.2f m/s; using the %.1f m/s "
                            "walking-speed prior instead (pass --scale or --camera-height to override)",
                            floor_scale, walking_speed * ratio, walking_speed)
        if floor_scale is not None and plausible:
            scale = floor_scale
            scale_method = "camera_height"
            log.info("metric scale from camera height %.2fm: %.4f m/unit (floor %d pts)", camera_height, scale, fit.n_floor)
        elif speed_scale is not None:
            scale = speed_scale
            scale_method = "walking_speed"
            log.info("metric scale from walking speed %.1f m/s: %.4f m/unit", walking_speed, scale)
        else:
            scale = 1.0
            scale_method = "none"
            log.warning("no floor and no timestamps; scale left at 1.0 (pass --scale)")
    if speed_scale is not None:
        log.info("implied walking speed: %.2f m/s", walking_speed * scale / speed_scale)
    ceiling = None
    if fit is not None and fit.ceiling_dist is not None and np.isfinite(fit.cam_floor_dist):
        ceiling = float((fit.ceiling_dist + fit.cam_floor_dist) * scale)
        if 2.0 <= ceiling <= 6.0:
            log.info("ceiling height estimate %.2fm (%d pts)", ceiling, fit.n_ceiling)
        else:
            log.info("ceiling estimate %.2fm looks implausible; ignored", ceiling)
            ceiling = None

    heading = principal_heading(model.positions, up)
    R = rotation_from_up_and_heading(up, heading)
    c0 = model.positions[0]
    # origin: first camera, floor under it at z=0
    t = -(R @ c0) * scale + np.array([0.0, 0.0, camera_height])
    T = np.eye(4)
    T[:3, :3] = R * scale
    T[:3, 3] = t
    return SceneAlignment(T, R, t, float(scale), up, float(camera_height), ceiling, source_name,
                          f"{method}; scale={scale_method}")


def run_align(ws: Workspace, camera_height: float = 1.5, scale: float | None = None,
              up_method: str = "auto", source: str = "auto", model_dir: os.PathLike | str | None = None,
              walking_speed: float = 1.0) -> SceneAlignment:
    model, src = load_scene(ws, source=source, model_dir=model_dir, use_rectified=False)
    al = align_scene(model, camera_height=camera_height, scale=scale, up_method=up_method, source_name=src,
                     walking_speed=walking_speed)
    write_json(ws.scene_json, al.to_json())
    log.info("wrote %s", ws.scene_json)
    if (ws.root / "rectified" / "cameras.json").exists():
        log.info("note: work/rectified exists and takes precedence in `gsu geometry` (delete it or re-run `gsu rectify`)")
    return al
