"""`gsu rectify`: turn the raw COLMAP sub-models into one metric, gravity-aligned,
drift-corrected scene for the geometry stage.

  for each sub-model (>= min_images):  align (up, floor, scale)  ->  drift correction
  then register the parts on each other where they interleave in time (stitch)
  -> work/rectified/{cameras.json, points.ply, scene.json (identity), rectify.json}

The original COLMAP model in colmap/sparse/0 is left untouched: splat training
needs the self-consistent raw reconstruction, the diagram needs the metric one.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from ..common import log, write_json
from ..export.ply import write_point_ply
from ..geometry.align import SceneAlignment, align_scene
from ..geometry.drift import correct_drift
from ..geometry.path import track_inliers
from ..paths import Workspace
from .model import SceneModel, write_cameras_json
from .stitch import _concat, register_tracks, _apply_2d


def rectified_dir(ws: Workspace) -> Path:
    return ws.root / "rectified"


def _refit_translation(A: SceneModel, B: SceneModel, R2: np.ndarray, reg: dict, max_dt: float = 0.25) -> np.ndarray:
    """Translation that best overlays B on A for time-close camera pairs, given a fixed rotation."""
    ta, pa, tb, pb = A.time_s, A.positions, B.time_s, B.positions
    oa = np.argsort(ta); ta, pa = ta[oa], pa[oa]
    j = np.searchsorted(ta, tb)
    diffs = []
    for k, (t, p) in enumerate(zip(tb, pb)):
        cands = [i for i in (j[k] - 1, j[k]) if 0 <= i < len(ta)]
        if not cands:
            continue
        i = min(cands, key=lambda i: abs(ta[i] - t))
        if abs(ta[i] - t) <= max_dt:
            diffs.append(pa[i, :2] - R2 @ p[:2])
    return np.median(np.array(diffs), axis=0) if diffs else np.asarray(reg["t"])


def run_rectify(ws: Workspace, model_dirs: list[os.PathLike | str] | None = None, camera_height: float = 1.5,
                walking_speed: float = 1.0, min_images: int = 30, drift: bool = True,
                window_s: float = 4.0, grid_snap: bool = True, grid_snap_deg: float = 15.0) -> SceneModel:
    if model_dirs is None:
        root = ws.sparse / "models"
        cands = sorted((d for d in root.iterdir() if d.is_dir() and (d / "images.bin").exists()),
                       key=lambda d: d.name) if root.exists() else []
        if not cands and ws.model.exists():
            cands = [ws.model]
        model_dirs = cands
    models = []
    for d in model_dirs:
        m = SceneModel.from_colmap(d, ws.frames_meta)
        if m.n_images >= min_images:
            models.append((Path(d), m))
        else:
            log.info("skipping %s (%d images < %d)", d, m.n_images, min_images)
    if not models:
        raise SystemExit("no models to rectify (run `gsu sfm` first)")
    has_time = all(m.time_s is not None and not np.all(np.isnan(m.time_s)) for _, m in models)
    if not has_time and (drift or len(models) > 1):
        log.warning("no timestamps (photo set?): drift correction and stitching need them; aligning only")
        drift = False
        models = models[:1]
    # time order
    if has_time:
        models.sort(key=lambda dm: float(np.nanmedian(dm[1].time_s)))

    parts = []
    aligned_parts: list[SceneModel] = []
    factors: dict[str, float] = {}
    for d, m in models:
        keep = track_inliers(m.positions)
        if not keep.all():
            m = m.subset(keep)
        al = align_scene(m, camera_height=camera_height, walking_speed=walking_speed)
        am = al.apply(m)
        info = {"model": str(d), "n_images": int(am.n_images), "n_points": int(len(am.points)),
                "t0": float(np.nanmin(am.time_s)) if has_time else None,
                "t1": float(np.nanmax(am.time_s)) if has_time else None,
                "align_scale_m_per_unit": al.scale, "align_method": al.method}
        if drift:
            am, prof = correct_drift(am, camera_height=camera_height, window_s=window_s)
            factors.update({n: float(f) for n, f in zip(am.names, prof.per_camera)})
            info["drift"] = {"t": prof.t.round(2).tolist(), "scale": [None if not np.isfinite(v) else round(float(v), 4) for v in prof.scale],
                             "n_points": prof.n_points.tolist(),
                             "factor_min": float(np.nanmin(prof.per_camera)), "factor_max": float(np.nanmax(prof.per_camera))}
        parts.append(info)
        aligned_parts.append(am)

    merged = aligned_parts[0]
    parts[0]["transform"] = "reference"
    for i in range(1, len(aligned_parts)):
        B = aligned_parts[i]
        reg = register_tracks(merged.time_s, merged.positions, B.time_s, B.positions)
        if reg is not None:
            theta = reg["theta"]
            snapped = float(np.round(theta / (np.pi / 2)) * (np.pi / 2))
            if grid_snap and abs(np.degrees(theta - snapped)) <= grid_snap_deg:
                # both parts are aligned to their own corridor grid; a short overlap cannot
                # measure the yaw better than the grid prior
                c, s_ = np.cos(snapped), np.sin(snapped)
                R2 = np.array([[c, -s_], [s_, c]])
                # re-fit the translation with the snapped rotation on the inlier pairs
                reg_t = _refit_translation(merged, B, R2, reg)
                theta, t = snapped, reg_t
            else:
                t = reg["t"]
            B = _apply_2d(B, theta, t, reg["dz"])
            parts[i]["transform"] = {"yaw_deg": float(np.degrees(theta)), "yaw_measured_deg": float(np.degrees(reg["theta"])),
                                     "t": np.asarray(t).tolist(), "dz": reg["dz"],
                                     "n_pairs": reg["n_pairs"], "n_inliers": reg["n_inliers"], "span_m": reg["span_m"],
                                     "rms_m": reg["rms_m"], "method": "time-interleaved tracks"
                                     + (" (yaw snapped to the building grid)" if theta != reg["theta"] else "")}
            log.info("stitched %s: yaw %.1f deg, dz %.2f m, %d/%d pairs, rms %.2f m over %.1f m", parts[i]["model"],
                     np.degrees(reg["theta"]), reg["dz"], reg["n_inliers"], reg["n_pairs"], reg["rms_m"], reg["span_m"])
        else:
            pa, pb = merged.positions, B.positions
            ta = pa[-1] - pa[max(0, len(pa) - 10)]; tb = pb[min(10, len(pb) - 1)] - pb[0]
            theta = float(np.arctan2(ta[1], ta[0]) - np.arctan2(tb[1], tb[0])) if np.linalg.norm(ta[:2]) > 1e-6 and np.linalg.norm(tb[:2]) > 1e-6 else 0.0
            # both parts were aligned to their own dominant corridor direction, i.e. to
            # the same building grid: the relative yaw is a multiple of 90 degrees
            theta_raw = theta
            theta = float(np.round(theta / (np.pi / 2)) * (np.pi / 2))
            c, s_ = np.cos(theta), np.sin(theta)
            t = pa[-1, :2] - np.array([[c, -s_], [s_, c]]) @ pb[0, :2]
            dz = float(pa[-1, 2] - pb[0, 2])
            B = _apply_2d(B, theta, t, dz)
            parts[i]["transform"] = {"yaw_deg": float(np.degrees(theta)), "yaw_raw_deg": float(np.degrees(theta_raw)),
                                     "t": t.tolist(), "dz": dz,
                                     "method": "end-to-start continuity snapped to the building grid (no temporal overlap: low confidence)"}
            log.warning("%s has no temporal overlap with the rest; attached end-to-start (low confidence)", parts[i]["model"])
        merged = _concat(merged, B)

    out = rectified_dir(ws)
    out.mkdir(parents=True, exist_ok=True)
    write_cameras_json(merged, out / "cameras.json", frame="rectified")
    write_point_ply(out / "points.ply", merged.points, merged.colors)
    al = SceneAlignment(np.eye(4), np.eye(3), np.zeros(3), 1.0, np.array([0.0, 0.0, 1.0]), camera_height, None,
                        "sfm", "rectified: per-part alignment, drift correction, time-stitched")
    write_json(out / "scene.json", al.to_json())
    write_json(out / "rectify.json", {"parts": parts, "n_images": merged.n_images, "n_points": int(len(merged.points)),
                                      "camera_height_m": camera_height, "drift_corrected": bool(drift)})
    write_json(out / "factors.json", {"note": "local scale factor applied per camera; > ~4 means the raw "
                                              "reconstruction had collapsed there and its points are unreliable",
                                      "factors": factors})
    # speed sanity after rectification
    if has_time:
        t = merged.time_s; P = merged.positions
        o = np.argsort(t); dt = np.diff(t[o]); dp = np.linalg.norm(np.diff(P[o], axis=0), axis=1)
        okk = dt > 1e-3
        v = dp[okk] / dt[okk]
        log.info("rectified walk: %d cameras, %.1f m of path, median speed %.2f m/s (p90 %.2f)", merged.n_images,
                 float(dp.sum()), float(np.median(v)), float(np.quantile(v, 0.9)))
    log.info("wrote %s (cameras.json, points.ply, scene.json, rectify.json)", out)
    return merged
