"""Stitch the sub-models of one continuous walk back together.

When the sequential chain breaks (blur, a fast turn on the stairs) GLOMAP/COLMAP
return several models that share no images, so they cannot be merged
geometrically. But they come from one walk: each model is gravity-aligned and
metrically scaled on its own (camera height prior), and then the camera tracks
are registered where the models interleave in time, with a 2D rigid transform
(yaw + xy) plus a height offset. The result is one scene in the aligned frame
(metres, +Z up) that the geometry stage can use directly.
"""
from __future__ import annotations

import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..common import log, read_json, write_json
from ..export.ply import write_point_ply
from ..geometry.align import SceneAlignment, align_scene
from ..paths import Workspace
from .model import SceneModel, write_cameras_json


@dataclass
class StitchResult:
    model: SceneModel
    parts: list[dict] = field(default_factory=list)
    alignment: SceneAlignment | None = None


def _time_order(models: list[SceneModel]) -> list[int]:
    return sorted(range(len(models)), key=lambda k: float(np.nanmedian(models[k].time_s)))


def _rigid2d(A: np.ndarray, B: np.ndarray) -> tuple[float, np.ndarray]:
    """theta, t such that A ~= R(theta) B + t (least squares)."""
    ca, cb = A.mean(axis=0), B.mean(axis=0)
    a, b = A - ca, B - cb
    H = b.T @ a
    theta = float(np.arctan2(H[0, 1] - H[1, 0], H[0, 0] + H[1, 1]))
    R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    return theta, ca - R @ cb


def register_tracks(ta: np.ndarray, pa: np.ndarray, tb: np.ndarray, pb: np.ndarray,
                    max_dt: float = 0.25, inlier_m: float = 0.35, seed: int = 0) -> dict | None:
    """Register track B onto track A using frames that are close in time.

    Returns {"theta", "t": (2,), "dz", "scale", "n_pairs", "n_inliers", "span_m", "rms_m"} or None.
    """
    ok_a = ~np.isnan(ta); ok_b = ~np.isnan(tb)
    ta, pa, tb, pb = ta[ok_a], pa[ok_a], tb[ok_b], pb[ok_b]
    if len(ta) < 3 or len(tb) < 3:
        return None
    order = np.argsort(ta); ta, pa = ta[order], pa[order]
    j = np.searchsorted(ta, tb)
    pairs = []
    for k, (t, p) in enumerate(zip(tb, pb)):
        cands = [i for i in (j[k] - 1, j[k]) if 0 <= i < len(ta)]
        if not cands:
            continue
        i = min(cands, key=lambda i: abs(ta[i] - t))
        if abs(ta[i] - t) <= max_dt:
            pairs.append((i, k))
    if len(pairs) < 4:
        return None
    A = np.array([pa[i] for i, _ in pairs]); B = np.array([pb[k] for _, k in pairs])
    span = float(np.linalg.norm(A[:, :2].max(axis=0) - A[:, :2].min(axis=0)))
    rng = np.random.default_rng(seed)
    best = None
    n = len(pairs)
    for _ in range(300):
        s = rng.choice(n, 2, replace=False)
        if np.linalg.norm(A[s[0], :2] - A[s[1], :2]) < 0.3:
            continue
        theta, t = _rigid2d(A[s, :2], B[s, :2])
        R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
        res = np.linalg.norm(A[:, :2] - (B[:, :2] @ R.T + t), axis=1)
        inl = res < inlier_m
        if best is None or inl.sum() > best[0]:
            best = (int(inl.sum()), inl)
    if best is None or best[0] < 4:
        return None
    inl = best[1]
    theta, t = _rigid2d(A[inl, :2], B[inl, :2])
    R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    res = np.linalg.norm(A[:, :2] - (B[:, :2] @ R.T + t), axis=1)
    inl = res < inlier_m
    theta, t = _rigid2d(A[inl, :2], B[inl, :2])
    R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    res = np.linalg.norm(A[:, :2] - (B[:, :2] @ R.T + t), axis=1)
    dz = float(np.median(A[inl, 2] - B[inl, 2]))
    # relative scale check (both models already metric): only trust it over a decent span
    da = np.linalg.norm(A[inl, :2] - A[inl, :2].mean(axis=0), axis=1).mean()
    db = np.linalg.norm(B[inl, :2] - B[inl, :2].mean(axis=0), axis=1).mean()
    scale = float(da / db) if db > 1e-6 else 1.0
    return {"theta": theta, "t": t, "dz": dz, "scale": scale, "n_pairs": int(n), "n_inliers": int(inl.sum()),
            "span_m": span, "rms_m": float(np.sqrt(np.mean(res[inl] ** 2)))}


def _apply_2d(model: SceneModel, theta: float, t: np.ndarray, dz: float, scale: float = 1.0) -> SceneModel:
    c, s = np.cos(theta), np.sin(theta)
    R = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    return model.transformed(R, np.array([t[0], t[1], dz]), scale)


def stitch_models(models: list[SceneModel], camera_height: float = 1.5, walking_speed: float = 1.0,
                  apply_scale: bool = True) -> StitchResult:
    """Align every model, then chain them in time order onto the first one."""
    if not models:
        raise ValueError("no models")
    order = _time_order(models)
    aligned: list[SceneModel] = []
    parts: list[dict] = []
    for k in order:
        m = models[k]
        al = align_scene(m, camera_height=camera_height, walking_speed=walking_speed)
        am = al.apply(m)
        aligned.append(am)
        parts.append({"index": int(k), "n_images": int(m.n_images), "n_points": int(len(m.points)),
                      "t0": float(np.nanmin(m.time_s)), "t1": float(np.nanmax(m.time_s)),
                      "scale_m_per_unit": al.scale, "method": al.method})
    merged = aligned[0]
    parts[0]["transform"] = "reference"
    for i in range(1, len(aligned)):
        B = aligned[i]
        reg = register_tracks(merged.time_s, merged.positions, B.time_s, B.positions)
        if reg is not None:
            sc = reg["scale"] if (apply_scale and reg["span_m"] >= 3.0 and 0.85 <= reg["scale"] <= 1.18) else 1.0
            if sc != 1.0:
                # rescale about B's centroid before the rigid fit is applied
                cB = B.positions.mean(axis=0)
                B = B.transformed(np.eye(3), cB * (1 - sc), sc)
                reg = register_tracks(merged.time_s, merged.positions, B.time_s, B.positions) or reg
            B = _apply_2d(B, reg["theta"], reg["t"], reg["dz"])
            parts[i]["transform"] = {"yaw_deg": float(np.degrees(reg["theta"])), "t": reg["t"].tolist(),
                                     "dz": reg["dz"], "scale": sc, "n_pairs": reg["n_pairs"],
                                     "n_inliers": reg["n_inliers"], "span_m": reg["span_m"], "rms_m": reg["rms_m"],
                                     "method": "time-interleaved tracks"}
            log.info("stitched part %d: yaw %.1f deg, dz %.2f m, scale %.3f, %d/%d pairs, rms %.2f m over %.1f m",
                     parts[i]["index"], np.degrees(reg["theta"]), reg["dz"], sc, reg["n_inliers"], reg["n_pairs"],
                     reg["rms_m"], reg["span_m"])
        else:
            # no temporal overlap: continue the walk from the end of the merged track
            pa, pb = merged.positions, B.positions
            ta = pa[-1] - pa[max(0, len(pa) - 10)]
            tb = pb[min(10, len(pb) - 1)] - pb[0]
            theta = float(np.arctan2(ta[1], ta[0]) - np.arctan2(tb[1], tb[0])) if np.linalg.norm(ta[:2]) > 1e-6 and np.linalg.norm(tb[:2]) > 1e-6 else 0.0
            c, s_ = np.cos(theta), np.sin(theta)
            R2 = np.array([[c, -s_], [s_, c]])
            t = pa[-1, :2] - R2 @ pb[0, :2]
            dz = float(pa[-1, 2] - pb[0, 2])
            B = _apply_2d(B, theta, t, dz)
            parts[i]["transform"] = {"yaw_deg": float(np.degrees(theta)), "t": t.tolist(), "dz": dz, "scale": 1.0,
                                     "method": "end-to-start continuity (no temporal overlap: low confidence)"}
            log.warning("part %d has no temporal overlap with the rest; attached end-to-start (low confidence)",
                        parts[i]["index"])
        merged = _concat(merged, B)
    al = SceneAlignment(np.eye(4), np.eye(3), np.zeros(3), 1.0, np.array([0.0, 0.0, 1.0]), camera_height,
                        None, "sfm", "stitched: parts aligned independently, registered by time-interleaved tracks")
    return StitchResult(merged, parts, al)


def _concat(a: SceneModel, b: SceneModel) -> SceneModel:
    names = a.names + b.names
    order = np.argsort(np.array([*a.time_s, *b.time_s]))
    c2w = np.concatenate([a.camera_to_world, b.camera_to_world])[order]
    K = np.concatenate([a.K, b.K])[order]
    sizes = np.concatenate([a.sizes, b.sizes])[order]
    ids = np.concatenate([a.image_ids, b.image_ids])[order] if len(a.image_ids) and len(b.image_ids) else np.zeros(0, int)
    cm = [(a.camera_models + b.camera_models)[i] for i in order] if a.camera_models and b.camera_models else []
    cp = [(a.camera_params + b.camera_params)[i] for i in order] if a.camera_params and b.camera_params else []
    m = SceneModel([names[i] for i in order], c2w, K, sizes, np.concatenate([a.points, b.points]),
                   np.concatenate([a.colors, b.colors]), np.concatenate([a.point_errors, b.point_errors]),
                   np.concatenate([a.track_lengths, b.track_lengths]), ids, cm, cp)
    m.frame_index = np.concatenate([a.frame_index, b.frame_index])[order]
    m.time_s = np.concatenate([a.time_s, b.time_s])[order]
    m.source = "stitched"
    return m


def write_colmap_text(model: SceneModel, out_dir: os.PathLike | str) -> None:
    """Write a minimal COLMAP text model (poses + intrinsics + points, no 2D tracks)."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    from scipy.spatial.transform import Rotation

    # one camera entry per distinct intrinsics
    cams: dict[tuple, int] = {}
    with open(out / "cameras.txt", "w") as fc, open(out / "images.txt", "w") as fi:
        fc.write("# Camera list with one line of data per camera:\n#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
        fi.write("# Image list with two lines of data per image:\n#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n#   POINTS2D[] as (X, Y, POINT3D_ID)\n")
        for i, name in enumerate(model.names):
            K = model.K[i]; w, h = int(model.sizes[i][0]), int(model.sizes[i][1])
            mdl = model.camera_models[i] if model.camera_models else "PINHOLE"
            params = model.camera_params[i] if model.camera_params else [K[0, 0], K[1, 1], K[0, 2], K[1, 2]]
            key = (mdl, w, h, tuple(np.round(params, 6)))
            if key not in cams:
                cams[key] = len(cams) + 1
                fc.write(f"{cams[key]} {mdl} {w} {h} " + " ".join(f"{p:.8f}" for p in params) + "\n")
            w2c = np.linalg.inv(model.camera_to_world[i])
            q = Rotation.from_matrix(w2c[:3, :3]).as_quat()  # xyzw
            t = w2c[:3, 3]
            fi.write(f"{i + 1} {q[3]:.9f} {q[0]:.9f} {q[1]:.9f} {q[2]:.9f} {t[0]:.6f} {t[1]:.6f} {t[2]:.6f} {cams[key]} {name}\n\n")
    with open(out / "points3D.txt", "w") as fp:
        fp.write("# 3D point list with one line of data per point:\n#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)\n")
        err = model.point_errors if len(model.point_errors) == len(model.points) else np.zeros(len(model.points))
        for j, (p, c) in enumerate(zip(model.points, model.colors)):
            fp.write(f"{j + 1} {p[0]:.6f} {p[1]:.6f} {p[2]:.6f} {int(c[0])} {int(c[1])} {int(c[2])} {float(err[j]):.4f}\n")


def run_stitch(ws: Workspace, model_dirs: list[os.PathLike | str] | None = None, camera_height: float = 1.5,
               walking_speed: float = 1.0, min_images: int = 30, undistort: bool = True) -> StitchResult:
    """Stitch sub-models (default: every model in colmap/sparse/models with >= min_images)
    and make the result the workspace model: colmap/sparse/0 (text), cameras.json,
    points_sfm.ply and an identity scene.json (the model is already aligned)."""
    if model_dirs is None:
        root = ws.sparse / "models"
        cands = sorted((d for d in root.iterdir() if d.is_dir() and (d / "images.bin").exists()), key=lambda d: d.name) if root.exists() else []
        if not cands and ws.model.exists():
            cands = [ws.model]
        model_dirs = cands
    models = []
    for d in model_dirs:
        m = SceneModel.from_colmap(d, ws.frames_meta)
        if m.n_images >= min_images:
            models.append(m)
        else:
            log.info("skipping %s (%d images < %d)", d, m.n_images, min_images)
    if not models:
        raise SystemExit("no models to stitch")
    if any(m.time_s is None or np.all(np.isnan(m.time_s)) for m in models):
        raise SystemExit("stitching needs video timestamps in frames_meta.json")
    res = stitch_models(models, camera_height=camera_height, walking_speed=walking_speed)
    merged = res.model
    # keep the previous selected model, install the stitched one
    if ws.model.exists():
        dst = ws.sparse / "models" / "before_stitch"
        if dst.exists():
            shutil.rmtree(dst)
        shutil.move(str(ws.model), str(dst))
    write_colmap_text(merged, ws.model)
    write_cameras_json(merged, ws.cameras_json, frame="stitched-aligned")
    write_point_ply(ws.points_ply, merged.points, merged.colors)
    write_json(ws.scene_json, res.alignment.to_json())
    write_json(ws.colmap / "stitch.json", {"parts": res.parts, "n_images": merged.n_images, "n_points": int(len(merged.points))})
    log.info("stitched %d parts -> %d cameras, %d points (frame: aligned metres, scene.json = identity)",
             len(models), merged.n_images, len(merged.points))
    if undistort:
        try:
            import pycolmap

            if ws.undistorted.exists():
                shutil.rmtree(ws.undistorted)
            pycolmap.undistort_images(output_path=ws.undistorted, input_path=ws.model, image_path=ws.frames,
                                      output_type="COLMAP")
        except Exception as exc:  # pragma: no cover
            log.warning("undistortion of the stitched model failed: %s", exc)
    return res
