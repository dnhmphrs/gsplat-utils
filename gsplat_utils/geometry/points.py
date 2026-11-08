"""One place to load 'scene points' for the geometry stages, from either the
COLMAP sparse model or the trained Gaussian splat (gaussian centres)."""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from ..common import log
from ..export.ply import read_gaussian_ply, read_point_ply
from ..paths import Workspace
from ..sfm.model import SceneModel


def gaussian_points(
    ply_path: os.PathLike | str,
    min_alpha: float = 0.3,
    max_scale_quantile: float = 0.98,
    max_points: int | None = 2_000_000,
) -> tuple[np.ndarray, np.ndarray]:
    """Gaussian centres filtered to the ones that represent surfaces:
    opaque enough and not huge 'sky/floater' blobs."""
    g = read_gaussian_ply(ply_path)
    alpha = g.alpha
    size = np.exp(g.scales).max(axis=1)
    keep = alpha >= min_alpha
    if 0 < max_scale_quantile < 1:
        keep &= size <= np.quantile(size[keep] if keep.any() else size, max_scale_quantile)
    xyz = g.means[keep].astype(np.float64)
    rgb = (g.rgb[keep] * 255).astype(np.uint8)
    if max_points and len(xyz) > max_points:
        sel = np.random.default_rng(0).choice(len(xyz), max_points, replace=False)
        xyz, rgb = xyz[sel], rgb[sel]
    log.info("splat points: %d / %d gaussians kept (alpha>=%.2f)", len(xyz), len(g), min_alpha)
    return xyz, rgb


def scene_json_for(ws: Workspace, model_dir: os.PathLike | str | None = None, use_rectified: bool = True) -> Path:
    """The alignment that belongs to the scene `load_scene` would return."""
    rect = ws.root / "rectified"
    if use_rectified and model_dir is None and (rect / "cameras.json").exists():
        return rect / "scene.json"
    return ws.scene_json


def load_scene(ws: Workspace, source: str = "auto", model_dir: os.PathLike | str | None = None,
               splat_ply: os.PathLike | str | None = None, use_rectified: bool = True) -> tuple[SceneModel, str]:
    """Return (SceneModel with points filled from the chosen source, source_name).

    Prefers work/rectified (drift-corrected, stitched, metric) when it exists and
    no explicit model_dir is given; otherwise the COLMAP model."""
    rect = ws.root / "rectified"
    splat_ply = Path(splat_ply) if splat_ply else ws.splat_ply
    if source == "auto":
        source = "splat" if splat_ply.exists() else "sfm"
    if use_rectified and model_dir is None and (rect / "cameras.json").exists():
        pts, cols = read_point_ply(rect / "points.ply")
        model = SceneModel.from_cameras_json(rect / "cameras.json", pts, cols)
        model.source = "rectified"
        log.info("scene: %d cameras, %d points (rectified; source=%s)", model.n_images, len(model.points), source)
        if source == "splat":
            log.warning("--source splat with a rectified scene: the splat is in the raw COLMAP frame, using SfM points instead")
        return model, "sfm"
    model_dir = Path(model_dir) if model_dir else ws.model
    if model_dir.exists():
        model = SceneModel.from_colmap(model_dir, ws.frames_meta)
    elif ws.cameras_json.exists():
        pts, cols = read_point_ply(ws.points_ply) if ws.points_ply.exists() else (None, None)
        model = SceneModel.from_cameras_json(ws.cameras_json, pts, cols)
    else:
        raise FileNotFoundError(f"no COLMAP model at {model_dir}; run `gsu sfm` first")
    if source == "splat":
        if not splat_ply.exists():
            raise FileNotFoundError(f"no trained splat at {splat_ply}; run `gsu train` or use --source sfm")
        xyz, rgb = gaussian_points(splat_ply)
        model.points, model.colors = xyz, rgb
        model.point_errors = np.zeros(len(xyz))
        model.track_lengths = np.zeros(len(xyz), dtype=int)
    else:
        # drop the least reliable SfM points
        if len(model.point_errors):
            keep = (model.track_lengths >= 3) & (model.point_errors <= np.quantile(model.point_errors, 0.9))
            if keep.sum() > 1000:
                model.points, model.colors = model.points[keep], model.colors[keep]
                model.point_errors, model.track_lengths = model.point_errors[keep], model.track_lengths[keep]
    log.info("scene: %d cameras, %d points (source=%s)", model.n_images, len(model.points), source)
    return model, source
