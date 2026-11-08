""".splat export for the web viewer (antimatter15 layout) + reader for tests.

Record (32 bytes, little endian):
    float32 x, y, z
    float32 sx, sy, sz            linear scales (exp of the PLY log-scales)
    uint8   r, g, b, a            r,g,b = clip(0.5 + C0*sh0)*255, a = sigmoid(opacity)*255
    uint8   qw, qx, qy, qz        each = clip(round(q*128 + 128), 0, 255), q normalised (w,x,y,z)
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from ..common import log, read_json, write_json
from ..paths import Workspace
from .ply import SH_C0, Gaussians, read_gaussian_ply, write_gaussian_ply

SPLAT_DTYPE = np.dtype([
    ("pos", "<f4", (3,)),
    ("scale", "<f4", (3,)),
    ("rgba", "u1", (4,)),
    ("rot", "u1", (4,)),
])
BYTES_PER_SPLAT = SPLAT_DTYPE.itemsize  # 32


def splat_records(g: Gaussians, sort: bool = True) -> np.ndarray:
    """Quantise gaussians into the .splat record array (optionally big-and-opaque first)."""
    n = len(g)
    scales_lin = np.exp(g.scales.astype(np.float64))
    alpha = 1.0 / (1.0 + np.exp(-g.opacities.astype(np.float64)))
    rgb = np.clip(0.5 + SH_C0 * g.sh0.astype(np.float64), 0.0, 1.0)
    q = g.quats.astype(np.float64)
    q = q / np.maximum(np.linalg.norm(q, axis=1, keepdims=True), 1e-12)
    order = np.arange(n)
    if sort and n:
        # antimatter15: largest volume * opacity first so progressive loading looks right
        key = -(scales_lin[:, 0] * scales_lin[:, 1] * scales_lin[:, 2]) * alpha
        order = np.argsort(key, kind="stable")
    rec = np.empty(n, dtype=SPLAT_DTYPE)
    rec["pos"] = g.means[order].astype(np.float32)
    rec["scale"] = scales_lin[order].astype(np.float32)
    rgba = np.concatenate([rgb[order] * 255.0, alpha[order, None] * 255.0], axis=1)
    rec["rgba"] = np.clip(np.round(rgba), 0, 255).astype(np.uint8)
    rec["rot"] = np.clip(np.round(q[order] * 128.0 + 128.0), 0, 255).astype(np.uint8)
    return rec


def write_splat(path: os.PathLike | str, g: Gaussians, sort: bool = True) -> int:
    rec = splat_records(g, sort=sort)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        f.write(rec.tobytes())
    return len(rec)


def read_splat(path: os.PathLike | str) -> dict[str, np.ndarray]:
    """Decode a .splat file: means (N,3), scales (N,3) linear, rgba (N,4) uint8,
    quats (N,4) wxyz float, alpha (N,) float, rgb (N,3) float."""
    raw = np.fromfile(str(path), dtype=SPLAT_DTYPE)
    quats = (raw["rot"].astype(np.float32) - 128.0) / 128.0
    quats /= np.maximum(np.linalg.norm(quats, axis=1, keepdims=True), 1e-12)
    return {
        "means": raw["pos"].astype(np.float32),
        "scales": raw["scale"].astype(np.float32),
        "rgba": raw["rgba"].copy(),
        "quats": quats,
        "alpha": raw["rgba"][:, 3].astype(np.float32) / 255.0,
        "rgb": raw["rgba"][:, :3].astype(np.float32) / 255.0,
    }


def run_export(
    ws: Workspace,
    ply: os.PathLike | str | None = None,
    min_opacity: float = 0.02,
    aligned: bool = False,
    fmt: str = "splat",
) -> Path:
    """Filter (+ optionally gravity-align) the trained PLY and write it for the web.

    aligned=True applies the similarity from work/scene.json (x' = s R x + t) so the
    viewer gets +Z up and metres. The sidecar <output>.json records which frame
    the file is in; the web build reads it.
    """
    src = Path(ply) if ply else ws.splat_ply
    if not src.exists():
        raise FileNotFoundError(f"no splat PLY at {src}; run `gsu train` (GPU) first or pass --ply")
    g = read_gaussian_ply(src)
    n_all = len(g)
    if min_opacity > 0:
        g = g.select(g.alpha >= min_opacity)
    log.info("loaded %d gaussians from %s, kept %d with alpha >= %.3f", n_all, src, len(g), min_opacity)

    frame = "colmap"
    up = None
    transform = None
    scene = read_json(ws.scene_json) if ws.scene_json.exists() else None
    if aligned:
        if scene is None:
            raise FileNotFoundError(f"--aligned needs {ws.scene_json}; run `gsu align` first")
        from ..geometry.align import SceneAlignment

        al = SceneAlignment.load(ws.scene_json)
        g = g.transformed(al.rotation, al.translation, al.scale)
        frame, up, transform = "aligned", [0.0, 0.0, 1.0], al.transform.tolist()
    elif scene is not None and scene.get("up_model") is not None:
        up = [float(v) for v in scene["up_model"]]

    if fmt == "ply":
        out = ws.splat / "model_export.ply"
        write_gaussian_ply(out, g)
        n = len(g)
    elif fmt == "splat":
        out = ws.splat_web
        n = write_splat(out, g)
    else:
        raise ValueError(f"unknown export format {fmt!r} (splat|ply)")
    sidecar = {
        "frame": frame, "n": int(n), "up": up, "source": str(src), "format": fmt,
        "min_opacity": float(min_opacity), "sh_degree": int(g.sh_degree) if fmt == "ply" else 0,
        "bytes_per_splat": BYTES_PER_SPLAT if fmt == "splat" else None,
        "transform_aligned_from_colmap": transform,
    }
    write_json(Path(str(out) + ".json"), sidecar)
    log.info("wrote %s (%d splats, %.1f MB) + sidecar", out, n, out.stat().st_size / 1e6)
    return out
