"""COLMAP reconstruction -> arrays the trainer needs (numpy first, torch optional).

Conventions: `w2c` are world-to-camera 4x4 matrices in the OpenCV camera
convention (x right, y down, z forward), exactly what `gsplat.rasterization`
expects as `viewmats`. Intrinsics `K` are scaled to the loaded image size.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image

from ..common import log

PINHOLE_MODELS = {"PINHOLE", "SIMPLE_PINHOLE"}


@dataclass
class ColmapData:
    names: list[str]
    K: np.ndarray                 # (N,3,3) float64, for the loaded image size
    w2c: np.ndarray               # (N,4,4) float64, OpenCV convention (gsplat viewmats)
    sizes: np.ndarray             # (N,2) width,height of the loaded images
    images: list[np.ndarray]      # N x (H,W,3) uint8 (empty when load_images=False)
    points: np.ndarray            # (M,3) float64
    points_rgb: np.ndarray        # (M,3) uint8
    scene_scale: float            # max distance of a camera centre from the centroid
    camera_model: str
    train_idx: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=int))
    val_idx: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=int))

    @property
    def n_images(self) -> int:
        return len(self.names)

    @property
    def c2w(self) -> np.ndarray:
        return np.linalg.inv(self.w2c)

    @property
    def positions(self) -> np.ndarray:
        return self.c2w[:, :3, 3]


def resolve_sparse_dir(path: os.PathLike | str) -> Path:
    """Accept a model folder, or its parent holding a single `0/` model."""
    p = Path(path)
    for cand in (p, p / "0"):
        if (cand / "cameras.bin").exists() or (cand / "cameras.txt").exists():
            return cand
    raise FileNotFoundError(f"no COLMAP model (cameras.bin/.txt) under {p}")


def load_colmap(
    sparse_dir: os.PathLike | str,
    images_dir: os.PathLike | str,
    downscale: int = 1,
    eval_every: int = 0,
    require_pinhole: bool = True,
    load_images: bool = True,
    max_images: int | None = None,
) -> ColmapData:
    """Read cameras, poses, points and (optionally) the images.

    downscale: integer factor applied to the images; K is scaled to match.
    eval_every: every N-th image (sorted by name) goes to `val_idx`; 0 = none.
    require_pinhole: refuse distorted camera models (train on the undistorted set).
    """
    import pycolmap

    sparse_dir = resolve_sparse_dir(sparse_dir)
    images_dir = Path(images_dir)
    downscale = max(1, int(downscale))
    rec = pycolmap.Reconstruction(str(sparse_dir))
    reg = sorted(rec.reg_image_ids(), key=lambda i: rec.images[i].name)
    if max_images:
        reg = reg[:max_images]
    if not reg:
        raise RuntimeError(f"no registered images in {sparse_dir}")

    names, Ks, w2cs, sizes, images = [], [], [], [], []
    models = set()
    for iid in reg:
        im = rec.images[iid]
        cam = rec.cameras[im.camera_id]
        model_name = cam.model_name
        model_name = str(model_name() if callable(model_name) else model_name)
        models.add(model_name)
        cfw = im.cam_from_world
        cfw = cfw() if callable(cfw) else cfw
        m = np.eye(4)
        m[:3, :4] = np.asarray(cfw.matrix(), dtype=np.float64)
        K = np.asarray(cam.calibration_matrix(), dtype=np.float64).copy()
        cam_w, cam_h = int(cam.width), int(cam.height)
        tw, th = max(1, round(cam_w / downscale)), max(1, round(cam_h / downscale))
        if load_images:
            path = images_dir / im.name
            if not path.exists():
                raise FileNotFoundError(f"image {path} referenced by the COLMAP model is missing")
            with Image.open(path) as pil:
                pil = pil.convert("RGB")
                if (pil.width, pil.height) != (cam_w, cam_h):
                    log.warning("%s is %dx%d but the COLMAP camera is %dx%d; scaling intrinsics",
                                im.name, pil.width, pil.height, cam_w, cam_h)
                    tw, th = max(1, round(pil.width / downscale)), max(1, round(pil.height / downscale))
                    K[0, :] *= pil.width / cam_w
                    K[1, :] *= pil.height / cam_h
                    cam_w, cam_h = pil.width, pil.height
                if (tw, th) != (pil.width, pil.height):
                    pil = pil.resize((tw, th), Image.LANCZOS)
                images.append(np.asarray(pil, dtype=np.uint8))
        # scale K to the target size (exact ratio, so odd sizes stay consistent)
        K[0, :] *= tw / cam_w
        K[1, :] *= th / cam_h
        names.append(im.name)
        Ks.append(K)
        w2cs.append(m)
        sizes.append((tw, th))

    if require_pinhole and not models <= PINHOLE_MODELS:
        raise ValueError(
            f"camera model(s) {sorted(models)} are not pinhole. gsplat renders pinhole images only: "
            f"train on the undistorted set (`gsu sfm` writes work/colmap/undistorted; use --data undistorted) "
            f"or re-run `gsu sfm --camera-model PINHOLE`."
        )

    pts = np.array([p.xyz for p in rec.points3D.values()], dtype=np.float64).reshape(-1, 3)
    rgb = np.array([p.color for p in rec.points3D.values()], dtype=np.uint8).reshape(-1, 3)
    w2c = np.array(w2cs)
    c2w = np.linalg.inv(w2c)
    centres = c2w[:, :3, 3]
    scene_scale = float(np.max(np.linalg.norm(centres - centres.mean(axis=0), axis=1))) if len(centres) > 1 else 1.0
    scene_scale = max(scene_scale, 1e-6)

    idx = np.arange(len(names))
    if eval_every and eval_every > 0 and len(names) > eval_every:
        val_idx = idx[idx % eval_every == 0]
        train_idx = idx[idx % eval_every != 0]
    else:
        val_idx, train_idx = idx[:0], idx
    log.info("COLMAP data: %d images (%d train / %d val), %d points, model %s, downscale %d, scene scale %.3f",
             len(names), len(train_idx), len(val_idx), len(pts), "/".join(sorted(models)), downscale, scene_scale)
    return ColmapData(names, np.array(Ks), w2c, np.array(sizes), images, pts, rgb, scene_scale,
                      "/".join(sorted(models)), train_idx, val_idx)


def to_torch(data: ColmapData, device: str = "cuda", image_device: str = "cpu") -> dict:
    """Thin torch conversion: Ks/w2c as float32 on `device`, images as uint8 tensors
    on `image_device` (CPU by default; moved per step). Pinned when going to CUDA."""
    import torch

    pin = image_device == "cpu" and str(device).startswith("cuda") and torch.cuda.is_available()
    images = []
    for im in data.images:
        t = torch.from_numpy(np.ascontiguousarray(im))
        if image_device != "cpu":
            t = t.to(image_device)
        elif pin:
            t = t.pin_memory()
        images.append(t)
    return {
        "Ks": torch.from_numpy(data.K).float().to(device),
        "w2c": torch.from_numpy(data.w2c).float().to(device),
        "images": images,
        "sizes": data.sizes,
    }
