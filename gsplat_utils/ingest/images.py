"""Folder of photos -> normalised frame set (EXIF-rotated JPEGs, optional resize/selection)."""
from __future__ import annotations

import os
import re
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

from ..common import log, write_json
from .select import pick_per_window, sharpness_score

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp", ".heic"}


def _natural_key(p: Path):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", p.name)]


def list_images(src_dir: os.PathLike | str) -> list[Path]:
    src = Path(src_dir)
    files = [p for p in src.iterdir() if p.suffix.lower() in IMAGE_EXTS and p.is_file()]
    return sorted(files, key=_natural_key)


def import_image_dir(
    src_dir: os.PathLike | str,
    out_dir: os.PathLike | str,
    max_size: int | None = 1600,
    target_frames: int | None = None,
    sequential: bool = False,
    jpeg_quality: int = 95,
    meta_path: os.PathLike | str | None = None,
) -> dict:
    """Copy/convert images into `out_dir` as frame_XXXXXX.jpg and write frames_meta.json.

    sequential: set True when the photos were taken as an ordered walk (enables
    sequential matching in the SfM stage); False means unordered photos.
    """
    files = list_images(src_dir)
    if not files:
        raise RuntimeError(f"no images found in {src_dir}")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for old in out_dir.glob("frame_*.jpg"):
        old.unlink()

    try:
        from pillow_heif import register_heif_opener  # type: ignore

        register_heif_opener()
    except Exception:
        pass

    records = []
    scores = []
    for i, f in enumerate(files):
        with Image.open(f) as im:
            im = ImageOps.exif_transpose(im)
            im = im.convert("RGB")
            if max_size and max(im.size) > max_size:
                s = max_size / max(im.size)
                im = im.resize((round(im.width * s), round(im.height * s)), Image.LANCZOS)
            small = im.copy()
            small.thumbnail((480, 480))
            sc = sharpness_score(np.asarray(small.convert("L")))
            name = f"frame_{i:06d}.jpg"
            im.save(out_dir / name, quality=jpeg_quality, subsampling=0)
        records.append({"name": name, "index": i, "source": f.name,
                        "width": im.width, "height": im.height, "sharpness": round(sc, 2)})
        scores.append(sc)
    if target_frames and target_frames < len(records):
        keep = set(pick_per_window(np.array(scores), target_frames))
        for r in records:
            if r["index"] not in keep:
                (out_dir / r["name"]).unlink()
        records = [r for r in records if r["index"] in keep]
        log.info("kept %d/%d sharpest images", len(records), len(files))
    meta = {
        "source": {"type": "images", "path": str(src_dir), "n_images": len(files)},
        "sequential": bool(sequential),
        "selection": {"max_size": max_size, "target_frames": target_frames},
        "frames": records,
    }
    write_json(meta_path or (out_dir.parent / "frames_meta.json"), meta)
    return meta
