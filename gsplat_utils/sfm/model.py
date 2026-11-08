"""Numpy view of a COLMAP reconstruction + cameras.json export.

Conventions (used by every later stage):
  * camera_to_world is a 4x4 matrix with OpenCV camera axes
    (x right, y down, z forward); camera position = camera_to_world[:3, 3].
  * images are sorted by name, which for video frames is time order.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..common import log, read_json, write_json


@dataclass
class SceneModel:
    names: list[str]
    camera_to_world: np.ndarray            # (N,4,4)
    K: np.ndarray                          # (N,3,3)
    sizes: np.ndarray                      # (N,2) width,height
    points: np.ndarray                     # (M,3)
    colors: np.ndarray                     # (M,3) uint8
    point_errors: np.ndarray = field(default_factory=lambda: np.zeros(0))
    track_lengths: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=int))
    image_ids: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=int))
    camera_models: list[str] = field(default_factory=list)
    camera_params: list[list[float]] = field(default_factory=list)
    frame_index: np.ndarray | None = None  # source frame index (video) or sequence index
    time_s: np.ndarray | None = None
    source: str = "colmap"

    # -- derived -----------------------------------------------------------
    @property
    def n_images(self) -> int:
        return len(self.names)

    @property
    def positions(self) -> np.ndarray:
        return self.camera_to_world[:, :3, 3]

    @property
    def forward(self) -> np.ndarray:
        return self.camera_to_world[:, :3, 2]

    @property
    def image_down(self) -> np.ndarray:
        """World direction of the image's +y (down) axis."""
        return self.camera_to_world[:, :3, 1]

    def subset(self, mask: np.ndarray) -> "SceneModel":
        """Keep only the cameras selected by `mask` (points unchanged)."""
        idx = np.nonzero(np.asarray(mask))[0]
        m = SceneModel(
            [self.names[i] for i in idx], self.camera_to_world[idx], self.K[idx], self.sizes[idx],
            self.points, self.colors, self.point_errors, self.track_lengths,
            self.image_ids[idx] if len(self.image_ids) else self.image_ids,
            [self.camera_models[i] for i in idx] if self.camera_models else [],
            [self.camera_params[i] for i in idx] if self.camera_params else [],
            None if self.frame_index is None else self.frame_index[idx],
            None if self.time_s is None else self.time_s[idx], self.source,
        )
        return m

    def transformed(self, R: np.ndarray, t: np.ndarray, s: float = 1.0) -> "SceneModel":
        """Similarity x' = s R x + t applied to cameras and points."""
        c2w = self.camera_to_world.copy()
        c2w[:, :3, :3] = np.einsum("ij,njk->nik", R, c2w[:, :3, :3])
        c2w[:, :3, 3] = (c2w[:, :3, 3] @ R.T) * s + t
        pts = (self.points @ R.T) * s + t
        return SceneModel(
            list(self.names), c2w, self.K.copy(), self.sizes.copy(), pts, self.colors.copy(),
            self.point_errors.copy(), self.track_lengths.copy(), self.image_ids.copy(),
            list(self.camera_models), [list(p) for p in self.camera_params],
            None if self.frame_index is None else self.frame_index.copy(),
            None if self.time_s is None else self.time_s.copy(), self.source,
        )

    # -- loading -------------------------------------------------------------
    @classmethod
    def from_colmap(cls, model_dir: os.PathLike | str, frames_meta: os.PathLike | str | None = None) -> "SceneModel":
        import pycolmap

        rec = pycolmap.Reconstruction(str(model_dir))
        reg = sorted(rec.reg_image_ids(), key=lambda i: rec.images[i].name)
        names, c2w, Ks, sizes, ids, models, params = [], [], [], [], [], [], []
        for iid in reg:
            im = rec.images[iid]
            cam = rec.cameras[im.camera_id]
            cfw = im.cam_from_world
            cfw = cfw() if callable(cfw) else cfw
            m = np.eye(4)
            m[:3, :4] = np.asarray(cfw.matrix())
            c2w.append(np.linalg.inv(m))
            names.append(im.name)
            Ks.append(np.asarray(cam.calibration_matrix()))
            sizes.append((cam.width, cam.height))
            ids.append(iid)
            models.append(str(getattr(cam, "model_name", None) or getattr(cam.model, "name", cam.model)))
            params.append([float(x) for x in cam.params])
        pts, cols, errs, tl = [], [], [], []
        for pid, p in rec.points3D.items():
            pts.append(p.xyz)
            cols.append(p.color)
            errs.append(p.error)
            tl.append(p.track.length())
        model = cls(
            names, np.array(c2w), np.array(Ks), np.array(sizes), np.array(pts, dtype=np.float64).reshape(-1, 3),
            np.array(cols, dtype=np.uint8).reshape(-1, 3), np.array(errs), np.array(tl, dtype=int),
            np.array(ids, dtype=int), models, params,
        )
        if frames_meta is not None and Path(frames_meta).exists():
            model.attach_frames_meta(read_json(frames_meta))
        return model

    def attach_frames_meta(self, meta: dict) -> None:
        by_name = {f["name"]: f for f in meta.get("frames", [])}
        fi, ts = [], []
        for n in self.names:
            f = by_name.get(n, {})
            fi.append(f.get("index", -1))
            ts.append(f.get("time_s") if f.get("time_s") is not None else np.nan)
        self.frame_index = np.array(fi, dtype=int)
        self.time_s = np.array(ts, dtype=float)

    @classmethod
    def from_cameras_json(cls, path: os.PathLike | str, points: np.ndarray | None = None,
                          colors: np.ndarray | None = None) -> "SceneModel":
        d = read_json(path)
        ims = d["images"]
        names = [i["name"] for i in ims]
        c2w = np.array([i["camera_to_world"] for i in ims], dtype=np.float64)
        K = np.array([[[i["fx"], 0, i["cx"]], [0, i["fy"], i["cy"]], [0, 0, 1]] for i in ims])
        sizes = np.array([[i["width"], i["height"]] for i in ims])
        pts = np.zeros((0, 3)) if points is None else points
        cols = np.zeros((0, 3), np.uint8) if colors is None else colors
        m = cls(names, c2w, K, sizes, pts, cols, image_ids=np.array([i.get("image_id", -1) for i in ims]),
                camera_models=[i.get("model", "PINHOLE") for i in ims],
                camera_params=[i.get("params", []) for i in ims])
        m.frame_index = np.array([i.get("frame_index", -1) for i in ims], dtype=int)
        m.time_s = np.array([np.nan if i.get("time_s") is None else i["time_s"] for i in ims], dtype=float)
        m.source = d.get("frame", "colmap")
        return m


def write_cameras_json(model: SceneModel, path: os.PathLike | str, frame: str = "colmap") -> None:
    ims = []
    for i, n in enumerate(model.names):
        K = model.K[i]
        ims.append({
            "name": n,
            "image_id": int(model.image_ids[i]) if len(model.image_ids) else -1,
            "frame_index": int(model.frame_index[i]) if model.frame_index is not None else -1,
            "time_s": (None if model.time_s is None or np.isnan(model.time_s[i]) else float(model.time_s[i])),
            "width": int(model.sizes[i][0]), "height": int(model.sizes[i][1]),
            "fx": float(K[0, 0]), "fy": float(K[1, 1]), "cx": float(K[0, 2]), "cy": float(K[1, 2]),
            "model": model.camera_models[i] if model.camera_models else "PINHOLE",
            "params": model.camera_params[i] if model.camera_params else [],
            "camera_to_world": np.round(model.camera_to_world[i], 8).tolist(),
        })
    write_json(path, {
        "frame": frame,
        "convention": "camera_to_world 4x4, OpenCV camera axes (x right, y down, z forward)",
        "n_images": len(ims),
        "images": ims,
    })
    log.info("wrote %s (%d cameras)", path, len(ims))


def read_cameras_json(path) -> SceneModel:
    return SceneModel.from_cameras_json(path)
