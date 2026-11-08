"""Workspace layout. Every stage reads/writes inside one work directory."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Workspace:
    root: Path

    def __post_init__(self):
        object.__setattr__(self, "root", Path(self.root))

    # --- ingest -----------------------------------------------------------
    @property
    def frames(self) -> Path:
        return self.root / "frames"

    @property
    def frames_meta(self) -> Path:
        return self.root / "frames_meta.json"

    # --- structure from motion -------------------------------------------
    @property
    def colmap(self) -> Path:
        return self.root / "colmap"

    @property
    def database(self) -> Path:
        return self.colmap / "database.db"

    @property
    def sparse(self) -> Path:
        return self.colmap / "sparse"

    @property
    def model(self) -> Path:
        """The reconstruction chosen by `gsu sfm` (largest model)."""
        return self.sparse / "0"

    @property
    def undistorted(self) -> Path:
        return self.colmap / "undistorted"

    @property
    def cameras_json(self) -> Path:
        return self.root / "cameras.json"

    @property
    def points_ply(self) -> Path:
        return self.root / "points_sfm.ply"

    # --- alignment --------------------------------------------------------
    @property
    def scene_json(self) -> Path:
        return self.root / "scene.json"

    # --- gaussian splat ---------------------------------------------------
    @property
    def splat(self) -> Path:
        return self.root / "splat"

    @property
    def splat_ply(self) -> Path:
        return self.splat / "model.ply"

    @property
    def splat_web(self) -> Path:
        return self.splat / "model.splat"

    # --- geometry / diagram ----------------------------------------------
    @property
    def geometry(self) -> Path:
        return self.root / "geometry"

    # --- web viewer -------------------------------------------------------
    @property
    def web(self) -> Path:
        return self.root / "web"

    @property
    def poses_json(self) -> Path:
        return self.root / "poses.json"

    def mkdirs(self) -> None:
        for p in (self.root, self.frames, self.colmap, self.splat, self.geometry):
            p.mkdir(parents=True, exist_ok=True)
