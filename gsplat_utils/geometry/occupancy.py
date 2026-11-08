"""2D occupancy grid of a horizontal band of scene points + wall-distance ray casting."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import median_filter


@dataclass
class OccupancyGrid:
    """2D grid. `counts` is what the ray caster thresholds: the number of points
    per cell, or, when built with heights, the number of distinct height bins
    per cell ("wall-likeness": a wall has points at many heights, a handrail,
    a chair or a stray point only at one). `n_points` keeps the raw count."""
    origin: np.ndarray      # (2,) world xy of cell (0,0) corner
    cell: float
    counts: np.ndarray      # (ny, nx) int
    n_points: np.ndarray | None = None

    @classmethod
    def from_points(cls, xy: np.ndarray, cell: float, bounds: tuple[float, float, float, float] | None = None,
                    pad: float = 1.0, z: np.ndarray | None = None, z_range: tuple[float, float] | None = None,
                    z_bins: int = 4) -> "OccupancyGrid":
        if bounds is None:
            if len(xy) == 0:
                bounds = (0, 0, 1, 1)
            else:
                bounds = (xy[:, 0].min(), xy[:, 1].min(), xy[:, 0].max(), xy[:, 1].max())
        x0, y0, x1, y1 = bounds
        origin = np.array([x0 - pad, y0 - pad])
        nx = int(np.ceil((x1 - x0 + 2 * pad) / cell)) + 1
        ny = int(np.ceil((y1 - y0 + 2 * pad) / cell)) + 1
        if nx * ny > 40_000_000:
            raise MemoryError(
                f"occupancy grid would be {nx}x{ny} cells: the scene is {x1 - x0:.0f} x {y1 - y0:.0f} m at "
                f"{cell} m cells. The metric scale is probably wrong (check scene.json / --scale) or use --cell.")
        n_points = np.zeros((ny, nx), np.int32)
        if len(xy):
            ij = np.floor((xy - origin) / cell).astype(int)
            ok = (ij[:, 0] >= 0) & (ij[:, 0] < nx) & (ij[:, 1] >= 0) & (ij[:, 1] < ny)
            np.add.at(n_points, (ij[ok, 1], ij[ok, 0]), 1)
            if z is not None and z_bins > 1:
                z = np.asarray(z, float)
                zr = z_range or (float(z.min()), float(z.max()) + 1e-9)
                zb = np.clip(((z - zr[0]) / max(zr[1] - zr[0], 1e-9) * z_bins).astype(int), 0, z_bins - 1)
                cover = np.zeros((z_bins, ny, nx), bool)
                cover[zb[ok], ij[ok, 1], ij[ok, 0]] = True
                counts = cover.sum(axis=0).astype(np.int32)
                return cls(origin, cell, counts, n_points)
        return cls(origin, cell, n_points.copy(), n_points)

    @property
    def shape(self):
        return self.counts.shape

    def to_cell(self, xy: np.ndarray) -> np.ndarray:
        return np.floor((np.asarray(xy) - self.origin) / self.cell).astype(int)

    def to_world(self, ij: np.ndarray) -> np.ndarray:
        return self.origin + (np.asarray(ij, float) + 0.5) * self.cell

    def occupied(self, min_count: int = 1) -> np.ndarray:
        return self.counts >= min_count

    def inside(self, i: int, j: int) -> bool:
        return 0 <= j < self.counts.shape[0] and 0 <= i < self.counts.shape[1]

    def cast(self, start_xy: np.ndarray, direction: np.ndarray, max_dist: float, min_count: int = 1,
             min_dist: float = 0.25, support: int = 1) -> float:
        """Distance from start along `direction` to the first occupied cell.

        `support`: the hit must be confirmed by at least this many occupied cells
        among the 3x3 neighbourhood (reduces hits on isolated noise points).
        Returns nan if nothing is hit within max_dist."""
        d = direction / max(np.linalg.norm(direction), 1e-12)
        occ = self.counts
        h, w = occ.shape
        step = self.cell * 0.5
        n = int(max_dist / step) + 1
        for k in range(int(min_dist / step), n):
            p = start_xy + d * (k * step)
            i, j = int(np.floor((p[0] - self.origin[0]) / self.cell)), int(np.floor((p[1] - self.origin[1]) / self.cell))
            if not (0 <= j < h and 0 <= i < w):
                return float("nan")
            if occ[j, i] >= min_count:
                if support <= 1:
                    return k * step
                j0, j1 = max(0, j - 1), min(h, j + 2)
                i0, i1 = max(0, i - 1), min(w, i + 2)
                if int((occ[j0:j1, i0:i1] >= min_count).sum()) >= support:
                    return k * step
        return float("nan")


def corridor_half_widths(
    grid: OccupancyGrid,
    centre_xy: np.ndarray,
    tangents: np.ndarray,
    max_half_width: float,
    min_count: int = 1,
    support: int = 2,
    smooth_window_m: float = 2.0,
    step: float = 0.25,
) -> dict[str, np.ndarray]:
    """Cast rays left/right of every path sample. Returns raw and smoothed
    half-widths (nan where no wall was found within max_half_width)."""
    n = len(centre_xy)
    left = np.full(n, np.nan); right = np.full(n, np.nan)
    for k in range(n):
        t = tangents[k]
        nrm = np.array([-t[1], t[0]])
        left[k] = grid.cast(centre_xy[k], nrm, max_half_width, min_count, support=support)
        right[k] = grid.cast(centre_xy[k], -nrm, max_half_width, min_count, support=support)
    win = max(3, int(round(smooth_window_m / step)) | 1)
    return {
        "left_raw": left, "right_raw": right,
        "left": _robust_fill(left, win), "right": _robust_fill(right, win),
    }


def _robust_fill(v: np.ndarray, win: int) -> np.ndarray:
    """Median-filter ignoring nans, then interpolate remaining gaps."""
    out = np.full_like(v, np.nan)
    half = win // 2
    for k in range(len(v)):
        a, b = max(0, k - half), min(len(v), k + half + 1)
        vals = v[a:b]
        vals = vals[~np.isnan(vals)]
        if len(vals) >= max(2, (b - a) // 3):
            out[k] = np.median(vals)
    good = ~np.isnan(out)
    if good.sum() == 0:
        return out
    if good.sum() < len(out):
        idx = np.arange(len(out))
        out[~good] = np.interp(idx[~good], idx[good], out[good])
    return out


def find_openings(raw: np.ndarray, smooth: np.ndarray, step: float, min_len_m: float = 0.7,
                  jump_m: float = 0.9) -> list[tuple[int, int]]:
    """Index ranges where the raw wall distance is missing or clearly further
    than the smoothed wall: doorways, side corridors, rooms."""
    gap = np.isnan(raw) | (raw > smooth + jump_m)
    out = []
    a = None
    for k in range(len(gap) + 1):
        if k < len(gap) and gap[k]:
            a = k if a is None else a
        elif a is not None:
            if (k - a) * step >= min_len_m:
                out.append((a, k))
            a = None
    return out
