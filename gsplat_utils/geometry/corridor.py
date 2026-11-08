"""Corridor pieces (straight centreline + wall distances) -> ribbons, outlines, walls, openings."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .path import StraightSegment


@dataclass
class Opening:
    side: str          # "left" | "right"
    s0: float          # metres along the piece
    s1: float


@dataclass
class Piece:
    seg: StraightSegment
    left: float
    right: float
    openings: list[Opening] = field(default_factory=list)
    left_found: bool = True
    right_found: bool = True

    @property
    def p0(self) -> np.ndarray:
        return self.seg.p0

    @property
    def p1(self) -> np.ndarray:
        return self.seg.p1

    @property
    def length(self) -> float:
        return self.seg.length


@dataclass
class Wall:
    a: np.ndarray            # (2,) xy start
    b: np.ndarray            # (2,) xy end
    side: str
    piece: int
    openings: list[tuple[float, float]] = field(default_factory=list)   # fractions of the wall [t0, t1]
    z0: float = 0.0          # floor z at a
    z1: float = 0.0          # floor z at b (differs on stairs)

    @property
    def length(self) -> float:
        return float(np.linalg.norm(self.b - self.a))

    def solid_parts(self) -> list[tuple[float, float]]:
        """Fractions of the wall that are solid (complement of openings)."""
        parts = []
        t = 0.0
        for o0, o1 in sorted(self.openings):
            if o0 > t:
                parts.append((t, min(o0, 1.0)))
            t = max(t, o1)
        if t < 1.0:
            parts.append((t, 1.0))
        return [(a, b) for a, b in parts if b - a > 1e-6]


@dataclass
class Run:
    """One continuous stretch of path: a corridor on one level or a flight of stairs."""
    id: int
    kind: str                       # "flat" | "stairs"
    level: int                      # for flat
    level_from: int                 # for stairs
    level_to: int
    pieces: list[Piece]
    left_boundary: np.ndarray       # (K,3)
    right_boundary: np.ndarray      # (K,3)
    walls: list[Wall]
    atrium: dict | None = None      # open-well stair: {"corners", "z0", "z1", "ring", "flight_width"}
    hall: bool = False              # open area drawn as a rectangle rather than a corridor ribbon
    corners: np.ndarray | None = None   # (n,4,3) per piece: l0, l1, r1, r0

    @property
    def outline(self) -> np.ndarray:
        """Closed polygon (xy) of the ribbon."""
        return np.vstack([self.left_boundary[:, :2], self.right_boundary[::-1, :2]])

    def piece_quads(self) -> list[np.ndarray]:
        """Per piece: 4 corner points (3D) l0, l1, r1, r0."""
        if self.corners is not None:
            return [self.corners[j] for j in range(len(self.pieces))]
        quads = []
        for j in range(len(self.pieces)):
            l0, l1 = self.left_boundary[j], self.left_boundary[j + 1]
            r0, r1 = self.right_boundary[j], self.right_boundary[j + 1]
            quads.append(np.array([l0, l1, r1, r0]))
        return quads


def piece_widths(widths: dict[str, np.ndarray], seg: StraightSegment, default_half: float,
                 min_half: float = 0.4, max_half: float = 6.0) -> tuple[float, float, bool, bool]:
    """Robust half widths for a piece from the per-sample smoothed distances."""
    a, b = seg.i0, max(seg.i1, seg.i0 + 1)
    out = []
    found = []
    for key in ("left", "right"):
        v = widths[key][a:b]
        v = v[~np.isnan(v)]
        if len(v) == 0:
            out.append(default_half); found.append(False)
        else:
            out.append(float(np.clip(np.median(v), min_half, max_half))); found.append(True)
    return out[0], out[1], found[0], found[1]


def piece_openings(widths: dict[str, np.ndarray], seg: StraightSegment, step: float,
                   min_len_m: float = 0.7, jump_m: float = 0.9) -> list[Opening]:
    from .occupancy import find_openings

    a, b = seg.i0, max(seg.i1, seg.i0 + 1)
    n = b - a
    L = seg.length
    out = []
    for side in ("left", "right"):
        raw = widths[side + "_raw"][a:b]
        sm = widths[side][a:b]
        for k0, k1 in find_openings(raw, sm, step, min_len_m=min_len_m, jump_m=jump_m):
            s0 = L * k0 / max(n, 1)
            s1 = L * k1 / max(n, 1)
            if s1 - s0 >= min_len_m * 0.8:
                out.append(Opening(side, float(s0), float(s1)))
    return out


def drop_corner_openings(pieces: list[Piece], end_slack_m: float = 1.0) -> None:
    """An 'opening' that just looks into the next/previous corridor leg at a
    turn is not a doorway: remove openings touching a junction on the inside
    of the turn. Short openings touching the very start/end of the run (the
    landing of a flight of stairs, the edge of the reconstruction) go too."""
    if pieces:
        first, last = pieces[0], pieces[-1]
        first.openings = [o for o in first.openings if not (o.s0 < 0.15 and o.s1 - o.s0 < end_slack_m)]
        last.openings = [o for o in last.openings if not (o.s1 > last.length - 0.15 and o.s1 - o.s0 < end_slack_m)]
    for j in range(len(pieces) - 1):
        a, b = pieces[j], pieces[j + 1]
        cross = a.seg.direction[0] * b.seg.direction[1] - a.seg.direction[1] * b.seg.direction[0]
        if abs(cross) < 0.3:
            continue
        inside = "left" if cross > 0 else "right"
        reach_a = (a.left if inside == "left" else a.right) + (b.left if inside == "left" else b.right) + 0.5
        a.openings = [o for o in a.openings if not (o.side == inside and o.s1 > a.length - reach_a)]
        reach_b = reach_a
        b.openings = [o for o in b.openings if not (o.side == inside and o.s0 < reach_b)]


def _offset_sides(pieces: list[Piece], side: str, max_mitre_m: float = 2.0) -> tuple[np.ndarray, np.ndarray]:
    """Per piece (start, end) boundary points on one side, (n,3) each.

    Consecutive pieces share a mitred corner when the mitre is sane; otherwise
    (U-turns, very different widths) each piece keeps its own rectangle end and
    the ribbon simply overlaps there."""
    sign = 1.0 if side == "left" else -1.0
    n = len(pieces)
    starts = np.zeros((n, 3)); ends = np.zeros((n, 3))
    for j, pc in enumerate(pieces):
        w = pc.left if side == "left" else pc.right
        off = pc.seg.normal * sign * w
        starts[j] = np.r_[pc.p0[:2] + off, pc.p0[2]]
        ends[j] = np.r_[pc.p1[:2] + off, pc.p1[2]]
    for j in range(n - 1):
        pc, nx = pieces[j], pieces[j + 1]
        w = pc.left if side == "left" else pc.right
        wn = nx.left if side == "left" else nx.right
        d1, d2 = pc.seg.direction, nx.seg.direction
        if float(np.clip(d1 @ d2, -1, 1)) <= -0.85:
            continue                       # U-turn: keep separate ends
        P = _intersect2(starts[j, :2], d1, starts[j + 1, :2], d2)
        if P is None:
            continue                       # parallel: jog, keep separate ends
        if np.linalg.norm(P - ends[j, :2]) > min(3.0 * max(w, wn) + 0.5, max_mitre_m):
            continue                       # mitre would spike
        ends[j, :2] = P
        starts[j + 1, :2] = P
    return starts, ends


def _offset_boundary(pieces: list[Piece], side: str) -> np.ndarray:
    """Continuous boundary polyline (n+1, 3): piece starts plus the last end."""
    starts, ends = _offset_sides(pieces, side)
    return np.vstack([starts, ends[-1:]])


def _intersect2(p, d, q, e):
    den = d[0] * e[1] - d[1] * e[0]
    if abs(den) < 1e-6:
        return None
    t = ((q[0] - p[0]) * e[1] - (q[1] - p[1]) * e[0]) / den
    return p + t * d


def build_run(run_id: int, kind: str, pieces: list[Piece], level: int = -1, level_from: int = -1,
              level_to: int = -1, floor_z: float | None = None, camera_height: float = 1.5) -> Run:
    ls, le = _offset_sides(pieces, "left")
    rs, re_ = _offset_sides(pieces, "right")
    corners = np.stack([ls, le, re_, rs], axis=1)          # (n,4,3)
    left = np.vstack([ls, le[-1:]]); right = np.vstack([rs, re_[-1:]])
    # floor height: level floor for corridors, camera z - height for stairs
    if kind == "flat" and floor_z is not None:
        for arr in (left, right, corners):
            arr[..., 2] = floor_z
    else:
        for arr in (left, right, corners):
            arr[..., 2] -= camera_height
    walls: list[Wall] = []
    for j, pc in enumerate(pieces):
        for side, (sa, ea), found in (("left", (ls, le), pc.left_found), ("right", (rs, re_), pc.right_found)):
            a, b = corners[j, 0 if side == "left" else 3], corners[j, 1 if side == "left" else 2]
            L = float(np.linalg.norm(b[:2] - a[:2]))
            if L < 1e-6:
                continue
            ops = []
            if not found:
                ops.append((0.0, 1.0))      # no wall seen on this side at all
            for o in pc.openings:
                if o.side == side:
                    ops.append((max(0.0, o.s0 / max(pc.length, 1e-6)), min(1.0, o.s1 / max(pc.length, 1e-6))))
            walls.append(Wall(a[:2].copy(), b[:2].copy(), side, j, ops, float(a[2]), float(b[2])))
    return Run(run_id, kind, level, level_from, level_to, pieces, left, right, walls, corners=corners)
