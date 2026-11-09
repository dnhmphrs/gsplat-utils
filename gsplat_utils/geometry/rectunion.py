"""Exact union of axis-aligned rectangles -> boundary loops (outer loops
counter-clockwise, holes clockwise). Used so coplanar surfaces that touch are
drawn as one shape without internal borders."""
from __future__ import annotations

import numpy as np


def is_axis_rect(quad: np.ndarray, tol: float = 1e-4) -> bool:
    q = np.asarray(quad, float)[:, :2]
    if len(q) != 4:
        return False
    xs, ys = np.unique(np.round(q[:, 0], 4)), np.unique(np.round(q[:, 1], 4))
    if len(xs) != 2 or len(ys) != 2:
        return False
    return abs((xs[1] - xs[0]) * (ys[1] - ys[0])) > tol


def rect_of(quad: np.ndarray) -> tuple[float, float, float, float]:
    q = np.asarray(quad, float)[:, :2]
    return float(q[:, 0].min()), float(q[:, 1].min()), float(q[:, 0].max()), float(q[:, 1].max())


def union_loops(rects: list[tuple[float, float, float, float]], snap: float = 1e-4) -> list[np.ndarray]:
    """Boundary loops of the union of rectangles (x0, y0, x1, y1)."""
    if not rects:
        return []
    r = np.array(rects, float)
    r = np.round(r / snap) * snap
    xs = np.unique(np.r_[r[:, 0], r[:, 2]]); ys = np.unique(np.r_[r[:, 1], r[:, 3]])
    nx, ny = len(xs) - 1, len(ys) - 1
    if nx < 1 or ny < 1:
        return []
    cx = 0.5 * (xs[:-1] + xs[1:]); cy = 0.5 * (ys[:-1] + ys[1:])
    covered = np.zeros((nx, ny), bool)
    for x0, y0, x1, y1 in r:
        ix = (cx > x0) & (cx < x1); iy = (cy > y0) & (cy < y1)
        covered[np.ix_(ix, iy)] = True
    # directed boundary edges with the interior on the left
    edges: dict[tuple[float, float], list[tuple[float, float]]] = {}

    def add(a, b):
        edges.setdefault(a, []).append(b)
    for i in range(nx):
        for j in range(ny):
            if not covered[i, j]:
                continue
            x0, x1, y0, y1 = xs[i], xs[i + 1], ys[j], ys[j + 1]
            if j == 0 or not covered[i, j - 1]:
                add((x0, y0), (x1, y0))        # bottom, interior above
            if i == nx - 1 or not covered[i + 1, j]:
                add((x1, y0), (x1, y1))        # right
            if j == ny - 1 or not covered[i, j + 1]:
                add((x1, y1), (x0, y1))        # top
            if i == 0 or not covered[i - 1, j]:
                add((x0, y1), (x0, y0))        # left
    loops = []
    while edges:
        start = next(iter(edges))
        loop = [start]
        cur = start
        prev_dir = None
        while True:
            nxts = edges.get(cur)
            if not nxts:
                break
            # at a vertex with several outgoing edges, prefer turning right-most relative to the
            # incoming direction? For rectilinear unions, choosing the edge that is NOT a
            # straight continuation of the opposite loop is handled by taking the first and
            # relying on each vertex having matched in/out degrees.
            nxt = nxts.pop(0)
            if not nxts:
                del edges[cur]
            loop.append(nxt)
            cur = nxt
            if cur == start:
                break
        pts = np.array(loop[:-1], float) if len(loop) > 1 and loop[-1] == loop[0] else np.array(loop, float)
        if len(pts) >= 4:
            # drop collinear vertices
            keep = []
            n = len(pts)
            for k in range(n):
                a, b, c = pts[k - 1], pts[k], pts[(k + 1) % n]
                if abs((b[0] - a[0]) * (c[1] - b[1]) - (b[1] - a[1]) * (c[0] - b[0])) > 1e-9:
                    keep.append(b)
            if len(keep) >= 4:
                loops.append(np.array(keep))
    return loops


def loop_is_ccw(loop: np.ndarray) -> bool:
    x, y = loop[:, 0], loop[:, 1]
    return float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))) > 0
