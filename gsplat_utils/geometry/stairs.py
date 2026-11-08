"""Atrium-style stairs: flights wrapping around an open void.

Detected from the plan shape of a stairs run: the walked path of an atrium
stair encloses an area (a square spiral), a straight flight does not."""
from __future__ import annotations

import numpy as np


def enclosed_area(xy: np.ndarray) -> float:
    """|shoelace area| of the polygon formed by a path (closed implicitly)."""
    if len(xy) < 3:
        return 0.0
    x, y = xy[:, 0], xy[:, 1]
    return 0.5 * abs(float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def loopiness(xy: np.ndarray) -> float:
    """Enclosed area relative to the area a square of the same perimeter would
    have (1 = perfect square loop, ~0 = straight line)."""
    if len(xy) < 3:
        return 0.0
    L = float(np.sum(np.linalg.norm(np.diff(xy, axis=0), axis=1)))
    if L <= 1e-6:
        return 0.0
    return enclosed_area(xy) / (L * L / 16.0)


def is_atrium(xy: np.ndarray, min_loopiness: float = 0.18, min_area_m2: float = 2.0) -> bool:
    return loopiness(xy) >= min_loopiness and enclosed_area(xy) >= min_area_m2


def atrium_void(xy: np.ndarray, theta: float, flight_half_width: float,
                min_side_m: float = 0.6) -> np.ndarray | None:
    """Rectangle (4 corners, world xy, counter-clockwise) of the open void: the
    bounding box of the stairs path in the Manhattan frame, shrunk by the flight
    half-width (the walked centreline runs mid-flight)."""
    if len(xy) < 3:
        return None
    c, s = np.cos(-theta), np.sin(-theta)
    Rz = np.array([[c, -s], [s, c]])
    q = xy @ Rz.T
    lo, hi = q.min(axis=0), q.max(axis=0)
    inset = float(flight_half_width)
    size = hi - lo
    if np.any(size - 2 * inset < min_side_m):
        inset = float(np.clip(np.min(size) * 0.3, 0.1, inset))
        if np.any(size - 2 * inset < min_side_m):
            return None
    lo2, hi2 = lo + inset, hi - inset
    rect = np.array([[lo2[0], lo2[1]], [hi2[0], lo2[1]], [hi2[0], hi2[1]], [lo2[0], hi2[1]]])
    return rect @ Rz   # rotate back (Rz is orthonormal: inverse = transpose, applied as rect @ Rz)


def gallery_ring(void: np.ndarray, width: float) -> list[np.ndarray]:
    """Four floor strips (quads, CCW) of width `width` around the void rectangle."""
    strips = []
    n = len(void)
    centre = void.mean(axis=0)
    for k in range(n):
        a, b = void[k], void[(k + 1) % n]
        e = b - a
        nrm = np.array([e[1], -e[0]])          # outward for a CCW rectangle
        nrm /= max(np.linalg.norm(nrm), 1e-9)
        if (a - centre) @ nrm < 0:
            nrm = -nrm
        # extend the strip to the corners of the outer ring so the four strips meet
        d = e / max(np.linalg.norm(e), 1e-9)
        a2, b2 = a - d * width, b + d * width
        strips.append(np.array([a2, b2, b2 + nrm * width, a2 + nrm * width]))
    return strips


def atrium_for_stairs(xy: np.ndarray, theta: float, flight_half_width: float, z0: float, z1: float,
                      ring: bool = True) -> dict | None:
    """Describe the open well of a stair whose walked path `xy` wraps around it."""
    void = atrium_void(xy, theta, flight_half_width)
    if void is None:
        return None
    width = 2.0 * float(flight_half_width)
    out = {
        "corners": np.round(void, 4).tolist(), "z0": float(z0), "z1": float(z1), "flight_width": width,
        "ring": [np.round(q, 4).tolist() for q in gallery_ring(void, width)] if ring else [],
        "inferred_ring": bool(ring),
    }
    return out
