"""Frame scoring and selection helpers (pure numpy, no OpenCV needed)."""
from __future__ import annotations

import numpy as np


def sharpness_score(gray: np.ndarray) -> float:
    """Variance of the 3x3 Laplacian. Higher = sharper. `gray` is HxW uint8/float."""
    g = gray.astype(np.float32)
    if g.ndim == 3:
        g = g.mean(axis=2)
    lap = (
        -4.0 * g[1:-1, 1:-1]
        + g[:-2, 1:-1]
        + g[2:, 1:-1]
        + g[1:-1, :-2]
        + g[1:-1, 2:]
    )
    return float(lap.var())


def motion_score(prev_gray: np.ndarray | None, gray: np.ndarray) -> float:
    """Mean absolute difference to the previous frame (0 when unknown)."""
    if prev_gray is None:
        return 0.0
    return float(np.mean(np.abs(gray.astype(np.int16) - prev_gray.astype(np.int16))))


def pick_per_window(
    scores: np.ndarray,
    n_pick: int,
    start: int = 0,
    end: int | None = None,
    min_gap: int = 1,
) -> list[int]:
    """Split [start, end) into `n_pick` equal windows and take the best index in each.

    `min_gap` enforces a minimum index distance between consecutive picks (the
    previous pick wins; the current window shifts its search start).
    """
    scores = np.asarray(scores, dtype=np.float64)
    end = len(scores) if end is None else min(end, len(scores))
    n_avail = max(0, end - start)
    if n_avail == 0 or n_pick <= 0:
        return []
    n_pick = min(n_pick, n_avail)
    edges = np.linspace(start, end, n_pick + 1)
    picks: list[int] = []
    for i in range(n_pick):
        a = int(np.floor(edges[i]))
        b = int(np.floor(edges[i + 1])) if i < n_pick - 1 else end
        if picks:
            a = max(a, picks[-1] + min_gap)
        if b <= a:
            continue
        j = a + int(np.argmax(scores[a:b]))
        picks.append(j)
    return picks


def pick_by_motion(
    scores: np.ndarray,
    motion: np.ndarray,
    n_pick: int,
    min_gap: int = 1,
    still_weight: float = 0.5,
) -> list[int]:
    """Motion-adaptive selection: windows contain roughly equal amounts of
    image change (sum of inter-frame motion), so fast pans and turns get dense
    coverage while static stretches get sparse coverage. Within each window the
    sharpest frame is kept.

    still_weight adds a floor of `still_weight * median(motion)` per frame so
    that a static camera still yields frames at a reduced rate.
    """
    scores = np.asarray(scores, float)
    motion = np.asarray(motion, float)
    n = len(scores)
    if n == 0 or n_pick <= 0:
        return []
    if n_pick >= n:
        return list(range(n))
    w = motion + still_weight * max(float(np.median(motion)), 1e-6)
    w[0] = 0.0
    c = np.cumsum(w)
    total = c[-1]
    if total <= 0:
        return pick_per_window(scores, n_pick, min_gap=min_gap)
    edges = np.searchsorted(c, np.linspace(0, total, n_pick + 1)[1:-1], side="left")
    bounds = np.concatenate([[0], edges, [n]]).astype(int)
    picks: list[int] = []
    for a, b in zip(bounds[:-1], bounds[1:]):
        if picks:
            a = max(a, picks[-1] + min_gap)
        if b <= a:
            continue
        picks.append(a + int(np.argmax(scores[a:b])))
    return picks
