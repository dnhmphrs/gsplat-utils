"""Camera path processing: smoothing, arc-length resampling, level / stairs
segmentation and rectilinear (Manhattan) snapping."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.ndimage import gaussian_filter1d, median_filter


# ---------------------------------------------------------------------------
# smoothing + resampling
# ---------------------------------------------------------------------------
def smooth_positions(pos: np.ndarray, sigma: float = 2.0) -> np.ndarray:
    """Gaussian smoothing along the sequence index (sigma in samples)."""
    if len(pos) < 5 or sigma <= 0:
        return pos.copy()
    return np.stack([gaussian_filter1d(pos[:, i], sigma, mode="nearest") for i in range(pos.shape[1])], 1)


def track_inliers(pos: np.ndarray, k: float = 6.0, window: int = 7) -> np.ndarray:
    """Mask of cameras that sit on the track: a camera further than
    k * (median consecutive step) from the median-filtered track is a
    mis-registered frame and is dropped from path/alignment estimates."""
    n = len(pos)
    if n < window + 2:
        return np.ones(n, bool)
    med = np.stack([median_filter(pos[:, i], size=window, mode="nearest") for i in range(pos.shape[1])], 1)
    step = np.median(np.linalg.norm(np.diff(pos, axis=0), axis=1))
    if step <= 0:
        step = np.median(np.linalg.norm(pos - med, axis=1)) + 1e-9
    resid = np.linalg.norm(pos - med, axis=1)
    keep = resid <= k * step
    # never drop more than 20% - if we would, the track itself is erratic, keep all
    if keep.mean() < 0.8:
        return np.ones(n, bool)
    return keep


def arc_length(p: np.ndarray) -> np.ndarray:
    d = np.linalg.norm(np.diff(p[:, :2], axis=0), axis=1) if p.shape[1] >= 2 else np.abs(np.diff(p[:, 0]))
    return np.concatenate([[0.0], np.cumsum(d)])


def resample_by_arclength(p: np.ndarray, step: float, use_xy: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """Return (points (M,3), source_index (M,) float) sampled every `step`
    along the (horizontal) arc length."""
    seg = p[:, :2] if use_xy else p
    d = np.linalg.norm(np.diff(seg, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(d)])
    total = s[-1]
    if total < step:
        return p[[0, -1]].copy(), np.array([0.0, len(p) - 1.0])
    # drop duplicate arc positions so interp is well-defined
    keep = np.concatenate([[True], d > 1e-9])
    s_k, p_k, idx_k = s[keep], p[keep], np.arange(len(p))[keep]
    s_new = np.arange(0.0, total + 1e-9, step)
    out = np.stack([np.interp(s_new, s_k, p_k[:, i]) for i in range(p.shape[1])], 1)
    src = np.interp(s_new, s_k, idx_k.astype(float))
    return out, src


def tangents_xy(p: np.ndarray) -> np.ndarray:
    t = np.gradient(p[:, :2], axis=0)
    n = np.linalg.norm(t, axis=1, keepdims=True)
    return t / np.maximum(n, 1e-9)


# ---------------------------------------------------------------------------
# levels and stairs
# ---------------------------------------------------------------------------
@dataclass
class PathSegment:
    kind: str                # "flat" | "stairs"
    i0: int                  # inclusive sample index
    i1: int                  # exclusive
    level: int = -1          # for flat segments
    level_from: int = -1     # for stairs
    level_to: int = -1


@dataclass
class Level:
    id: int
    floor_z: float
    camera_z: float          # median camera z on this level
    n_samples: int = 0
    name: str = ""
    ceiling_z: float | None = None


def segment_levels(
    pts: np.ndarray,
    step: float,
    camera_height: float,
    stair_slope: float = 0.22,
    min_run_m: float = 1.5,
    level_merge_m: float = 1.5,
    slope_window_m: float = 1.5,
    max_landing_m: float = 3.5,
    ramp_slope: float = 0.12,
    ramp_window_m: float = 3.0,
) -> tuple[list[PathSegment], list[Level]]:
    """Split a resampled path into flat runs and stairs; group flat runs into levels.

    Flat runs that are not separated by stairs share a level even if the model
    drifts a little in z; levels whose camera heights differ by less than
    `level_merge_m` are merged (same physical floor visited twice: floors are
    >= 2.5 m apart, drift over a walk is usually less). A short flat run
    (<= max_landing_m) between two flights climbing in the same direction is a
    landing and becomes part of one stairs run. Besides the local slope test, a
    sustained rise (|dz/ds| over +-ramp_window_m above ramp_slope) also counts as
    stairs, so gentle ramps and spiral stairs whose rise is spread around corners
    are not missed; the ends of such runs are trimmed back to where the climb
    really starts."""
    n = len(pts)
    if n < 3:
        lv = Level(0, float(pts[:, 2].mean() - camera_height), float(pts[:, 2].mean()), n)
        return [PathSegment("flat", 0, n, level=0)], [lv]
    z = pts[:, 2]
    win = max(3, int(round(slope_window_m / step)) | 1)
    zs = gaussian_filter1d(z, win / 4.0, mode="nearest")
    slope = np.gradient(zs) / step
    slope = median_filter(slope, size=win, mode="nearest")
    stairs = np.abs(slope) > stair_slope
    w2 = max(2, int(round(ramp_window_m / step)))
    if n > 2 * w2 + 1:
        long_slope = np.zeros(n)
        long_slope[w2:-w2] = (zs[2 * w2:] - zs[:-2 * w2]) / (2 * w2 * step)
        long_slope[:w2] = long_slope[w2]
        long_slope[-w2:] = long_slope[-w2 - 1]
        stairs |= np.abs(long_slope) > ramp_slope

    # run-length encode and absorb runs shorter than min_run_m
    runs = _runs(stairs)
    # trim stairs runs back to where the climb really happens (the long window
    # also flags a few metres of flat path on either side of a steep flight)
    trimmed = []
    for kind, a, b in runs:
        if kind and b - a > 2:
            m = float(np.mean(np.abs(slope[a:b])))
            while a < b - 1 and abs(slope[a]) < 0.5 * m:
                a += 1
            while b > a + 1 and abs(slope[b - 1]) < 0.5 * m:
                b -= 1
        trimmed.append([kind, a, b])
    # re-close gaps left by trimming (flat fills them)
    runs = []
    pos = 0
    for kind, a, b in trimmed:
        if a > pos:
            if runs and not runs[-1][0]:
                runs[-1] = (False, runs[-1][1], a)
            else:
                runs.append((False, pos, a))
        if runs and runs[-1][0] == kind:
            runs[-1] = (kind, runs[-1][1], b)
        else:
            runs.append((kind, a, b))
        pos = b
    if pos < n:
        if runs and not runs[-1][0]:
            runs[-1] = (False, runs[-1][1], n)
        else:
            runs.append((False, pos, n))
    min_run = max(1, int(round(min_run_m / step)))
    changed = True
    while changed and len(runs) > 1:
        changed = False
        lens = [b - a for _, a, b in runs]
        j = int(np.argmin(lens))
        if lens[j] < min_run:
            kind, a, b = runs[j]
            # merge into the neighbour (prefer the longer one)
            if j == 0:
                runs[1] = (runs[1][0], a, runs[1][2]); del runs[0]
            elif j == len(runs) - 1:
                runs[j - 1] = (runs[j - 1][0], runs[j - 1][1], b); del runs[j]
            else:
                runs[j - 1] = (runs[j - 1][0], runs[j - 1][1], runs[j + 1][2]); del runs[j:j + 2]
            changed = True
    # stairs runs must actually climb: total |dz| > 0.6 m, else they are flat
    final = []
    for kind, a, b in runs:
        if kind and abs(zs[b - 1] - zs[a]) < 0.6:
            kind = False
        if final and final[-1][0] == kind:
            final[-1] = (kind, final[-1][1], b)
        else:
            final.append((kind, a, b))
    runs = final
    # landings: stairs, short flat, stairs (same direction) -> one stairs run
    max_landing = int(round(max_landing_m / step))
    merged = True
    while merged and len(runs) >= 3:
        merged = False
        for j in range(1, len(runs) - 1):
            k0, a0, b0 = runs[j - 1]; k1, a1, b1 = runs[j]; k2, a2, b2 = runs[j + 1]
            if k0 and k2 and not k1 and (b1 - a1) <= max_landing:
                d0 = np.sign(zs[b0 - 1] - zs[a0]); d2 = np.sign(zs[b2 - 1] - zs[a2])
                if d0 == d2:
                    runs[j - 1:j + 2] = [(True, a0, b2)]
                    merged = True
                    break

    segments: list[PathSegment] = []
    levels: list[Level] = []
    for kind, a, b in runs:
        if kind:
            segments.append(PathSegment("stairs", a, b))
        else:
            cz = float(np.median(z[a:b]))
            lid = None
            for lv in levels:
                if abs(lv.camera_z - cz) < level_merge_m:
                    lid = lv.id
                    # running weighted median approximation
                    lv.camera_z = (lv.camera_z * lv.n_samples + cz * (b - a)) / (lv.n_samples + (b - a))
                    lv.floor_z = lv.camera_z - camera_height
                    lv.n_samples += b - a
                    break
            if lid is None:
                lid = len(levels)
                levels.append(Level(lid, cz - camera_height, cz, b - a))
            segments.append(PathSegment("flat", a, b, level=lid))
    # order levels by height and renumber
    order = sorted(range(len(levels)), key=lambda k: levels[k].camera_z)
    remap = {old: new for new, old in enumerate(order)}
    levels = [levels[k] for k in order]
    for new, lv in enumerate(levels):
        lv.id = new
        lv.name = f"Level {new}"
    for sg in segments:
        if sg.kind == "flat":
            sg.level = remap[sg.level]
    for k, sg in enumerate(segments):
        if sg.kind == "stairs":
            prev = next((s.level for s in reversed(segments[:k]) if s.kind == "flat"), -1)
            nxt = next((s.level for s in segments[k + 1:] if s.kind == "flat"), -1)
            sg.level_from, sg.level_to = prev, nxt
    return segments, levels


def _runs(mask: np.ndarray) -> list[tuple[bool, int, int]]:
    runs = []
    start = 0
    for i in range(1, len(mask) + 1):
        if i == len(mask) or mask[i] != mask[start]:
            runs.append((bool(mask[start]), start, i))
            start = i
    return runs


# ---------------------------------------------------------------------------
# rectilinear snapping
# ---------------------------------------------------------------------------
def dominant_angle(tangents: np.ndarray, weights: np.ndarray | None = None) -> float:
    """Dominant direction modulo 90 degrees (radians in [0, pi/2))."""
    ang = np.arctan2(tangents[:, 1], tangents[:, 0])
    w = np.ones(len(ang)) if weights is None else weights
    # fold to 4-fold symmetry: angle*4 on the unit circle
    c = np.sum(w * np.cos(4 * ang)); s = np.sum(w * np.sin(4 * ang))
    return float(np.arctan2(s, c) / 4.0 % (np.pi / 2))


@dataclass
class StraightSegment:
    """A straight piece of centreline (after snapping) with sample indices."""
    p0: np.ndarray          # (3,) start (z = camera z)
    p1: np.ndarray
    i0: int                 # resampled path sample range this piece came from
    i1: int
    axis: str = ""          # "x" | "y" | "" (diagonal / unsnapped)
    kind: str = "flat"
    level: int = -1

    @property
    def length(self) -> float:
        return float(np.linalg.norm(self.p1[:2] - self.p0[:2]))

    @property
    def direction(self) -> np.ndarray:
        d = self.p1[:2] - self.p0[:2]
        n = np.linalg.norm(d)
        return d / n if n > 1e-9 else np.array([1.0, 0.0])

    @property
    def normal(self) -> np.ndarray:  # left-hand normal
        d = self.direction
        return np.array([-d[1], d[0]])


def snap_rectilinear(
    pts: np.ndarray,
    i0: int,
    i1: int,
    theta: float,
    min_run_m: float,
    step: float,
    diag_tol_deg: float = 30.0,
) -> list[StraightSegment]:
    """Turn pts[i0:i1] into axis-aligned straight pieces in the frame rotated by -theta.

    Samples whose tangent is within `diag_tol_deg` of an axis are snapped to it;
    other samples stay as (merged) diagonal pieces. Consecutive pieces are
    joined at line intersections so corners are exact right angles."""
    seg = pts[i0:i1]
    n = len(seg)
    if n < 2:
        return []
    c, s = np.cos(-theta), np.sin(-theta)
    Rz = np.array([[c, -s], [s, c]])
    q = seg[:, :2] @ Rz.T
    t = tangents_xy(np.c_[q, seg[:, 2]])
    ang = np.degrees(np.arctan2(np.abs(t[:, 1]), np.abs(t[:, 0])))  # 0 = x-ish, 90 = y-ish
    cls = np.where(ang < diag_tol_deg, 0, np.where(ang > 90 - diag_tol_deg, 1, 2))
    runs = []
    a = 0
    for i in range(1, n + 1):
        if i == n or cls[i] != cls[a]:
            runs.append([int(cls[a]), a, i]); a = i
    min_run = max(2, int(round(min_run_m / step)))
    # absorb short runs into the longer neighbour
    changed = True
    while changed and len(runs) > 1:
        changed = False
        lens = [b - a for _, a, b in runs]
        j = int(np.argmin(lens))
        if lens[j] < min_run:
            if j == 0:
                runs[1][1] = runs[0][1]; del runs[0]
            elif j == len(runs) - 1:
                runs[j - 1][2] = runs[j][2]; del runs[j]
            else:
                left, right = lens[j - 1], lens[j + 1]
                if runs[j - 1][0] == runs[j + 1][0]:
                    runs[j - 1][2] = runs[j + 1][2]; del runs[j:j + 2]
                elif left >= right:
                    runs[j - 1][2] = runs[j][2]; del runs[j]
                else:
                    runs[j + 1][1] = runs[j][1]; del runs[j]
            changed = True
    # lines per run
    pieces = []
    for k, (cl, a, b) in enumerate(runs):
        qq = q[a:b]
        zz = seg[a:b, 2]
        if cl == 0:    # x axis: y constant
            y = float(np.median(qq[:, 1]))
            x0, x1 = float(qq[0, 0]), float(qq[-1, 0])
            pieces.append({"axis": "x", "p0": np.array([x0, y]), "p1": np.array([x1, y]), "a": a, "b": b,
                           "z0": zz[0], "z1": zz[-1]})
        elif cl == 1:
            x = float(np.median(qq[:, 0]))
            y0, y1 = float(qq[0, 1]), float(qq[-1, 1])
            pieces.append({"axis": "y", "p0": np.array([x, y0]), "p1": np.array([x, y1]), "a": a, "b": b,
                           "z0": zz[0], "z1": zz[-1]})
        else:
            # diagonal: least-squares line through the samples
            m = qq.mean(axis=0)
            u, sv, vt = np.linalg.svd(qq - m)
            d = vt[0]
            if d @ (qq[-1] - qq[0]) < 0:
                d = -d
            pr = (qq - m) @ d
            pieces.append({"axis": "", "p0": m + d * pr.min(), "p1": m + d * pr.max(), "a": a, "b": b,
                           "z0": zz[0], "z1": zz[-1]})
    # join consecutive pieces: intersect their lines; parallel pieces get a jog
    out: list[dict] = []
    for k, pc in enumerate(pieces):
        if k > 0:
            prev = out[-1]
            P = _intersect(prev["p0"], prev["p1"], pc["p0"], pc["p1"])
            if P is None:
                # parallel: connect with a perpendicular jog (end of prev -> start of this)
                jog_p0 = prev["p1"].copy()
                jog_p1 = pc["p0"].copy()
                if np.linalg.norm(jog_p1 - jog_p0) > 1e-6:
                    out.append({"axis": "y" if prev["axis"] == "x" else ("x" if prev["axis"] == "y" else ""),
                                "p0": jog_p0, "p1": jog_p1, "a": pc["a"], "b": pc["a"] + 1,
                                "z0": prev["z1"], "z1": pc["z0"], "jog": True})
            else:
                # keep corners reasonable: do not let the intersection run away
                if np.linalg.norm(P - prev["p1"]) < 3 * min_run_m + 2.0 and np.linalg.norm(P - pc["p0"]) < 3 * min_run_m + 2.0:
                    prev["p1"] = P
                    pc["p0"] = P
                else:
                    out.append({"axis": "", "p0": prev["p1"].copy(), "p1": pc["p0"].copy(), "a": pc["a"],
                                "b": pc["a"] + 1, "z0": prev["z1"], "z1": pc["z0"], "jog": True})
        out.append(pc)
    # back to world frame
    Rinv = Rz.T
    result = []
    for pc in out:
        p0 = np.r_[pc["p0"] @ Rinv.T, pc["z0"]]
        p1 = np.r_[pc["p1"] @ Rinv.T, pc["z1"]]
        result.append(StraightSegment(p0, p1, i0 + pc["a"], i0 + pc["b"], pc["axis"]))
    return [r for r in result if r.length > 1e-6]


def _intersect(a0, a1, b0, b1):
    d1 = a1 - a0; d2 = b1 - b0
    den = d1[0] * d2[1] - d1[1] * d2[0]
    if abs(den) < 1e-9 * (np.linalg.norm(d1) * np.linalg.norm(d2) + 1e-12):
        return None
    t = ((b0[0] - a0[0]) * d2[1] - (b0[1] - a0[1]) * d2[0]) / den
    return a0 + t * d1


def polyline_segments(pts: np.ndarray, i0: int, i1: int, tol_m: float = 0.4) -> list[StraightSegment]:
    """Unsnapped alternative: Douglas-Peucker simplification of pts[i0:i1]."""
    idx = _douglas_peucker(pts[i0:i1, :2], tol_m)
    out = []
    for a, b in zip(idx[:-1], idx[1:]):
        out.append(StraightSegment(pts[i0 + a].copy(), pts[i0 + b].copy(), i0 + a, i0 + b + 1, ""))
    return out


def _douglas_peucker(p: np.ndarray, tol: float) -> list[int]:
    n = len(p)
    if n < 3:
        return list(range(n))
    keep = np.zeros(n, bool); keep[0] = keep[-1] = True
    stack = [(0, n - 1)]
    while stack:
        a, b = stack.pop()
        if b <= a + 1:
            continue
        d = p[b] - p[a]
        L = np.linalg.norm(d)
        if L < 1e-9:
            dist = np.linalg.norm(p[a + 1:b] - p[a], axis=1)
        else:
            dist = np.abs(np.cross(d / L, p[a + 1:b] - p[a]))
        j = int(np.argmax(dist))
        if dist[j] > tol:
            keep[a + 1 + j] = True
            stack += [(a, a + 1 + j), (a + 1 + j, b)]
    return list(np.nonzero(keep)[0])
