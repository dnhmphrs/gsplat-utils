"""Scale-drift correction along a walk.

Structure-from-motion on forward motion through textureless corridors is only
weakly constrained in scale: the reconstruction can shrink or grow along the
sequence (we have seen 20-50x on a 30 m corridor). The camera's height above the
floor is physically constant though, so the reconstructed camera-to-floor
distance measured in a sliding time window gives the *local* scale factor.
`unroll` then re-integrates the camera path with those factors and moves every
point with the camera it belongs to."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import gaussian_filter1d, median_filter
from scipy.spatial import cKDTree

from ..common import log
from ..sfm.model import SceneModel


@dataclass
class ScaleProfile:
    t: np.ndarray            # window centres (s)
    scale: np.ndarray        # per window multiplicative correction (nan where unknown)
    n_points: np.ndarray
    per_camera: np.ndarray   # smoothed factor per camera (same order as the model)


def _smooth_track(C: np.ndarray, sigma: float = 3.0) -> np.ndarray:
    if len(C) < 7:
        return C.copy()
    return np.stack([gaussian_filter1d(C[:, i], sigma, mode="nearest") for i in range(3)], 1)


def _bands(dist: np.ndarray, bins: int = 48, lo: float = -3.0, hi: float = 2.0, rel: float = 0.15):
    """Density peaks of log10(distance) -> list of (distance, count)."""
    if len(dist) < 30:
        return []
    lb = np.log10(np.clip(dist, 10 ** lo, 10 ** hi))
    cnt, ed = np.histogram(lb, bins=bins, range=(lo, hi))
    k = np.array([1, 2, 3, 2, 1], float); k /= k.sum()
    hs = np.convolve(cnt, k, mode="same")
    out = []
    for i in range(1, bins - 1):
        if hs[i] > hs[i - 1] and hs[i] >= hs[i + 1] and hs[i] > rel * hs.max():
            out.append((float(10 ** (0.5 * (ed[i] + ed[i + 1]))), int(cnt[max(0, i - 1):i + 2].sum())))
    return out


def local_scale_profile(model: SceneModel, camera_height: float = 1.5, window_s: float = 4.0, step_s: float = 1.0,
                        min_points: int = 60, max_rate_per_s: float = 1.6, max_factor: float = 80.0,
                        max_jump: float = 4.0, speed_range: tuple[float, float] = (0.12, 2.8),
                        walking_speed: float = 1.0) -> ScaleProfile:
    """Per-camera scale correction so that the camera-to-floor distance is
    `camera_height` everywhere (model must be gravity aligned, +Z up, metric in
    the region the global alignment used).

    Per time window the density bands of point distances below the camera are
    candidates for the floor underfoot. Starting from the window that agrees best
    with the global alignment, the floor band is *tracked* forwards and backwards
    in time (never jumping more than `max_jump` x between windows, preferring the
    candidate nearest the previous one), and a candidate is rejected when the
    walking speed it implies is outside `speed_range`. The ceiling band gives a
    second vote on the *change* of scale between windows. Windows without a
    usable band fall back to the walking-speed prior when the raw motion is
    clearly above noise, else they are interpolated."""
    t = model.time_s
    C = model.positions
    P = model.points
    ok = ~np.isnan(t)
    Cs = _smooth_track(C)
    near = cKDTree(Cs[:, :2]).query(P[:, :2], k=1)[1]
    h = P[:, 2] - Cs[near, 2]
    tt = t[ok]
    centres = np.arange(np.nanmin(tt), np.nanmax(tt) + step_s, step_s)
    nW = len(centres)
    floor_cands: list[list[tuple[float, int]]] = []
    ceil_cands: list[list[tuple[float, int]]] = []
    v_raw = np.full(nW, np.nan)
    npts = np.zeros(nW, int)
    for k, tc in enumerate(centres):
        cams = np.nonzero(ok & (np.abs(t - tc) <= window_s / 2))[0]
        if len(cams) < 3:
            floor_cands.append([]); ceil_cands.append([])
            continue
        cams = cams[np.argsort(t[cams])]
        dt = float(t[cams[-1]] - t[cams[0]])
        if dt > 0.5:
            v_raw[k] = float(np.sum(np.linalg.norm(np.diff(Cs[cams, :2], axis=0), axis=1)) / dt)
        pm = np.isin(near, cams)
        hh = h[pm]
        below = -hh[hh < 0]
        npts[k] = len(below)
        floor_cands.append([(d, c) for d, c in _bands(below) if c >= min_points] if len(below) >= min_points else [])
        above = hh[hh > 0]
        ceil_cands.append([(d, c) for d, c in _bands(above) if c >= min_points] if len(above) >= min_points else [])

    def speed_ok(k: int, d: float) -> bool:
        if not np.isfinite(v_raw[k]):
            return True
        v = v_raw[k] * camera_height / d
        return speed_range[0] <= v <= speed_range[1] or v_raw[k] < 1e-3

    # anchor: window whose strongest band is closest to camera_height with most support
    best_k, best_score = None, None
    for k in range(nW):
        for d, c in floor_cands[k]:
            if abs(np.log(d / camera_height)) < np.log(1.3) and speed_ok(k, d):
                score = c
                if best_score is None or score > best_score:
                    best_k, best_score = k, score
    dist = np.full(nW, np.nan)
    if best_k is None:
        log.warning("drift correction: no window agrees with the global floor; leaving the scale as is")
        return ScaleProfile(centres, dist, npts, np.ones(len(C)))
    dist[best_k] = min(floor_cands[best_k], key=lambda dc: abs(np.log(dc[0] / camera_height)))[0]

    def track(order):
        prev_d = dist[best_k]
        prev_k = best_k
        prev_ceil = None
        for k in order:
            # prediction: previous distance, adjusted by how the ceiling band moved
            pred = prev_d
            cc = ceil_cands[k]
            if prev_ceil is not None and cc:
                cd = min(cc, key=lambda dc: abs(np.log(dc[0] / prev_ceil)))[0]
                if abs(np.log(cd / prev_ceil)) < np.log(max_jump):
                    pred = prev_d * cd / prev_ceil
            cands = [(d, c) for d, c in floor_cands[k]
                     if abs(np.log(d / prev_d)) <= np.log(max_jump) * max(1.0, (k - prev_k if order[0] > best_k else prev_k - k)) ** 0.5
                     and speed_ok(k, d)]
            if cands:
                d = min(cands, key=lambda dc: abs(np.log(dc[0] / pred)) - 0.1 * np.log1p(dc[1]) * 0.05)[0]
                dist[k] = d
                prev_d, prev_k = d, k
                if cc:
                    prev_ceil = min(cc, key=lambda dc: abs(np.log(dc[0] / (prev_ceil or dc[0]))))[0]
            elif cc and prev_ceil is not None:
                cd = min(cc, key=lambda dc: abs(np.log(dc[0] / prev_ceil)))[0]
                if abs(np.log(cd / prev_ceil)) < np.log(max_jump):
                    dist[k] = prev_d * cd / prev_ceil          # ceiling-only update
                    prev_d, prev_k, prev_ceil = dist[k], k, cd

    track(range(best_k + 1, nW))
    track(range(best_k - 1, -1, -1))

    ls = np.full(nW, np.nan)
    good = np.isfinite(dist)
    ls[good] = np.log(camera_height / dist[good])
    # fallback: walking-speed prior where the raw motion is clearly above noise
    for k in np.nonzero(~good)[0]:
        if np.isfinite(v_raw[k]) and v_raw[k] > 0.005:
            ls[k] = np.log(walking_speed / v_raw[k])
    good2 = np.isfinite(ls)
    if good2.sum() < 2:
        return ScaleProfile(centres, np.exp(ls), npts, np.ones(len(C)))
    vals = ls[good2]
    if len(vals) >= 3:
        vals = median_filter(vals, size=3, mode="nearest")
    ls[good2] = vals
    idx = np.arange(nW)
    ls[~good2] = np.interp(idx[~good2], idx[good2], ls[good2])
    lim = np.log(max_rate_per_s) * step_s
    for _ in range(2):
        for k in range(1, nW):
            ls[k] = np.clip(ls[k], ls[k - 1] - lim, ls[k - 1] + lim)
        for k in range(nW - 2, -1, -1):
            ls[k] = np.clip(ls[k], ls[k + 1] - lim, ls[k + 1] + lim)
    ls = gaussian_filter1d(ls, 1.0, mode="nearest")
    ls = np.clip(ls, -np.log(max_factor), np.log(max_factor))
    per_cam = np.ones(len(C))
    per_cam[ok] = np.exp(np.interp(t[ok], centres, ls))
    return ScaleProfile(centres, np.exp(ls), npts, per_cam)


def unroll(model: SceneModel, factors: np.ndarray, camera_height: float = 1.5, slope_threshold: float = 0.28,
           slope_window_s: float = 1.0, max_speed: float = 2.5, min_sloped_s: float = 3.0,
           min_rise_m: float = 0.6, stair_pitch: float = 0.6, trusted_factor: tuple[float, float] = (0.7, 1.4),
           max_integrate_factor: float = 6.0) -> SceneModel:
    """Re-integrate the camera path with per-camera scale factors and carry the
    points along with their nearest camera. Cameras must be in time order.

    Height is only integrated where the raw track is sloped (|dz/dxy| above
    `slope_threshold`, a scale-free test; buildings have stairs at pitch 0.5-0.7
    or near-flat ramps, little in between): corridors are flat and amplifying
    centimetre-level vertical drift by a large factor would otherwise turn them
    into ramps. Where the local factor is far from 1 (an untrustworthy stretch of
    the reconstruction) the rise of a flight is taken as `stair_pitch` x the
    corrected horizontal run instead of the amplified raw height change; where
    the factor exceeds `max_integrate_factor` the reconstruction had collapsed
    and its vertical component is noise, so the path is kept level there. Each
    step is also capped at `max_speed` x dt so a jump in the reconstruction
    across a frame gap cannot become a leap."""
    C = model.positions
    n = len(C)
    f = np.asarray(factors, float)
    t = model.time_s if model.time_s is not None else np.arange(n, dtype=float)
    t = np.where(np.isnan(t), np.nan, t)
    dts = np.diff(t)
    med_dt = float(np.nanmedian(dts[dts > 0])) if np.any(dts > 0) else 1.0
    Cs_old = _smooth_track(C)
    # scale-free slope of the raw track over +-slope_window_s
    w = max(1, int(round(slope_window_s / max(med_dt, 1e-3))))
    # raw position noise: residual of the raw track against its smoothed version
    resid = np.linalg.norm(C - Cs_old, axis=1)
    sloped = np.zeros(n, bool)
    for i in range(n):
        a, b = max(0, i - w), min(n - 1, i + w)
        dxy = np.linalg.norm(Cs_old[b, :2] - Cs_old[a, :2])
        dz = Cs_old[b, 2] - Cs_old[a, 2]
        noise = float(np.median(resid[a:b + 1])) if b > a else 0.0
        # a slope decision needs motion clearly above the noise of the reconstruction
        if dxy < 5.0 * noise or abs(dz) < 5.0 * noise:
            continue
        sloped[i] = abs(dz) > slope_threshold * max(dxy, 1e-9)
    # a flight of stairs is sloped for several seconds and climbs a good fraction
    # of a metre; shorter bursts are vertical noise of a weak reconstruction
    runs = []
    a = 0
    for i in range(1, n + 1):
        if i == n or sloped[i] != sloped[a]:
            runs.append((bool(sloped[a]), a, i)); a = i
    run_sign = np.zeros(n)
    for kind, a, b in runs:
        if not kind:
            continue
        dur = float(t[b - 1] - t[a]) if np.isfinite(t[a]) and np.isfinite(t[b - 1]) else (b - a) * med_dt
        rise = 0.0
        for i in range(max(1, a), b):
            rise += np.sqrt(f[i - 1] * f[i]) * (C[i, 2] - C[i - 1, 2])
        if dur < min_sloped_s or abs(rise) < min_rise_m:
            sloped[a:b] = False
        else:
            run_sign[a:b] = np.sign(Cs_old[b - 1, 2] - Cs_old[a, 2]) or 1.0
    newC = np.zeros_like(C)
    newC[0] = C[0]
    for i in range(1, n):
        s = np.sqrt(f[i - 1] * f[i])
        step = s * (C[i] - C[i - 1])
        on_slope = (sloped[i] or sloped[i - 1]) and s <= max_integrate_factor
        if not on_slope:
            step[2] = 0.0
        elif not (trusted_factor[0] <= s <= trusted_factor[1]):
            sgn = run_sign[i] if run_sign[i] != 0 else run_sign[i - 1]
            step[2] = sgn * stair_pitch * float(np.linalg.norm(step[:2]))
        dt = dts[i - 1] if np.isfinite(dts[i - 1]) and dts[i - 1] > 0 else med_dt
        L = np.linalg.norm(step)
        cap = max_speed * dt
        if L > cap:
            step *= cap / L
        newC[i] = newC[i - 1] + step
    Cs_new = _smooth_track(newC)
    near = cKDTree(Cs_old[:, :2]).query(model.points[:, :2], k=1)[1]
    newP = Cs_new[near] + (model.points - Cs_old[near]) * f[near][:, None]
    out = model.transformed(np.eye(3), np.zeros(3), 1.0)
    out.camera_to_world[:, :3, 3] = newC
    out.points = newP
    dz0 = newC[0, 2] - camera_height
    out.camera_to_world[:, :3, 3] -= np.array([0, 0, dz0])
    out.points = out.points - np.array([0, 0, dz0])
    return out


def correct_drift(model: SceneModel, camera_height: float = 1.5, **kw) -> tuple[SceneModel, ScaleProfile]:
    prof = local_scale_profile(model, camera_height, **kw)
    f = prof.per_camera
    spread = float(np.nanmax(f) / np.nanmin(f)) if np.all(np.isfinite(f)) and np.nanmin(f) > 0 else 1.0
    log.info("drift correction: local scale factors %.2f .. %.2f (x%.1f spread) over %d windows",
             float(np.nanmin(f)), float(np.nanmax(f)), spread, int(np.isfinite(prof.scale).sum()))
    return unroll(model, f, camera_height), prof
