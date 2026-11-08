"""Synthetic building: an L-shaped corridor on level 0, a straight stair flight,
and a straight corridor on level 1, with wall/floor/ceiling points and a camera
walk through it. Everything is then expressed in an arbitrary 'COLMAP-like'
frame (random rotation + scale) so the alignment stage has real work to do."""
from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation

from gsplat_utils.sfm.model import SceneModel

CORRIDOR_HALF = 1.0      # metres
CEIL = 2.8
CAM_H = 1.5
STAIR_RISE = 3.2


def _wall_points(rng, a, b, z0, z1, density, side_normal):
    """Random points on a vertical rectangle from xy a to b between z0 and z1."""
    a = np.asarray(a, float); b = np.asarray(b, float)
    L = np.linalg.norm(b - a)
    n = max(1, int(density * L * (z1 - z0)))
    t = rng.random(n); z = rng.uniform(z0, z1, n)
    xy = a + (b - a) * t[:, None]
    return np.c_[xy, z]


def _floor_points(rng, xs, ys, z, density):
    n = max(1, int(density * (xs[1] - xs[0]) * (ys[1] - ys[0])))
    return np.c_[rng.uniform(*xs, n), rng.uniform(*ys, n), np.full(n, z)]


def build_scene(density: float = 40.0, noise: float = 0.02, seed: int = 0, n_cams: int = 400):
    rng = np.random.default_rng(seed)
    pts = []
    # level 0: corridor along +x from x=0..12 (y in [-1,1]), then along +y from y=0..8 at x=12 (x in [11,13])
    w = CORRIDOR_HALF
    pts.append(_wall_points(rng, (0, w), (11 + 0, w), 0, CEIL, density, 1))          # left wall of leg 1 (y=+1) until the corner opening
    pts.append(_wall_points(rng, (0, -w), (13, -w), 0, CEIL, density, 1))           # right wall (y=-1) runs to the outer corner
    pts.append(_wall_points(rng, (13, -w), (13, 8), 0, CEIL, density, 1))           # outer wall of leg 2 (x=13)
    pts.append(_wall_points(rng, (11, w), (11, 8), 0, CEIL, density, 1))            # inner wall of leg 2 (x=11)
    pts.append(_wall_points(rng, (0, -w), (0, w), 0, CEIL, density, 1))             # end wall at x=0
    # a doorway gap in the left wall of leg 1 between x=4 and x=5.2 (no points there), plus a side room glimpse
    # (we remove points in the gap after generation)
    pts.append(_floor_points(rng, (0, 13), (-w, w), 0.0, density * 0.5))
    pts.append(_floor_points(rng, (11, 13), (w, 8), 0.0, density * 0.5))
    pts.append(_floor_points(rng, (0, 13), (-w, w), CEIL, density * 0.3))            # ceiling
    pts.append(_floor_points(rng, (11, 13), (w, 8), CEIL, density * 0.3))
    # stairs: from (12, 8) going +y to (12, 14), rising STAIR_RISE, walls x=11 and x=13
    for z0 in [0.0]:
        pts.append(_wall_points(rng, (11, 8), (11, 14), 0, CEIL + STAIR_RISE, density * 0.6, 1))
        pts.append(_wall_points(rng, (13, 8), (13, 14), 0, CEIL + STAIR_RISE, density * 0.6, 1))
    n = int(density * 6 * 2)
    ty = rng.uniform(8, 14, n)
    pts.append(np.c_[rng.uniform(11, 13, n), ty, (ty - 8) / 6 * STAIR_RISE])      # stair surface
    # level 1: corridor along -x from x=12 to x=0 at y in [13,15] (floor z = STAIR_RISE)
    z1 = STAIR_RISE
    pts.append(_wall_points(rng, (13, 15), (0, 15), z1, z1 + CEIL, density, 1))
    pts.append(_wall_points(rng, (11, 13), (0, 13), z1, z1 + CEIL, density, 1))
    pts.append(_wall_points(rng, (0, 13), (0, 15), z1, z1 + CEIL, density, 1))
    pts.append(_floor_points(rng, (0, 13), (13, 15), z1, density * 0.5))
    pts.append(_floor_points(rng, (0, 13), (13, 15), z1 + CEIL, density * 0.3))
    P = np.vstack(pts)
    # doorway: remove wall points in the gap x in [4, 5.2] on the y=+1 wall
    gap = (np.abs(P[:, 1] - w) < 1e-6) & (P[:, 0] > 4.0) & (P[:, 0] < 5.2)
    P = P[~gap]
    P += rng.normal(0, noise, P.shape)
    # a few outliers
    P = np.vstack([P, rng.uniform([-3, -3, -1], [16, 18, 8], (int(len(P) * 0.01), 3))])

    # camera walk: leg1 (0.5..12, y=0) -> leg2 (x=12, y=0..8) -> stairs (y 8..14, z rises) -> level1 (x 12..0.5, y=14)
    def seg(a, b, k):
        return np.linspace(a, b, k, endpoint=False)
    legs = [
        seg(np.array([0.5, 0.0, CAM_H]), np.array([12.0, 0.0, CAM_H]), int(n_cams * 0.3)),
        seg(np.array([12.0, 0.0, CAM_H]), np.array([12.0, 8.0, CAM_H]), int(n_cams * 0.2)),
        seg(np.array([12.0, 8.0, CAM_H]), np.array([12.0, 14.0, CAM_H + STAIR_RISE]), int(n_cams * 0.2)),
        seg(np.array([12.0, 14.0, CAM_H + STAIR_RISE]), np.array([0.5, 14.0, CAM_H + STAIR_RISE]), int(n_cams * 0.3)),
    ]
    C = np.vstack(legs)
    C += rng.normal(0, 0.05, C.shape)   # handheld wobble
    # camera orientation: looking along the walk direction, image y pointing down (-z world)
    fwd = np.gradient(C, axis=0); fwd /= np.linalg.norm(fwd, axis=1, keepdims=True)
    c2w = []
    for f in fwd:
        up = np.array([0, 0, 1.0])
        right = np.cross(f, up); right /= np.linalg.norm(right)
        down = np.cross(f, right)  # = -up for level walking
        R = np.stack([right, down, f], axis=1)   # columns: camera x (right), y (down), z (forward) in world
        c2w.append(R)
    c2w = np.array(c2w)
    # random similarity to a "COLMAP" frame
    Rw = Rotation.from_euler("xyz", [37, -112, 59], degrees=True).as_matrix()
    s = 0.37
    tw = np.array([1.3, -2.1, 0.4])

    def to_colmap(x):
        return (x @ Rw.T) * s + tw

    P_c = to_colmap(P)
    C_c = to_colmap(C)
    c2w_c = np.einsum("ij,njk->nik", Rw, c2w)
    N = len(C)
    M = np.tile(np.eye(4), (N, 1, 1)); M[:, :3, :3] = c2w_c; M[:, :3, 3] = C_c
    K = np.tile(np.array([[400.0, 0, 240], [0, 400.0, 424], [0, 0, 1]]), (N, 1, 1))
    names = [f"frame_{i:06d}.jpg" for i in range(N)]
    model = SceneModel(names, M, K, np.tile([480, 848], (N, 1)), P_c, np.full((len(P_c), 3), 128, np.uint8),
                       np.zeros(len(P_c)), np.full(len(P_c), 5, int), np.arange(N))
    model.frame_index = np.arange(N) * 10
    model.time_s = np.arange(N) * (37.0 / N)   # ~1 m/s walk
    truth = {"R": Rw, "s": s, "t": tw, "cam_h": CAM_H, "ceil": CEIL, "rise": STAIR_RISE, "half": CORRIDOR_HALF,
             "P_world": P, "C_world": C}
    return model, truth
