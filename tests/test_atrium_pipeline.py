"""Open-well (atrium) stair through the full geometry stage, no SfM needed:
two corridors on different levels joined by a square spiral stair."""
import json
import numpy as np

from gsplat_utils.common import write_json
from gsplat_utils.export.ply import write_point_ply
from gsplat_utils.geometry.align import SceneAlignment
from gsplat_utils.geometry.pipeline import GeometryConfig, run_geometry
from gsplat_utils.paths import Workspace
from gsplat_utils.sfm.model import SceneModel, write_cameras_json


def _spiral_path(step=0.08):
    pts = []
    # corridor along +x on level 0 (camera z 1.5), from x=-10 to x=-0.6 at y=-0.6
    for x in np.arange(-10, -0.6, step):
        pts.append([x, -0.6, 1.5])
    # square spiral around the void [0,3]x[0,3]: centreline 0.6 m outside it, rising 3.2 m over 1.5 turns
    corners = [[-0.6, -0.6], [3.6, -0.6], [3.6, 3.6], [-0.6, 3.6], [-0.6, -0.6], [3.6, -0.6], [3.6, 3.6]]
    legs = [np.array(corners[i + 1]) - np.array(corners[i]) for i in range(len(corners) - 1)]
    total = sum(np.linalg.norm(l) for l in legs)
    z = 1.5
    for i, leg in enumerate(legs):
        L = np.linalg.norm(leg)
        n = int(L / step)
        for k in range(n):
            t = k / n
            p = np.array(corners[i]) + leg * t
            pts.append([p[0], p[1], z])
            z += 3.2 * (L / n) / total
    # corridor along +x on level 1 from x=3.6 to 14 at y=3.6
    for x in np.arange(3.6, 14, step):
        pts.append([x, 3.6, 4.7])
    return np.array(pts)


def test_open_well_stair(tmp_path):
    rng = np.random.default_rng(1)
    C = _spiral_path()
    C += rng.normal(0, 0.02, C.shape)
    N = len(C)
    fwd = np.gradient(C, axis=0); fwd /= np.linalg.norm(fwd, axis=1, keepdims=True)
    M = np.tile(np.eye(4), (N, 1, 1))
    for i, f in enumerate(fwd):
        up = np.array([0, 0, 1.0]); right = np.cross(f, up); right /= np.linalg.norm(right); down = np.cross(f, right)
        M[i, :3, :3] = np.stack([right, down, f], 1); M[i, :3, 3] = C[i]
    # floor points plus balustrade/wall points 0.6 m either side of the whole walk
    pts = np.c_[rng.uniform(-10, 14, 3000), rng.uniform(-1.5, 4.5, 3000), np.zeros(3000)]
    pts = np.vstack([pts, np.c_[rng.uniform(3, 14, 1500), rng.uniform(2.6, 4.6, 1500), np.full(1500, 3.2)]])
    side = np.cross(fwd, np.array([0, 0, 1.0])); side /= np.linalg.norm(side, axis=1, keepdims=True)
    for sgn in (1, -1):
        for rep in range(3):
            w = C + sgn * 0.6 * side
            w = w.copy(); w[:, 2] += rng.uniform(-1.1, 0.5, N)   # 0.4..2.0 m above the local floor
            pts = np.vstack([pts, w])
    K = np.tile(np.array([[400.0, 0, 240], [0, 400.0, 424], [0, 0, 1]]), (N, 1, 1))
    names = [f"frame_{i:06d}.jpg" for i in range(N)]
    model = SceneModel(names, M, K, np.tile([480, 848], (N, 1)), pts, np.full((len(pts), 3), 128, np.uint8),
                       np.zeros(len(pts)), np.full(len(pts), 5, int), np.arange(N))
    model.frame_index = np.arange(N); model.time_s = np.arange(N) * 0.08
    ws = Workspace(tmp_path); ws.mkdirs()
    write_cameras_json(model, ws.cameras_json); write_point_ply(ws.points_ply, model.points, model.colors)
    write_json(ws.frames_meta, {"frames": [{"name": n, "index": i, "time_s": i * 0.08} for i, n in enumerate(names)]})
    al = SceneAlignment(np.eye(4), np.eye(3), np.zeros(3), 1.0, np.array([0, 0, 1.0]), 1.5, 2.8, "sfm", "test")
    write_json(ws.scene_json, al.to_json())
    d = run_geometry(ws, GeometryConfig(source="sfm", debug_png=False))
    stairs = [r for r in d["runs"] if r["kind"] == "stairs"]
    assert len(stairs) == 1 and len(d["levels"]) == 2
    at = stairs[0]["atrium"]
    assert at is not None, "open well not detected"
    c = np.array(at["corners"])
    lo, hi = c.min(axis=0), c.max(axis=0)
    assert np.allclose(lo, [0, 0], atol=0.35) and np.allclose(hi, [3, 3], atol=0.35), (lo, hi)
    assert abs(at["z1"] - at["z0"] - 3.2) < 0.4
    assert len(at["ring"]) == 4
    assert any("Open stair" in l["text"] for l in d["labels"])
    for f in ["iso.svg", "iso_exploded.svg", "plan_all.svg", "diagram.glb"]:
        assert (ws.geometry / f).exists()
