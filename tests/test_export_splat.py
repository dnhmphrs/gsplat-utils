import json
from pathlib import Path

import numpy as np
import pytest

from gsplat_utils.export.ply import SH_C0, Gaussians, read_gaussian_ply, write_gaussian_ply
from gsplat_utils.export.splat import BYTES_PER_SPLAT, read_splat, run_export, write_splat
from gsplat_utils.paths import Workspace


def _random_gaussians(n=200, sh_degree=1, seed=0) -> Gaussians:
    rng = np.random.default_rng(seed)
    K = (sh_degree + 1) ** 2 - 1
    q = rng.normal(size=(n, 4)).astype(np.float32)
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    return Gaussians(
        means=rng.uniform(-5, 5, size=(n, 3)).astype(np.float32),
        scales=rng.uniform(-4, -1, size=(n, 3)).astype(np.float32),
        quats=q,
        opacities=rng.uniform(-2, 6, size=n).astype(np.float32),
        sh0=rng.uniform(-1.5, 1.5, size=(n, 3)).astype(np.float32),
        shN=rng.normal(scale=0.1, size=(n, K, 3)).astype(np.float32),
    )


def _match(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Index into b for each row of a (nearest neighbour; export may reorder)."""
    d = np.linalg.norm(a[:, None, :] - b[None, :, :], axis=2)
    return d.argmin(axis=1)


def _workspace(tmp_path: Path, g: Gaussians) -> Workspace:
    ws = Workspace(tmp_path / "work")
    ws.mkdirs()
    write_gaussian_ply(ws.splat_ply, g)
    return ws


def test_splat_roundtrip(tmp_path):
    g = _random_gaussians()
    out = tmp_path / "a.splat"
    n = write_splat(out, g, sort=False)
    assert n == len(g)
    assert out.stat().st_size == len(g) * BYTES_PER_SPLAT == len(g) * 32
    r = read_splat(out)
    np.testing.assert_allclose(r["means"], g.means, atol=1e-6)
    np.testing.assert_allclose(r["scales"], np.exp(g.scales), rtol=1e-5)
    np.testing.assert_allclose(r["rgb"], np.clip(0.5 + SH_C0 * g.sh0, 0, 1), atol=1.0 / 255 + 1e-6)
    np.testing.assert_allclose(r["alpha"], 1 / (1 + np.exp(-g.opacities)), atol=1.0 / 255 + 1e-6)
    # quaternions: q and -q are the same rotation, compare |dot|
    dots = np.abs(np.sum(r["quats"] * g.quats, axis=1))
    assert dots.min() > 0.999


def test_sorted_export_is_big_first(tmp_path):
    g = _random_gaussians(50)
    out = tmp_path / "s.splat"
    write_splat(out, g, sort=True)
    r = read_splat(out)
    key = r["scales"].prod(axis=1) * r["alpha"]
    # non-increasing up to quantisation noise of alpha
    assert np.all(np.diff(key) <= key.max() * 0.02 + 1e-9)


def test_run_export_filters_and_sidecar(tmp_path):
    g = _random_gaussians(300)
    g.opacities[:40] = -10.0  # alpha ~ 4.5e-5 -> dropped
    ws = _workspace(tmp_path, g)
    out = run_export(ws, min_opacity=0.02)
    assert out == ws.splat_web and out.exists()
    r = read_splat(out)
    assert len(r["means"]) == 260
    side = json.loads(Path(str(out) + ".json").read_text())
    assert side["frame"] == "colmap" and side["n"] == 260 and side["up"] is None
    assert side["source"] == str(ws.splat_ply)
    idx = _match(g.means[40:], r["means"])
    np.testing.assert_allclose(r["means"][idx], g.means[40:], atol=1e-5)
    np.testing.assert_allclose(r["scales"][idx], np.exp(g.scales[40:]), rtol=1e-5)


def test_run_export_aligned(tmp_path):
    from scipy.spatial.transform import Rotation

    from gsplat_utils.geometry.align import SceneAlignment

    g = _random_gaussians(120, seed=3)
    ws = _workspace(tmp_path, g)
    R = Rotation.from_euler("xyz", [20, -35, 70], degrees=True).as_matrix()
    t = np.array([1.0, -2.0, 0.5])
    s = 2.5
    T = np.eye(4); T[:3, :3] = R * s; T[:3, 3] = t
    al = SceneAlignment(T, R, t, s, R.T @ np.array([0, 0, 1.0]), 1.5, 2.8, "sfm", "test")
    ws.scene_json.write_text(json.dumps(al.to_json(), default=lambda o: o.tolist()))

    out = run_export(ws, min_opacity=0.0, aligned=True)
    r = read_splat(out)
    expect = (g.means @ R.T) * s + t
    idx = _match(expect, r["means"])
    np.testing.assert_allclose(r["means"][idx], expect, atol=1e-4)
    np.testing.assert_allclose(r["scales"][idx], np.exp(g.scales) * s, rtol=1e-4)
    # rotated quaternion must equal R * q
    q_in = Rotation.from_quat(g.quats[:, [1, 2, 3, 0]])
    q_exp = (Rotation.from_matrix(R) * q_in).as_quat()[:, [3, 0, 1, 2]]
    dots = np.abs(np.sum(r["quats"][idx] * q_exp, axis=1))
    assert dots.min() > 0.995
    side = json.loads(Path(str(out) + ".json").read_text())
    assert side["frame"] == "aligned" and side["up"] == [0.0, 0.0, 1.0]
    assert np.allclose(np.array(side["transform_aligned_from_colmap"]), T)

    # unaligned export with a scene.json present reports the model-frame up axis
    out2 = run_export(ws, min_opacity=0.0, aligned=False)
    side2 = json.loads(Path(str(out2) + ".json").read_text())
    assert side2["frame"] == "colmap"
    np.testing.assert_allclose(side2["up"], R.T @ np.array([0, 0, 1.0]), atol=1e-9)


def test_run_export_ply_format(tmp_path):
    g = _random_gaussians(80, sh_degree=2)
    ws = _workspace(tmp_path, g)
    out = run_export(ws, min_opacity=0.0, fmt="ply")
    assert out.name == "model_export.ply"
    g2 = read_gaussian_ply(out)
    assert len(g2) == 80 and g2.sh_degree == 2
    np.testing.assert_allclose(g2.means, g.means)
    np.testing.assert_allclose(g2.shN, g.shN)


def test_run_export_missing_ply(tmp_path):
    ws = Workspace(tmp_path / "w")
    ws.mkdirs()
    with pytest.raises(FileNotFoundError):
        run_export(ws)
