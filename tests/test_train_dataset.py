"""Dataset / initialisation tests that run without torch; torch-only parts skip."""
import math
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from gsplat_utils.train.dataset import load_colmap, resolve_sparse_dir
from gsplat_utils.train.trainer import TrainConfig, data_dirs, init_gaussians_numpy
from gsplat_utils.paths import Workspace

W, H = 64, 48
POSES = {
    # name: (qw, qx, qy, qz, tx, ty, tz)
    "frame_000000.jpg": (1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
    "frame_000001.jpg": (0.9238795, 0.0, 0.3826834, 0.0, 0.5, 0.0, 0.0),
    "frame_000002.jpg": (0.9659258, 0.2588190, 0.0, 0.0, 0.0, 0.25, 1.0),
}


def _quat_to_R(qw, qx, qy, qz):
    return np.array([
        [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
        [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
        [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)],
    ])


def write_text_model(sparse: Path, images: Path, model: str = "PINHOLE") -> None:
    sparse.mkdir(parents=True, exist_ok=True)
    images.mkdir(parents=True, exist_ok=True)
    params = "50 50 32 24" if model == "PINHOLE" else "50 32 24 0.01"
    (sparse / "cameras.txt").write_text(f"# cams\n1 {model} {W} {H} {params}\n")
    lines = ["# images"]
    for i, (name, p) in enumerate(POSES.items(), start=1):
        lines.append(f"{i} " + " ".join(str(v) for v in p) + f" 1 {name}")
        lines.append("")
        rgb = np.zeros((H, W, 3), np.uint8)
        rgb[..., 0] = np.linspace(0, 255, W, dtype=np.uint8)[None, :]
        rgb[..., 1] = 40 * i
        Image.fromarray(rgb).save(images / name, quality=95)
    (sparse / "images.txt").write_text("\n".join(lines) + "\n")
    pts = ["# points"]
    rng = np.random.default_rng(0)
    for j in range(40):
        x, y, z = rng.uniform(-1, 1), rng.uniform(-1, 1), rng.uniform(2, 4)
        pts.append(f"{j + 1} {x:.4f} {y:.4f} {z:.4f} {j * 6 % 256} 20 30 0.5")
    (sparse / "points3D.txt").write_text("\n".join(pts) + "\n")


def test_load_colmap_numpy(tmp_path):
    sparse, images = tmp_path / "sparse" / "0", tmp_path / "images"
    write_text_model(sparse, images)
    assert resolve_sparse_dir(tmp_path / "sparse") == sparse

    d = load_colmap(sparse, images, downscale=1)
    assert d.names == sorted(POSES)
    assert d.K.shape == (3, 3, 3) and d.w2c.shape == (3, 4, 4)
    np.testing.assert_allclose(d.K[0], [[50, 0, 32], [0, 50, 24], [0, 0, 1]])
    for i, name in enumerate(d.names):
        qw, qx, qy, qz, tx, ty, tz = POSES[name]
        R = _quat_to_R(qw, qx, qy, qz)
        np.testing.assert_allclose(d.w2c[i, :3, :3], R, atol=1e-6)
        np.testing.assert_allclose(d.w2c[i, :3, 3], [tx, ty, tz], atol=1e-6)
    assert np.allclose(d.w2c[:, 3], [0, 0, 0, 1])
    assert len(d.images) == 3 and d.images[0].shape == (H, W, 3) and d.images[0].dtype == np.uint8
    assert d.points.shape == (40, 3) and d.points_rgb.shape == (40, 3) and d.points_rgb.dtype == np.uint8
    assert d.scene_scale > 0 and d.camera_model == "PINHOLE"
    assert len(d.train_idx) == 3 and len(d.val_idx) == 0
    # camera centre of the identity pose is the origin
    np.testing.assert_allclose(d.positions[0], [0, 0, 0], atol=1e-9)


def test_load_colmap_downscale_and_split(tmp_path):
    sparse, images = tmp_path / "sparse", tmp_path / "images"
    write_text_model(sparse, images)
    d = load_colmap(sparse, images, downscale=2, eval_every=2)
    assert d.images[0].shape == (H // 2, W // 2, 3)
    np.testing.assert_allclose(d.K[0], [[25, 0, 16], [0, 25, 12], [0, 0, 1]])
    assert tuple(d.sizes[0]) == (W // 2, H // 2)
    assert list(d.val_idx) == [0, 2] and list(d.train_idx) == [1]
    d2 = load_colmap(sparse, images, load_images=False)
    assert d2.images == [] and d2.K.shape == (3, 3, 3)


def test_load_colmap_refuses_distorted(tmp_path):
    sparse, images = tmp_path / "sparse", tmp_path / "images"
    write_text_model(sparse, images, model="SIMPLE_RADIAL")
    with pytest.raises(ValueError, match="undistorted"):
        load_colmap(sparse, images)
    d = load_colmap(sparse, images, require_pinhole=False)
    assert d.camera_model == "SIMPLE_RADIAL"


def test_data_dirs(tmp_path):
    ws = Workspace(tmp_path / "work")
    ws.mkdirs()
    write_text_model(ws.model, ws.frames)
    assert data_dirs(ws, "sparse") == (ws.model, ws.frames)
    with pytest.raises(FileNotFoundError):
        data_dirs(ws, "undistorted")
    write_text_model(ws.undistorted / "sparse", ws.undistorted / "images")
    assert data_dirs(ws, "undistorted") == (ws.undistorted / "sparse", ws.undistorted / "images")
    with pytest.raises(ValueError):
        data_dirs(ws, "nope")


def test_init_gaussians_numpy():
    rng = np.random.default_rng(1)
    pts = rng.uniform(-1, 1, size=(500, 3))
    rgb = rng.integers(0, 256, size=(500, 3)).astype(np.uint8)
    a = init_gaussians_numpy(pts, rgb, sh_degree=3, init_opacity=0.1, init_scale=1.0, max_points=None, seed=0)
    assert a["means"].shape == (500, 3) and a["scales"].shape == (500, 3)
    assert a["quats"].shape == (500, 4) and a["opacities"].shape == (500,)
    assert a["sh0"].shape == (500, 1, 3) and a["shN"].shape == (500, 15, 3)
    np.testing.assert_allclose(np.linalg.norm(a["quats"], axis=1), 1.0, atol=1e-5)
    np.testing.assert_allclose(a["opacities"], math.log(0.1 / 0.9), atol=1e-5)
    np.testing.assert_allclose(0.5 + 0.28209479177387814 * a["sh0"][:, 0, :], rgb / 255.0, atol=1e-5)
    # scales are log mean 3-NN distance: positive distances, finite logs
    assert np.all(np.isfinite(a["scales"])) and np.all(np.exp(a["scales"]) < 1.0)
    b = init_gaussians_numpy(pts, rgb, sh_degree=1, init_opacity=0.5, init_scale=0.1, max_points=100, seed=0)
    assert b["means"].shape == (100, 3) and b["shN"].shape == (100, 3, 3)


def test_train_config_resolved():
    c = TrainConfig(iters=30000).resolved()
    assert (c.opacity_reg, c.scale_reg, c.init_opacity, c.init_scale, c.refine_stop_iter) == (0.0, 0.0, 0.1, 1.0, 15000)
    m = TrainConfig(iters=30000, strategy="mcmc", resume="x.pt").resolved()
    assert (m.opacity_reg, m.scale_reg, m.init_opacity, m.init_scale, m.refine_stop_iter) == (0.01, 0.01, 0.5, 0.1, 25000)
    assert isinstance(m.resume, Path) and m.to_json()["resume"] == "x.pt"
    assert TrainConfig(iters=30000).to_json()["resume"] is None


def test_torch_conversion_and_ssim(tmp_path):
    torch = pytest.importorskip("torch")
    from gsplat_utils.train.dataset import to_torch
    from gsplat_utils.train.losses import psnr, ssim

    sparse, images = tmp_path / "sparse", tmp_path / "images"
    write_text_model(sparse, images)
    d = load_colmap(sparse, images)
    t = to_torch(d, device="cpu")
    assert t["Ks"].shape == (3, 3, 3) and t["Ks"].dtype == torch.float32
    assert t["w2c"].shape == (3, 4, 4)
    assert t["images"][0].dtype == torch.uint8 and tuple(t["images"][0].shape) == (H, W, 3)

    img = torch.from_numpy(d.images[0]).permute(2, 0, 1)[None].float() / 255.0
    assert float(ssim(img, img)) == pytest.approx(1.0, abs=1e-5)
    noisy = (img + 0.1 * torch.rand_like(img)).clamp(0, 1)
    s = float(ssim(img, noisy))
    assert 0.0 < s < 1.0
    assert float(psnr(img, img)) > 100
    # numpy reference (same gaussian window, 'same' padding) within tolerance
    from scipy.ndimage import convolve

    x = np.arange(11) - 5.0
    g1 = np.exp(-x ** 2 / (2 * 1.5 ** 2)); g1 /= g1.sum()
    win = g1[:, None] * g1[None, :]
    a, b = img[0].numpy(), noisy[0].numpy()
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    vals = []
    for ch in range(3):
        f = lambda im: convolve(im, win, mode="constant", cval=0.0)
        mu1, mu2 = f(a[ch]), f(b[ch])
        s1 = f(a[ch] ** 2) - mu1 ** 2; s2 = f(b[ch] ** 2) - mu2 ** 2; s12 = f(a[ch] * b[ch]) - mu1 * mu2
        vals.append((((2 * mu1 * mu2 + c1) * (2 * s12 + c2)) / ((mu1 ** 2 + mu2 ** 2 + c1) * (s1 + s2 + c2))).mean())
    assert s == pytest.approx(float(np.mean(vals)), abs=1e-4)
