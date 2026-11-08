"""Tests for `gsu poses` and `gsu web` (pose authoring + static site build)."""
from __future__ import annotations

import json
import math
import struct
from pathlib import Path

import numpy as np
import pytest

from gsplat_utils.paths import Workspace
from gsplat_utils.web.poses import (
    matrix_from_quat_xyzw, pose_from_camera, quat_xyzw_from_matrix, resolve_camera, run_poses,
)
from gsplat_utils.web.build import build_config, run_web


def rot_z(deg: float) -> np.ndarray:
    a = math.radians(deg)
    return np.array([[math.cos(a), -math.sin(a), 0], [math.sin(a), math.cos(a), 0], [0, 0, 1.0]])


def make_cameras(ws: Workspace, n: int = 5) -> list[dict]:
    cams = []
    for i in range(n):
        R = rot_z(10 * i)
        c2w = np.eye(4)
        c2w[:3, :3] = R
        c2w[:3, 3] = [i * 2.0, 0.5 * i, 1.4]
        cams.append({
            "name": f"frame_{i * 10:06d}.jpg", "image_id": i + 1, "frame_index": i * 10, "time_s": i * 0.5,
            "width": 848, "height": 480, "fx": 600.0, "fy": 600.0, "cx": 424.0, "cy": 240.0,
            "model": "OPENCV", "params": [], "camera_to_world": c2w.tolist(),
        })
    ws.root.mkdir(parents=True, exist_ok=True)
    ws.cameras_json.write_text(json.dumps({"frame": "colmap", "images": cams}))
    return cams


def write_splat(path: Path, n: int = 4) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        for i in range(n):
            f.write(struct.pack("<6f", i, 0, 0, 0.05, 0.05, 0.05))
            f.write(bytes([255, 0, 0, 255, 255, 128, 128, 128]))


@pytest.fixture
def ws(tmp_path: Path) -> Workspace:
    w = Workspace(tmp_path / "work")
    make_cameras(w)
    return w


def test_quaternion_matrix_roundtrip():
    rng = np.random.default_rng(0)
    for _ in range(20):
        q = rng.normal(size=4); q /= np.linalg.norm(q)
        R = matrix_from_quat_xyzw(q)
        assert np.allclose(R @ R.T, np.eye(3)) and np.isclose(np.linalg.det(R), 1)
        q2 = quat_xyzw_from_matrix(R)
        assert np.allclose(matrix_from_quat_xyzw(q2), R, atol=1e-9)


def test_resolve_camera_by_name_index_time(ws: Workspace):
    cams = json.loads(ws.cameras_json.read_text())["images"]
    assert resolve_camera(cams, "frame_000020.jpg")["frame_index"] == 20
    assert resolve_camera(cams, "frame_000020")["frame_index"] == 20
    assert resolve_camera(cams, "23")["frame_index"] == 20          # nearest frame index
    assert resolve_camera(cams, "1.4")["time_s"] == 1.5              # nearest time
    assert resolve_camera(cams, "0.74s")["time_s"] == 0.5
    with pytest.raises(SystemExit):
        resolve_camera(cams, "nope.jpg")


def test_pose_from_camera_fov_and_quaternion(ws: Workspace):
    cams = json.loads(ws.cameras_json.read_text())["images"]
    p = pose_from_camera(cams[2])
    assert p["position"] == pytest.approx([4.0, 1.0, 1.4])
    assert p["fov_y_deg"] == pytest.approx(math.degrees(2 * math.atan(240 / 600)), abs=1e-3)
    assert np.allclose(matrix_from_quat_xyzw(p["quaternion"]), rot_z(20), atol=1e-9)
    assert pose_from_camera(cams[2], fov=50)["fov_y_deg"] == 50


def test_run_poses_merge_order_labels_and_list(ws: Workspace, capsys):
    run_poses(ws, sets=[["studio", "frame_000010.jpg"], ["shop", "30"]], labels=[], fov=None, do_list=False)
    d = json.loads(ws.poses_json.read_text())
    assert d["frame"] == "colmap"
    assert [p["id"] for p in d["poses"]] == ["studio", "shop"]
    assert d["poses"][0]["label"] == "Studio" and d["poses"][0]["source_image"] == "frame_000010.jpg"
    assert d["poses"][1]["source_image"] == "frame_000030.jpg"
    assert d["transitions"]["default"]["mode"] == "spline"
    # overwrite shop by time, add about, relabel; order must stay stable
    run_poses(ws, sets=[["shop", "0.5"], ["about", "40"]], labels=[["shop", "The Shop"]], fov=55.0, do_list=True)
    d = json.loads(ws.poses_json.read_text())
    assert [p["id"] for p in d["poses"]] == ["studio", "shop", "about"]
    shop = d["poses"][1]
    assert shop["source_image"] == "frame_000010.jpg" and shop["label"] == "The Shop" and shop["fov_y_deg"] == 55.0
    out = capsys.readouterr().out
    assert "studio" in out and "about" in out
    # user extras on a pose survive a re-set
    d["poses"][0]["transition"] = {"mode": "arc"}
    ws.poses_json.write_text(json.dumps(d))
    run_poses(ws, sets=[["studio", "20"]], labels=[], fov=None, do_list=False)
    d = json.loads(ws.poses_json.read_text())
    assert d["poses"][0]["transition"] == {"mode": "arc"} and d["poses"][0]["source_image"] == "frame_000020.jpg"
    with pytest.raises(SystemExit):
        run_poses(ws, sets=[], labels=[["ghost", "x"]], fov=None, do_list=False)


def test_build_site_colmap_frame(ws: Workspace, tmp_path: Path):
    write_splat(ws.splat_web)
    run_poses(ws, sets=[["a", "0"], ["b", "40"]], labels=[], fov=None, do_list=False)
    out = run_web(ws, out=None, splat=None, title="Test site", serve=False, port=0)
    assert out == ws.web
    for name in ["index.html", "viewer.js", "transitions.js", "sort-worker.js", "splat-loader.js", "ui.js", "camera.js",
                 "shaders.js", "style.css", "model.splat", "viewer-config.json", "README.txt"]:
        assert (out / name).exists(), name
    cfg = json.loads((out / "viewer-config.json").read_text())
    assert cfg["title"] == "Test site" and cfg["model"] == "model.splat" and cfg["frame"] == "colmap"
    assert cfg["route"] is None and cfg["poses_auto"] is False
    assert [p["id"] for p in cfg["poses"]] == ["a", "b"]
    assert cfg["poses"][1]["position"] == pytest.approx([8.0, 2.0, 1.4])
    # up = -(mean image-down axis) of the cameras, normalised
    cams = json.loads(ws.cameras_json.read_text())["images"]
    down = np.mean([np.array(c["camera_to_world"])[:3, 1] for c in cams], axis=0)
    assert cfg["up"] == pytest.approx((-down / np.linalg.norm(down)).tolist(), abs=1e-9)
    assert abs(np.linalg.norm(cfg["up"]) - 1) < 1e-9
    assert cfg["colmap_to_model"]["scale"] == 1.0
    assert cfg["transitions"]["default"]["easing"] == "easeInOutCubic"


def test_build_site_aligned_frame_transforms_poses_and_keeps_route(ws: Workspace):
    write_splat(ws.splat_web)
    ws.splat_web.with_name("model.splat.json").write_text(json.dumps({"frame": "aligned", "n": 4, "up": [0, 0, 1]}))
    R = rot_z(90); t = np.array([1.0, 2.0, 3.0]); s = 2.0
    T = np.eye(4); T[:3, :3] = R * s; T[:3, 3] = t
    ws.scene_json.write_text(json.dumps({"transform": T.tolist(), "rotation": R.tolist(), "translation": t.tolist(),
                                         "scale": s, "up_model": [0, 0, 1], "camera_height_m": 1.5}))
    ws.geometry.mkdir(parents=True)
    (ws.geometry / "path.json").write_text(json.dumps({"frame": "aligned", "points": [[0, 0, 0], [1, 0, 0], [2, 0, 0]]}))
    run_poses(ws, sets=[["a", "0"], ["b", "20"]], labels=[], fov=None, do_list=False)
    cfg = build_config(ws, ws.splat_web, "model.splat", "t")
    assert cfg["frame"] == "aligned" and cfg["up"] == [0, 0, 1]
    pb = cfg["poses"][1]
    assert pb["position"] == pytest.approx((s * R @ np.array([4.0, 1.0, 1.4]) + t).tolist())
    assert np.allclose(matrix_from_quat_xyzw(pb["quaternion"]), R @ rot_z(20), atol=1e-9)
    assert cfg["route"] == [[0, 0, 0], [1, 0, 0], [2, 0, 0]]
    assert cfg["colmap_to_model"]["scale"] == 2.0
    # same scene but a COLMAP-frame model: route must be mapped back with the inverse similarity
    ws.splat_web.with_name("model.splat.json").write_text(json.dumps({"frame": "colmap", "n": 4, "up": [0, 0, 1]}))
    cfg2 = build_config(ws, ws.splat_web, "model.splat", "t")
    expect = ((np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0]]) - t) @ R) / s   # R^T (x - t) / s
    assert np.allclose(np.array(cfg2["route"]), expect, atol=1e-5)
    assert cfg2["poses"][1]["position"] == pytest.approx([4.0, 1.0, 1.4])


def test_build_site_fallbacks_points_ply_and_auto_poses(ws: Workspace):
    from gsplat_utils.export.ply import write_point_ply

    write_point_ply(ws.points_ply, np.random.default_rng(0).normal(size=(50, 3)), np.full((50, 3), 120, np.uint8))
    out = run_web(ws, out=ws.root / "site", splat=None, title="pts", serve=False, port=0)
    assert (out / "model.ply").exists() and not (out / "model.splat").exists()
    cfg = json.loads((out / "viewer-config.json").read_text())
    assert cfg["model"] == "model.ply" and cfg["poses_auto"] is True
    assert [p["id"] for p in cfg["poses"]] == ["start", "point-2", "point-3", "point-4", "end"]
    assert all(p.get("auto") for p in cfg["poses"])


def test_cli_web_and_poses(ws: Workspace):
    from gsplat_utils.cli import main

    write_splat(ws.splat_web)
    assert main(["poses", "--work", str(ws.root), "--set", "studio", "0.5", "--set", "shop", "40", "--label", "shop", "Shop"]) == 0
    assert main(["poses", "--work", str(ws.root), "--list"]) == 0
    assert main(["web", "--work", str(ws.root), "--title", "cli"]) == 0
    cfg = json.loads((ws.web / "viewer-config.json").read_text())
    assert [p["label"] for p in cfg["poses"]] == ["Studio", "Shop"]


def test_missing_model_is_a_clear_error(ws: Workspace):
    with pytest.raises(SystemExit):
        run_web(ws, out=None, splat=None, title="x", serve=False, port=0)
