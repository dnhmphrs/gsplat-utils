import json
import numpy as np
import pytest

from gsplat_utils.geometry.align import align_scene
from gsplat_utils.geometry.pipeline import GeometryConfig, run_geometry
from gsplat_utils.paths import Workspace
from gsplat_utils.sfm.model import write_cameras_json
from gsplat_utils.export.ply import write_point_ply
from gsplat_utils.common import write_json

from synthetic_scene import build_scene


@pytest.fixture(scope="module")
def scene():
    return build_scene()


def test_alignment_recovers_up_and_scale(scene):
    model, truth = scene
    al = align_scene(model, camera_height=truth["cam_h"])
    # up axis in the model frame should be R @ (0,0,1)
    up_true = truth["R"] @ np.array([0, 0, 1.0])
    ang = np.degrees(np.arccos(np.clip(al.up_model @ up_true, -1, 1)))
    assert ang < 2.0, ang
    # metres per model unit = 1/s
    assert abs(al.scale * truth["s"] - 1.0) < 0.05, al.scale * truth["s"]
    assert al.ceiling_height_m is not None and abs(al.ceiling_height_m - truth["ceil"]) < 0.3
    am = al.apply(model)
    # first camera at origin, camera height above floor
    assert np.allclose(am.positions[0][:2], 0, atol=0.05)
    assert abs(am.positions[0][2] - truth["cam_h"]) < 0.05


def test_geometry_pipeline(tmp_path, scene):
    model, truth = scene
    ws = Workspace(tmp_path)
    ws.mkdirs()
    write_cameras_json(model, ws.cameras_json)
    write_point_ply(ws.points_ply, model.points, model.colors)
    write_json(ws.frames_meta, {"frames": [{"name": n, "index": int(i), "time_s": float(t)}
                                            for n, i, t in zip(model.names, model.frame_index, model.time_s)]})
    al = align_scene(model, camera_height=truth["cam_h"])
    write_json(ws.scene_json, al.to_json())
    labels = tmp_path / "labels.json"
    labels.write_text(json.dumps([{"text": "Studio", "image": model.names[40]}, {"text": "Shop", "time_s": model.time_s[-20]}]))
    d = run_geometry(ws, GeometryConfig(source="sfm", labels=labels))
    levels = d["levels"]
    assert len(levels) == 2, levels
    assert abs(levels[1]["floor_z"] - levels[0]["floor_z"] - truth["rise"]) < 0.4
    kinds = [r["kind"] for r in d["runs"]]
    assert kinds.count("stairs") == 1, kinds
    flat = [r for r in d["runs"] if r["kind"] == "flat"]
    # corridor half widths close to 1 m on both sides
    halves = [(p["left"], p["right"]) for r in flat for p in r["pieces"] if p["length"] > 2.0]
    assert halves
    arr = np.array(halves)
    assert np.all(np.abs(arr - truth["half"]) < 0.35), arr
    # the L corridor on level 0 should be two main axis-aligned pieces
    l0 = [r for r in flat if r["level"] == 0][0]
    long_pieces = [p for p in l0["pieces"] if p["length"] > 3]
    assert len(long_pieces) == 2 and {p["axis"] for p in long_pieces} == {"x", "y"}
    # doorway opening detected on the left wall of the first leg between x=4 and 5.2
    ops = [o for p in l0["pieces"] for o in p["openings"]]
    assert any(o["side"] == "left" and 3.0 < o["s0"] < 5.5 for o in ops), ops
    # labels placed
    texts = {l["text"] for l in d["labels"]}
    assert {"Studio", "Shop", "Start", "End"} <= texts
    for f in ["diagram.json", "path.json", "plan_L0.svg", "plan_L1.svg", "plan_all.svg", "iso.svg",
              "iso_exploded.svg", "diagram.glb", "diagram.obj", "diagram.mtl", "occupancy_L0.png"]:
        assert (ws.geometry / f).exists(), f
    # GLB loads as a scene with geometry
    trimesh = pytest.importorskip("trimesh")
    sc = trimesh.load(ws.geometry / "diagram.glb")
    assert sum(len(g.faces) for g in sc.geometry.values()) > 100
