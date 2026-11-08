"""Run COLMAP (via pycolmap) on the frame folder.

features -> matching (sequential for video / ordered walks, exhaustive or
vocab-tree for photo sets) -> mapping (incremental or GLOMAP global) ->
largest model kept as colmap/sparse/0 -> cameras.json + points_sfm.ply ->
undistorted copy for Gaussian splat training.
"""
from __future__ import annotations

import os
import shutil
from pathlib import Path

import numpy as np

from ..common import Timer, log, read_json, write_json
from ..export.ply import write_point_ply
from ..paths import Workspace
from .model import SceneModel, write_cameras_json


PRESETS = {
    # Handheld video walking through rooms/corridors: low texture, forward
    # motion (small triangulation angles), blur. Dense features, denser pairs,
    # permissive registration.
    "video": dict(peak_threshold=0.001, dsp=True, affine=True, guided=True, max_ratio=0.8,
                  init_min_tri_angle=6.0, init_min_num_inliers=50, abs_pose_min_num_inliers=15,
                  abs_pose_min_inlier_ratio=0.15, min_num_matches=10, filter_min_tri_angle=1.0,
                  tri_min_angle=1.0, max_reg_trials=5, mapper="global", focal_factor=0.7),
    # Deliberate photo sets with good overlap and texture: COLMAP defaults.
    "photos": dict(peak_threshold=0.0067, dsp=False, affine=False, guided=False, max_ratio=0.8,
                   init_min_tri_angle=16.0, init_min_num_inliers=100, abs_pose_min_num_inliers=30,
                   abs_pose_min_inlier_ratio=0.25, min_num_matches=15, filter_min_tri_angle=1.5,
                   tri_min_angle=1.5, max_reg_trials=3, mapper="incremental", focal_factor=1.2),
}


def _camera_params_string(model: str, w: int, h: int, focal_px: float) -> str:
    cx, cy = w / 2.0, h / 2.0
    f = float(focal_px)
    layouts = {
        "SIMPLE_PINHOLE": [f, cx, cy],
        "PINHOLE": [f, f, cx, cy],
        "SIMPLE_RADIAL": [f, cx, cy, 0.0],
        "RADIAL": [f, cx, cy, 0.0, 0.0],
        "OPENCV": [f, f, cx, cy, 0.0, 0.0, 0.0, 0.0],
        "OPENCV_FISHEYE": [f, f, cx, cy, 0.0, 0.0, 0.0, 0.0],
        "FULL_OPENCV": [f, f, cx, cy] + [0.0] * 8,
        "SIMPLE_RADIAL_FISHEYE": [f, cx, cy, 0.0],
        "RADIAL_FISHEYE": [f, cx, cy, 0.0, 0.0],
    }
    if model not in layouts:
        raise ValueError(f"no parameter layout for camera model {model}")
    return ",".join(f"{v:.6f}" for v in layouts[model])


def _prior_from_model(model_dir: Path, camera_model: str) -> tuple[str, str] | None:
    """Reuse intrinsics refined by an earlier reconstruction as the prior of this run."""
    try:
        import pycolmap

        rec = pycolmap.Reconstruction(str(model_dir))
        cams = list(rec.cameras.values())
        if not cams:
            return None
        cam = cams[0]
        if cam.model_name == camera_model:
            return cam.model_name, ",".join(f"{float(v):.6f}" for v in cam.params)
        return camera_model, _camera_params_string(camera_model, cam.width, cam.height, float(cam.mean_focal_length()))
    except Exception as exc:  # pragma: no cover
        log.debug("no usable intrinsics prior: %s", exc)
        return None


def run_sfm(
    ws: Workspace,
    matcher: str = "auto",
    overlap: int = 15,
    loop_closure: bool = False,
    vocab_tree: os.PathLike | str | None = None,
    camera_model: str = "OPENCV",
    max_features: int = 8192,
    max_image_size: int = 1600,
    undistort: bool = True,
    use_gpu: bool = False,
    mapper: str = "auto",
    camera_mode: str = "auto",
    min_model_size: int = 10,
    preset: str = "auto",
    focal_px: float | None = None,
    intrinsics_prior: str = "auto",
    verbose: bool = False,
) -> SceneModel:
    """Run the whole SfM stage.

    preset: "video" | "photos" | "auto" (video preset for video sources).
    mapper: "global" (GLOMAP: fast, one model even for weak sequential chains),
        "incremental" (classic COLMAP), or "auto" (preset default: global for video).
    focal_px: prior focal length in pixels (phone video is usually ~0.8-0.9 x
        the image width; COLMAP's default guess of 1.2 x the long side is far
        off for wide phone lenses and makes initialisation fragile).
    intrinsics_prior: "auto" reuses the intrinsics of a previous run of this
        workspace (colmap/sparse/0) when it exists; "none" disables; or a path to
        a COLMAP model directory.
    """
    import pycolmap

    pycolmap.logging.minloglevel = 0 if verbose else 1  # silence COLMAP INFO spam unless -v

    meta = read_json(ws.frames_meta)
    frames = meta["frames"]
    n_images = len(frames)
    is_video = meta.get("source", {}).get("type") == "video"
    sequential = bool(meta.get("sequential", is_video))
    if matcher == "auto":
        if sequential:
            matcher = "sequential"
        elif n_images <= 500 or vocab_tree is None:
            matcher = "exhaustive"
        else:
            matcher = "vocabtree"
    if matcher == "vocabtree" and vocab_tree is None:
        raise SystemExit("--matcher vocabtree needs --vocab-tree <file> (download from colmap.github.io)")
    if preset == "auto":
        preset = "video" if (is_video or sequential) else "photos"
    P = PRESETS[preset]
    if mapper == "auto":
        mapper = P["mapper"]

    # intrinsics prior (before we wipe the previous run)
    prior: tuple[str, str] | None = None
    if focal_px is not None:
        from PIL import Image

        with Image.open(ws.frames / frames[0]["name"]) as im:
            w, h = im.size
        prior = (camera_model, _camera_params_string(camera_model, w, h, focal_px))
    elif intrinsics_prior == "auto" and ws.model.exists():
        prior = _prior_from_model(ws.model, camera_model)
    elif intrinsics_prior not in ("auto", "none", "", None):
        prior = _prior_from_model(Path(intrinsics_prior), camera_model)
    if prior:
        log.info("intrinsics prior: %s %s", prior[0], prior[1])

    ws.colmap.mkdir(parents=True, exist_ok=True)
    if ws.database.exists():
        log.info("removing old database %s", ws.database)
        ws.database.unlink()
    if ws.sparse.exists():
        shutil.rmtree(ws.sparse)
    ws.sparse.mkdir(parents=True)

    want_cuda = use_gpu and bool(getattr(pycolmap, "has_cuda", False))
    device = pycolmap.Device.cuda if want_cuda else pycolmap.Device.cpu
    log.info("%d images, preset=%s, matcher=%s, mapper=%s, device=%s, camera=%s", n_images, preset, matcher,
             mapper, "cuda" if want_cuda else "cpu", camera_model)

    with Timer("feature extraction"):
        ext = pycolmap.FeatureExtractionOptions()
        ext.max_image_size = int(max_image_size)
        ext.use_gpu = want_cuda
        ext.sift.max_num_features = int(max_features)
        ext.sift.peak_threshold = float(P["peak_threshold"])
        ext.sift.domain_size_pooling = bool(P["dsp"])
        ext.sift.estimate_affine_shape = bool(P["affine"])
        reader = pycolmap.ImageReaderOptions()
        reader.camera_model = camera_model
        reader.default_focal_length_factor = float(P["focal_factor"])
        if prior:
            reader.camera_model = prior[0]
            reader.camera_params = prior[1]
        if camera_mode == "auto":
            cmode = pycolmap.CameraMode.SINGLE if is_video else pycolmap.CameraMode.AUTO
        else:
            cmode = getattr(pycolmap.CameraMode, camera_mode.upper())
        pycolmap.extract_features(
            ws.database, ws.frames, camera_mode=cmode, reader_options=reader,
            extraction_options=ext, device=device,
        )

    with Timer(f"feature matching ({matcher})"):
        mo = pycolmap.FeatureMatchingOptions()
        mo.use_gpu = want_cuda
        mo.guided_matching = bool(P["guided"])
        mo.sift.max_ratio = float(P["max_ratio"])
        if matcher == "sequential":
            # pass 1: every neighbour within `overlap` (dense chain for registration)
            po = pycolmap.SequentialPairingOptions()
            po.overlap = int(overlap)
            po.quadratic_overlap = False
            if loop_closure:
                if vocab_tree is None:
                    log.warning("--loop-closure needs --vocab-tree; continuing without loop detection")
                else:
                    po.loop_detection = True
                    po.vocab_tree_path = str(vocab_tree)
            pycolmap.match_sequential(ws.database, matching_options=mo, pairing_options=po, device=device)
            # pass 2: i+2^k pairs for long-range links (already matched pairs are skipped)
            po2 = pycolmap.SequentialPairingOptions()
            po2.overlap = int(max(1, np.ceil(np.log2(max(2, n_images)))))
            po2.quadratic_overlap = True
            pycolmap.match_sequential(ws.database, matching_options=mo, pairing_options=po2, device=device)
        elif matcher == "exhaustive":
            pycolmap.match_exhaustive(ws.database, matching_options=mo, device=device)
        else:
            po = pycolmap.VocabTreePairingOptions()
            po.vocab_tree_path = str(vocab_tree)
            pycolmap.match_vocabtree(ws.database, matching_options=mo, pairing_options=po, device=device)

    with Timer(f"mapping ({mapper})"):
        if mapper == "global":
            opts = pycolmap.GlobalPipelineOptions()
            opts.min_model_size = int(min_model_size)
            opts.min_num_matches = int(P["min_num_matches"])
            recs = pycolmap.global_mapping(ws.database, ws.frames, ws.sparse, options=opts)
        else:
            opts = pycolmap.IncrementalPipelineOptions()
            opts.min_model_size = int(min_model_size)
            opts.min_num_matches = int(P["min_num_matches"])
            opts.mapper.init_min_tri_angle = float(P["init_min_tri_angle"])
            opts.mapper.init_min_num_inliers = int(P["init_min_num_inliers"])
            opts.mapper.abs_pose_min_num_inliers = int(P["abs_pose_min_num_inliers"])
            opts.mapper.abs_pose_min_inlier_ratio = float(P["abs_pose_min_inlier_ratio"])
            opts.mapper.filter_min_tri_angle = float(P["filter_min_tri_angle"])
            opts.mapper.max_reg_trials = int(P["max_reg_trials"])
            opts.triangulation.min_angle = float(P["tri_min_angle"])
            recs = pycolmap.incremental_mapping(ws.database, ws.frames, ws.sparse, options=opts)

    if not recs:
        raise RuntimeError(
            "COLMAP could not reconstruct anything. Try more frames (--frames), a smaller "
            "--max-size, --matcher exhaustive, or check the footage for motion blur."
        )

    # Keep every model under sparse/models/<k>, promote the largest to sparse/0.
    models_dir = ws.sparse / "models"
    models_dir.mkdir(exist_ok=True)
    for k in list(recs):
        src = ws.sparse / str(k)
        if src.exists():
            shutil.move(str(src), str(models_dir / str(k)))
    summary = []
    for k, rec in sorted(recs.items()):
        summary.append({
            "model": int(k),
            "images": int(rec.num_reg_images()),
            "points": int(rec.num_points3D()),
            "mean_reproj_error_px": float(rec.compute_mean_reprojection_error()),
            "mean_track_length": float(rec.compute_mean_track_length()),
            "path": str(models_dir / str(k)),
        })
    best = max(summary, key=lambda s: s["images"])
    rec = recs[best["model"]]
    ws.model.mkdir(parents=True, exist_ok=True)
    rec.write(ws.model)
    rec.write_text(ws.model)
    for s in summary:
        log.info("model %d: %d images, %d points, reproj %.2fpx%s", s["model"], s["images"], s["points"],
                 s["mean_reproj_error_px"], "  <- selected" if s is best else "")
    frac = best["images"] / max(1, n_images)
    if len(summary) > 1 or frac < 0.8:
        log.warning(
            "registered %d/%d images (%.0f%%) in the largest model. Sub-models are kept in %s; "
            "a break in the walk (blur, fast turn) splits the reconstruction.",
            best["images"], n_images, 100 * frac, models_dir,
        )

    model = SceneModel.from_colmap(ws.model, ws.frames_meta)
    write_cameras_json(model, ws.cameras_json)
    write_point_ply(ws.points_ply, model.points, model.colors)
    log.info("wrote %s (%d points)", ws.points_ply, len(model.points))

    if undistort:
        with Timer("undistorting images for training"):
            if ws.undistorted.exists():
                shutil.rmtree(ws.undistorted)
            pycolmap.undistort_images(
                output_path=ws.undistorted, input_path=ws.model, image_path=ws.frames, output_type="COLMAP",
            )

    write_json(ws.colmap / "sfm_summary.json", {
        "n_input_images": n_images, "matcher": matcher, "mapper": mapper, "camera_model": camera_model,
        "preset": preset, "intrinsics_prior": prior,
        "selected_model": best["model"], "models": summary,
        "registered_fraction": frac,
    })
    return model


def promote_model(ws: Workspace, model_dir: os.PathLike | str, undistort: bool = True) -> SceneModel:
    """Make `model_dir` the workspace's reconstruction: copy it to colmap/sparse/0
    (the previous one is kept under colmap/sparse/models/replaced_<n>), rewrite
    cameras.json / points_sfm.ply and the undistorted training set."""
    import pycolmap

    src = Path(model_dir)
    if not src.exists():
        raise FileNotFoundError(src)
    rec = pycolmap.Reconstruction(str(src))
    if ws.model.exists() and not os.path.samefile(src, ws.model):
        models_dir = ws.sparse / "models"
        models_dir.mkdir(parents=True, exist_ok=True)
        n = 0
        while (models_dir / f"replaced_{n}").exists():
            n += 1
        shutil.move(str(ws.model), str(models_dir / f"replaced_{n}"))
        log.info("previous model kept as %s", models_dir / f"replaced_{n}")
    if not ws.model.exists():
        ws.model.mkdir(parents=True, exist_ok=True)
        rec.write(ws.model)
        rec.write_text(ws.model)
    model = SceneModel.from_colmap(ws.model, ws.frames_meta)
    write_cameras_json(model, ws.cameras_json)
    write_point_ply(ws.points_ply, model.points, model.colors)
    for stale in (ws.scene_json,):
        if stale.exists():
            stale.unlink()
            log.info("removed %s (re-run `gsu align`)", stale)
    if undistort:
        with Timer("undistorting images for training"):
            if ws.undistorted.exists():
                shutil.rmtree(ws.undistorted)
            pycolmap.undistort_images(output_path=ws.undistorted, input_path=ws.model, image_path=ws.frames,
                                      output_type="COLMAP")
    log.info("workspace now uses model with %d images, %d points", model.n_images, len(model.points))
    return model
