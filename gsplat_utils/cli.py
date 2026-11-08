"""`gsu` command line.

    gsu ingest   --work W (--video V | --images DIR)   frames + frames_meta.json
    gsu sfm      --work W                              COLMAP sparse model, cameras.json
    gsu use-model --work W --model DIR                 switch to another sub-model
    gsu align    --work W                              scene.json (up axis, floor, scale)
    gsu train    --work W                              Gaussian splat (GPU, gsplat)
    gsu export   --work W                              model.splat for the web viewer
    gsu geometry --work W                              corridor/wall geometry + diagrams
    gsu poses    --work W ...                          named camera poses for the website
    gsu web      --work W                              static WebGPU site
    gsu run      --work W (--video V | --images DIR)   ingest -> sfm -> align -> geometry
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .common import log, setup_logging
from .paths import Workspace


def _add_work(p: argparse.ArgumentParser) -> None:
    p.add_argument("--work", required=True, type=Path, help="work directory for this scene")


def _add_ingest_args(p: argparse.ArgumentParser) -> None:
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--video", type=Path, help="input video file")
    src.add_argument("--images", type=Path, help="input folder of photos")
    p.add_argument("--frames", type=int, default=300,
                   help="number of frames to keep (sharpest per time window)")
    p.add_argument("--every", type=int, default=None,
                   help="video only: keep every N-th frame instead of sharpness selection")
    p.add_argument("--max-size", type=int, default=None,
                   help="cap the long image side in pixels (default: native for video, 1600 for photos)")
    p.add_argument("--start", type=float, default=0.0, help="video start time (s)")
    p.add_argument("--end", type=float, default=None, help="video end time (s)")
    p.add_argument("--min-gap", type=float, default=0.0, help="video: min seconds between kept frames")
    p.add_argument("--sampling", choices=["motion", "uniform"], default="motion",
                   help="video: motion-adaptive windows (default) or uniform time windows")
    p.add_argument("--drop-blur", type=float, default=0.1,
                   help="video: drop frames sharper than this fraction of the median (0 = keep all)")
    p.add_argument("--sequential", action="store_true",
                   help="photos were shot as an ordered walk (use sequential matching)")


def cmd_ingest(a: argparse.Namespace) -> None:
    ws = Workspace(a.work)
    ws.mkdirs()
    if a.video:
        from .ingest.video import extract_video_frames

        extract_video_frames(
            a.video, ws.frames, target_frames=a.frames, every=a.every, max_size=a.max_size,
            start_s=a.start, end_s=a.end, min_gap_s=a.min_gap, drop_blur=a.drop_blur,
            sampling=a.sampling, meta_path=ws.frames_meta,
        )
    else:
        from .ingest.images import import_image_dir

        import_image_dir(
            a.images, ws.frames, max_size=a.max_size or 1600,
            target_frames=a.frames if a.frames else None,
            sequential=a.sequential, meta_path=ws.frames_meta,
        )
    log.info("frames written to %s", ws.frames)


def _add_sfm_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--matcher", choices=["auto", "sequential", "exhaustive", "vocabtree"], default="auto")
    p.add_argument("--preset", choices=["auto", "video", "photos"], default="auto",
                   help="feature/mapper settings: 'video' = dense features + permissive registration")
    p.add_argument("--focal", type=float, default=None,
                   help="prior focal length in pixels (phone video ~0.85 x image width)")
    p.add_argument("--intrinsics-prior", default="auto",
                   help="'auto' reuses intrinsics from a previous run, 'none', or a COLMAP model dir")
    p.add_argument("--overlap", type=int, default=15, help="sequential matcher window")
    p.add_argument("--loop-closure", action="store_true", help="sequential: also detect loops (needs --vocab-tree)")
    p.add_argument("--vocab-tree", type=Path, default=None, help="COLMAP vocabulary tree file")
    p.add_argument("--camera-model", default="OPENCV", help="COLMAP camera model (OPENCV, SIMPLE_RADIAL, PINHOLE, ...)")
    p.add_argument("--max-features", type=int, default=8192)
    p.add_argument("--max-image-size", type=int, default=1600, help="SIFT extraction image size cap")
    p.add_argument("--no-undistort", action="store_true", help="skip undistorted export for training")
    p.add_argument("--use-gpu", action="store_true", help="use CUDA SIFT if pycolmap has CUDA")
    p.add_argument("--mapper", choices=["auto", "incremental", "global"], default="auto",
                   help="global = GLOMAP (default for video), incremental = classic COLMAP")


def cmd_sfm(a: argparse.Namespace) -> None:
    from .sfm.colmap import run_sfm

    run_sfm(
        Workspace(a.work), matcher=a.matcher, overlap=a.overlap, loop_closure=a.loop_closure,
        vocab_tree=a.vocab_tree, camera_model=a.camera_model, max_features=a.max_features,
        max_image_size=a.max_image_size, undistort=not a.no_undistort, use_gpu=a.use_gpu,
        mapper=a.mapper, preset=a.preset, focal_px=a.focal, intrinsics_prior=a.intrinsics_prior,
        verbose=a.verbose,
    )


def cmd_use_model(a: argparse.Namespace) -> None:
    from .sfm.colmap import promote_model

    promote_model(Workspace(a.work), a.model, undistort=not a.no_undistort)


def _add_align_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--camera-height", type=float, default=1.5,
                   help="assumed camera height above the floor in metres (sets metric scale)")
    p.add_argument("--scale", type=float, default=None, help="override metric scale (metres per model unit)")
    p.add_argument("--walking-speed", type=float, default=1.0,
                   help="assumed walking speed (m/s) used as a scale sanity check / fallback for video")
    p.add_argument("--up", choices=["auto", "cameras", "floor"], default="auto",
                   help="how to estimate the up axis")
    if not any(o.dest == "model" for o in p._actions):
        p.add_argument("--model", type=Path, default=None,
                       help="COLMAP model directory to use instead of work/colmap/sparse/0 (e.g. a sub-model)")
    if not any(o.dest == "source" for o in p._actions):
        p.add_argument("--source", choices=["auto", "sfm", "splat"], default="auto",
                       help="point source for the floor fit (COLMAP points or trained splat)")


def cmd_align(a: argparse.Namespace) -> None:
    from .geometry.align import run_align

    run_align(Workspace(a.work), camera_height=a.camera_height, scale=a.scale,
              up_method=a.up, source=getattr(a, "source", "auto"), walking_speed=a.walking_speed,
              model_dir=getattr(a, "model", None))


def _add_train_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--iters", type=int, default=30000)
    p.add_argument("--strategy", choices=["default", "mcmc"], default="default")
    p.add_argument("--max-gaussians", type=int, default=2_000_000)
    p.add_argument("--sh-degree", type=int, default=3)
    p.add_argument("--downscale", type=int, default=1, help="train on 1/N resolution images")
    p.add_argument("--data", choices=["undistorted", "sparse"], default="undistorted")
    p.add_argument("--eval-every", type=int, default=0, help="hold out every N-th image for PSNR")
    p.add_argument("--save-every", type=int, default=10000)
    p.add_argument("--resume", type=Path, default=None, help="checkpoint .pt to resume from")
    p.add_argument("--device", default="cuda")


def cmd_train(a: argparse.Namespace) -> None:
    from .train.trainer import TrainConfig, train

    cfg = TrainConfig(
        iters=a.iters, strategy=a.strategy, max_gaussians=a.max_gaussians, sh_degree=a.sh_degree,
        downscale=a.downscale, data=a.data, eval_every=a.eval_every, save_every=a.save_every,
        resume=a.resume, device=a.device,
    )
    train(Workspace(a.work), cfg)


def _add_export_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--ply", type=Path, default=None, help="3DGS PLY to export (default: work/splat/model.ply)")
    p.add_argument("--min-opacity", type=float, default=0.02)
    p.add_argument("--aligned", action="store_true", help="bake scene.json alignment into the export")
    p.add_argument("--format", choices=["splat", "ply"], default="splat")


def cmd_export(a: argparse.Namespace) -> None:
    from .export.splat import run_export

    run_export(Workspace(a.work), ply=a.ply, min_opacity=a.min_opacity, aligned=a.aligned, fmt=a.format)


def _add_geometry_args(p: argparse.ArgumentParser) -> None:
    if not any(o.dest == "source" for o in p._actions):
        p.add_argument("--source", choices=["auto", "sfm", "splat"], default="auto",
                       help="points from the COLMAP model or from the trained splat")
    if not any(o.dest == "model" for o in p._actions):
        p.add_argument("--model", type=Path, default=None,
                       help="COLMAP model directory to use instead of work/colmap/sparse/0")
    p.add_argument("--cell", type=float, default=0.15, help="plan grid cell size (m)")
    p.add_argument("--max-width", type=float, default=6.0, help="max corridor half-width searched (m)")
    p.add_argument("--wall-height", type=float, default=2.6, help="extruded wall height (m)")
    p.add_argument("--cutaway", type=float, default=1.2,
                   help="wall height used in the axonometric sign diagram (m)")
    p.add_argument("--path-sample", type=float, default=0.25, help="path resampling step (m)")
    p.add_argument("--stair-slope", type=float, default=0.22, help="|dz/ds| above which the path is on stairs")
    p.add_argument("--manhattan", action="store_true", default=True,
                   help="snap walls/corridor to the two dominant directions (default on)")
    p.add_argument("--no-manhattan", dest="manhattan", action="store_false")
    p.add_argument("--labels", type=Path, default=None,
                   help="JSON of labels to draw: [{\"text\":\"Studio\",\"image\":\"frame_000123.jpg\"} | {\"text\":..,\"xyz\":[..]}]")
    p.add_argument("--wall-bins", type=int, default=2,
                   help="height bins (of 4) a grid cell must cover to count as wall (1 = any point)")
    p.add_argument("--explode", type=float, default=0.0,
                   help="extra vertical spacing per level in iso.svg (m); 0 = true heights")
    p.add_argument("--title", default=None, help="title drawn on the diagrams")


def cmd_geometry(a: argparse.Namespace) -> None:
    from .geometry.pipeline import GeometryConfig, run_geometry

    cfg = GeometryConfig(
        source=a.source, cell=a.cell, max_half_width=a.max_width, wall_height=a.wall_height,
        cutaway_height=a.cutaway, path_step=a.path_sample, stair_slope=a.stair_slope,
        manhattan=a.manhattan, labels=a.labels, model_dir=getattr(a, "model", None),
        explode=a.explode, title=a.title, min_wall_bins=a.wall_bins,
    )
    run_geometry(Workspace(a.work), cfg)


def _add_poses_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--set", nargs=2, action="append", metavar=("ID", "IMAGE_OR_TIME"), default=[],
                   help="name a pose from a training image name or a video time in seconds, e.g. --set studio 42.5")
    p.add_argument("--label", nargs=2, action="append", metavar=("ID", "LABEL"), default=[])
    p.add_argument("--fov", type=float, default=None, help="vertical field of view override (deg)")
    p.add_argument("--list", action="store_true", help="print the current poses")


def cmd_poses(a: argparse.Namespace) -> None:
    from .web.poses import run_poses

    run_poses(Workspace(a.work), sets=a.set, labels=a.label, fov=a.fov, do_list=a.list)


def _add_web_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--out", type=Path, default=None, help="output site folder (default work/web)")
    p.add_argument("--splat", type=Path, default=None, help=".splat/.ply to ship (default work/splat/model.splat)")
    p.add_argument("--title", default="gsplat")
    p.add_argument("--serve", action="store_true", help="serve the site on http://localhost:8000")
    p.add_argument("--port", type=int, default=8000)


def cmd_web(a: argparse.Namespace) -> None:
    from .web.build import run_web

    run_web(Workspace(a.work), out=a.out, splat=a.splat, title=a.title, serve=a.serve, port=a.port)


def cmd_run(a: argparse.Namespace) -> None:
    """ingest -> sfm -> align -> geometry (CPU parts). Training is separate (GPU)."""
    ws = Workspace(a.work)
    if not ws.frames_meta.exists() or a.force:
        cmd_ingest(a)
    else:
        log.info("frames exist, skipping ingest (use --force to redo)")
    if not ws.model.exists() or a.force:
        cmd_sfm(a)
    else:
        log.info("COLMAP model exists, skipping sfm")
    cmd_align(a)
    cmd_geometry(a)


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="gsu", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-v", "--verbose", action="store_true")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("ingest", help="video/photos -> frames"); _add_work(p); _add_ingest_args(p); p.set_defaults(fn=cmd_ingest)
    p = sub.add_parser("sfm", help="COLMAP structure from motion"); _add_work(p); _add_sfm_args(p); p.set_defaults(fn=cmd_sfm)
    p = sub.add_parser("use-model", help="promote a COLMAP sub-model (e.g. colmap/sparse/models/3) to the workspace model")
    _add_work(p); p.add_argument("--model", required=True, type=Path); p.add_argument("--no-undistort", action="store_true")
    p.set_defaults(fn=cmd_use_model)
    p = sub.add_parser("align", help="up axis / floor / metric scale"); _add_work(p); _add_align_args(p); p.set_defaults(fn=cmd_align)
    p = sub.add_parser("train", help="train the Gaussian splat (GPU)"); _add_work(p); _add_train_args(p); p.set_defaults(fn=cmd_train)
    p = sub.add_parser("export", help="export splat for the web"); _add_work(p); _add_export_args(p); p.set_defaults(fn=cmd_export)
    p = sub.add_parser("geometry", help="corridor/wall geometry + diagrams"); _add_work(p); _add_geometry_args(p); p.set_defaults(fn=cmd_geometry)
    p = sub.add_parser("poses", help="named camera poses for the website"); _add_work(p); _add_poses_args(p); p.set_defaults(fn=cmd_poses)
    p = sub.add_parser("web", help="build the WebGPU site"); _add_work(p); _add_web_args(p); p.set_defaults(fn=cmd_web)
    p = sub.add_parser("run", help="ingest -> sfm -> align -> geometry"); _add_work(p)
    _add_ingest_args(p); _add_sfm_args(p); _add_align_args(p); _add_geometry_args(p)
    p.add_argument("--force", action="store_true", help="redo stages whose outputs exist")
    p.set_defaults(fn=cmd_run)
    return ap


def main(argv: list[str] | None = None) -> int:
    ap = build_parser()
    a = ap.parse_args(argv)
    setup_logging(a.verbose)
    try:
        a.fn(a)
    except KeyboardInterrupt:
        return 130
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
