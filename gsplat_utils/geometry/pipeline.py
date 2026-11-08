"""`gsu geometry`: aligned scene -> levels, corridors, walls, stairs -> diagrams.

Outputs in work/geometry/:
  diagram.json      everything below in one structured file (metres, +Z up)
  path.json         smoothed, resampled walk path (used by the web viewer's route mode)
  plan_L<k>.svg     one plan per level (+ plan_all.svg)
  iso.svg           axonometric view for the sign (cut-away walls, stairs, labels)
  diagram.glb/.obj  3D meshes (Y up) for Blender / three.js / etc.
  occupancy_L<k>.png  debug: wall-band points, path and wall hits per level
  points_aligned.ply  the scene points in the aligned frame
"""
from __future__ import annotations

import os
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np

from ..common import log, read_json, write_json
from ..export.ply import write_point_ply
from ..paths import Workspace
from ..sfm.model import SceneModel
from .align import SceneAlignment, run_align
from .corridor import Piece, Run, build_run, drop_corner_openings, piece_openings, piece_widths
from .mesh import Mesh, write_glb, write_obj
from .occupancy import OccupancyGrid, corridor_half_widths
from .path import (Level, PathSegment, dominant_angle, polyline_segments, resample_by_arclength,
                   segment_levels, smooth_positions, snap_rectilinear, tangents_xy, track_inliers)
from .points import load_scene
from .svg import AxoView, write_axo_svg, write_plan_svg


@dataclass
class GeometryConfig:
    source: str = "auto"
    cell: float = 0.15
    max_half_width: float = 6.0
    wall_height: float = 2.6
    cutaway_height: float = 1.2
    path_step: float = 0.25
    stair_slope: float = 0.22
    manhattan: bool = True
    labels: os.PathLike | str | None = None
    explode: float = 0.0              # extra vertical spacing per level in iso.svg
    iso_azimuth: float = -55.0
    iso_elevation: float = 35.0
    band_lo: float = 0.35             # wall band above the floor (m)
    band_hi: float = 2.1
    min_run_m: float = 1.5            # shortest straight piece / stairs run
    smooth_sigma_m: float = 0.25      # camera path smoothing (metres along the track)
    default_half_width: float = 1.0   # when no wall is found on a side
    min_wall_bins: int = 2            # a wall cell has points in >= this many of 4 height bins
    debug_png: bool = True
    model_dir: os.PathLike | str | None = None
    title: str | None = None
    stairs_labels: bool = True
    stairs_wall_height: float = 0.9     # balustrade height drawn beside flights


def _grid_for(points: np.ndarray, z_lo: float, z_hi: float, centre_xy: np.ndarray, cell: float,
              reach: float, z_bins: int = 4) -> OccupancyGrid:
    band = points[(points[:, 2] >= z_lo) & (points[:, 2] <= z_hi)]
    bounds = (centre_xy[:, 0].min() - reach, centre_xy[:, 1].min() - reach,
              centre_xy[:, 0].max() + reach, centre_xy[:, 1].max() + reach)
    if len(band):
        inb = ((band[:, 0] >= bounds[0]) & (band[:, 0] <= bounds[2]) &
               (band[:, 1] >= bounds[1]) & (band[:, 1] <= bounds[3]))
        band = band[inb]
    return OccupancyGrid.from_points(band[:, :2], cell, bounds=bounds, pad=0.5, z=band[:, 2],
                                     z_range=(z_lo, z_hi), z_bins=z_bins)


def _grid_for_stairs(points: np.ndarray, centre: np.ndarray, cell: float, reach: float,
                     lo: float = -1.0, hi: float = 0.6) -> OccupancyGrid:
    """Occupancy for a flight of stairs: keep points whose height relative to the
    *local* camera height is in [lo, hi] (walls beside the flight), which excludes
    the treads below and the ceiling above even though the flight climbs."""
    from scipy.spatial import cKDTree

    bounds = (centre[:, 0].min() - reach, centre[:, 1].min() - reach,
              centre[:, 0].max() + reach, centre[:, 1].max() + reach)
    inb = ((points[:, 0] >= bounds[0]) & (points[:, 0] <= bounds[2]) &
           (points[:, 1] >= bounds[1]) & (points[:, 1] <= bounds[3]))
    cand = points[inb]
    z_rel = np.zeros(len(cand))
    if len(cand) and len(centre):
        k = cKDTree(centre[:, :2]).query(cand[:, :2], k=1)[1]
        z_rel = cand[:, 2] - centre[k, 2]
        keep = (z_rel >= lo) & (z_rel <= hi)
        cand, z_rel = cand[keep], z_rel[keep]
    return OccupancyGrid.from_points(cand[:, :2], cell, bounds=bounds, pad=0.5, z=z_rel, z_range=(lo, hi), z_bins=4)


def resolve_labels(cfg: GeometryConfig, ws: Workspace, model_aligned: SceneModel, al: SceneAlignment,
                   path_pts: np.ndarray, path_level: np.ndarray, levels: list[Level],
                   camera_height: float) -> list[dict]:
    """Labels from --labels JSON, poses.json (website nav spots) and start/end."""
    entries: list[dict] = []
    if cfg.labels:
        entries += read_json(cfg.labels)
    if ws.poses_json.exists():
        for p in read_json(ws.poses_json).get("poses", []):
            entries.append({"text": p.get("label") or p["id"], "image": p.get("source_image"),
                            "xyz_colmap": p.get("position")})
    names = list(model_aligned.names)
    out = []
    for e in entries:
        xyz = None
        if e.get("image") and e["image"] in names:
            xyz = model_aligned.positions[names.index(e["image"])]
        elif e.get("time_s") is not None and model_aligned.time_s is not None:
            k = int(np.nanargmin(np.abs(model_aligned.time_s - float(e["time_s"]))))
            xyz = model_aligned.positions[k]
        elif e.get("frame_index") is not None and model_aligned.frame_index is not None:
            k = int(np.argmin(np.abs(model_aligned.frame_index - int(e["frame_index"]))))
            xyz = model_aligned.positions[k]
        elif e.get("xyz") is not None:
            xyz = np.array(e["xyz"], float)
            if e.get("frame", "aligned") == "colmap":
                xyz = al.apply_points(xyz[None])[0]
        elif e.get("xyz_colmap") is not None:
            xyz = al.apply_points(np.array(e["xyz_colmap"], float)[None])[0]
        if xyz is None:
            log.warning("label %r could not be placed", e.get("text"))
            continue
        out.append(_label_at(e["text"], xyz, path_pts, path_level, levels, camera_height))
    out.append(_label_at("Start", path_pts[0], path_pts, path_level, levels, camera_height))
    out.append(_label_at("End", path_pts[-1], path_pts, path_level, levels, camera_height))
    return out


def _label_at(text, xyz, path_pts, path_level, levels, camera_height) -> dict:
    k = int(np.argmin(np.linalg.norm(path_pts[:, :2] - xyz[:2], axis=1)))
    lv = int(path_level[k])
    if lv < 0:  # on stairs: nearest flat sample
        flat = np.nonzero(path_level >= 0)[0]
        lv = int(path_level[flat[np.argmin(np.abs(flat - k))]]) if len(flat) else 0
    floor_z = levels[lv].floor_z if 0 <= lv < len(levels) else float(xyz[2] - camera_height)
    return {"text": text, "xyz": [float(xyz[0]), float(xyz[1]), float(floor_z)], "level": lv, "floor_z": floor_z}


def run_geometry(ws: Workspace, cfg: GeometryConfig) -> dict:
    model, src = load_scene(ws, source=cfg.source, model_dir=cfg.model_dir)
    if ws.scene_json.exists():
        al = SceneAlignment.load(ws.scene_json)
    else:
        log.info("no scene.json; running alignment first")
        al = run_align(ws, source=cfg.source, model_dir=cfg.model_dir)
    keep = track_inliers(model.positions)
    if not keep.all():
        log.info("dropping %d mis-registered cameras from the path", int((~keep).sum()))
        model = model.subset(keep)
    am = al.apply(model)
    camera_height = al.camera_height_m
    ws.geometry.mkdir(parents=True, exist_ok=True)
    write_point_ply(ws.geometry / "points_aligned.ply", am.points, am.colors)

    # ---- camera path --------------------------------------------------------
    cams = am.positions
    step_m = float(np.median(np.linalg.norm(np.diff(cams, axis=0), axis=1))) if len(cams) > 1 else 1.0
    sm = smooth_positions(cams, cfg.smooth_sigma_m / max(step_m, 1e-6))
    res, src_idx = resample_by_arclength(sm, cfg.path_step)
    tang = tangents_xy(res)
    segments, levels = segment_levels(res, cfg.path_step, camera_height, cfg.stair_slope, cfg.min_run_m)
    path_level = np.full(len(res), -1, int)
    for sg in segments:
        if sg.kind == "flat":
            path_level[sg.i0:sg.i1] = sg.level
    ceiling_h = al.ceiling_height_m or cfg.wall_height
    for lv in levels:
        lv.ceiling_z = lv.floor_z + ceiling_h
    log.info("path %.1f m, %d samples, %d levels, %d stairs runs", res.shape[0] * cfg.path_step, len(res),
             len(levels), sum(1 for s in segments if s.kind == "stairs"))

    # ---- wall distances ----------------------------------------------------
    min_count = cfg.min_wall_bins   # height bins (of 4) a cell must cover to count as wall
    widths = {k: np.full(len(res), np.nan) for k in ("left_raw", "right_raw", "left", "right")}
    grids: dict[int, OccupancyGrid] = {}
    reach = cfg.max_half_width + 1.0
    for lv in levels:
        idx = np.nonzero(path_level == lv.id)[0]
        if len(idx) == 0:
            continue
        g = _grid_for(am.points, lv.floor_z + cfg.band_lo, lv.floor_z + cfg.band_hi, res[idx, :2], cfg.cell, reach)
        grids[lv.id] = g
        w = corridor_half_widths(g, res[idx, :2], tang[idx], cfg.max_half_width, min_count=min_count,
                                 support=1, step=cfg.path_step)
        for k in widths:
            widths[k][idx] = w[k]
    for sg in segments:
        if sg.kind != "stairs":
            continue
        idx = np.arange(sg.i0, sg.i1)
        g = _grid_for_stairs(am.points, res[idx], cfg.cell, reach, lo=-(camera_height - 0.5), hi=0.6)
        w = corridor_half_widths(g, res[idx, :2], tang[idx], cfg.max_half_width, min_count=min_count,
                                 support=1, step=cfg.path_step)
        for k in widths:
            widths[k][idx] = w[k]

    # ---- straight pieces -----------------------------------------------------
    flat_idx = np.nonzero(path_level >= 0)[0]
    theta = dominant_angle(tang[flat_idx] if len(flat_idx) else tang)
    runs: list[Run] = []
    for ri, sg in enumerate(segments):
        if cfg.manhattan:
            segs = snap_rectilinear(res, sg.i0, sg.i1, theta, cfg.min_run_m, cfg.path_step)
        else:
            segs = polyline_segments(res, sg.i0, sg.i1, tol_m=0.4)
        if not segs:
            continue
        pieces = []
        for s in segs:
            s.kind = sg.kind
            s.level = sg.level
            l, r, lf, rf = piece_widths(widths, s, cfg.default_half_width, max_half=cfg.max_half_width)
            ops = piece_openings(widths, s, cfg.path_step) if sg.kind == "flat" else []
            pieces.append(Piece(s, l, r, ops, lf, rf))
        drop_corner_openings(pieces)
        floor_z = levels[sg.level].floor_z if sg.kind == "flat" else None
        runs.append(build_run(len(runs), sg.kind, pieces, level=sg.level, level_from=sg.level_from,
                              level_to=sg.level_to, floor_z=floor_z, camera_height=camera_height))

    labels = resolve_labels(cfg, ws, am, al, res, path_level, levels, camera_height)
    if cfg.stairs_labels:
        for r in runs:
            if r.kind == "stairs" and r.pieces:
                mid = 0.5 * (r.left_boundary[len(r.left_boundary) // 2] + r.right_boundary[len(r.right_boundary) // 2])
                up_down = "up" if r.level_to > r.level_from else "down"
                lv_name = levels[r.level_to].name if 0 <= r.level_to < len(levels) else "?"
                labels.append({"text": f"Stairs {up_down} to {lv_name}", "xyz": [float(mid[0]), float(mid[1]), float(mid[2])],
                               "level": int(min(r.level_from, r.level_to)) if r.level_from >= 0 else r.level_to,
                               "floor_z": float(mid[2]), "kind": "stairs"})

    # ---- diagram.json -------------------------------------------------------
    diagram = {
        "frame": "aligned", "units": "m", "camera_height_m": camera_height,
        "note": f"scale from assumed camera height {camera_height:.2f} m; {src} points",
        "scene_scale_m_per_unit": al.scale,
        "levels": [{"id": lv.id, "name": lv.name, "floor_z": lv.floor_z, "ceiling_z": lv.ceiling_z,
                    "camera_z": lv.camera_z} for lv in levels],
        "runs": [_run_json(r) for r in runs],
        "path": {"step": cfg.path_step, "points": res.round(4).tolist(), "level": path_level.tolist(),
                 "source_camera_index": src_idx.round(2).tolist()},
        "labels": labels,
        "config": {k: (str(v) if isinstance(v, Path) else v) for k, v in asdict(cfg).items()},
        "stats": {
            "n_cameras": int(am.n_images), "n_points": int(len(am.points)),
            "path_length_m": float(len(res) * cfg.path_step),
            "n_runs": len(runs), "n_walls": int(sum(len(r.walls) for r in runs)),
            "n_openings": int(sum(len(p.openings) for r in runs for p in r.pieces)),
            "width_found_fraction": float(np.mean(~np.isnan(widths["left_raw"]) | ~np.isnan(widths["right_raw"]))),
        },
    }
    write_json(ws.geometry / "diagram.json", diagram)
    write_json(ws.geometry / "path.json", {"frame": "aligned", "step": cfg.path_step,
                                           "points": res.round(4).tolist(), "level": path_level.tolist()})

    # ---- SVG ----------------------------------------------------------------
    allxy = np.vstack([np.array(r["outline"]) for r in diagram["runs"]] + [res[:, :2]])
    bounds = (allxy[:, 0].min(), allxy[:, 1].min(), allxy[:, 0].max(), allxy[:, 1].max())
    for lv in levels:
        write_plan_svg(ws.geometry / f"plan_L{lv.id}.svg", diagram, lv.id, bounds, title=f"{lv.name} (floor z={lv.floor_z:.1f} m)")
    write_plan_svg(ws.geometry / "plan_all.svg", diagram, None, bounds, title=cfg.title or "All levels")
    view = AxoView(cfg.iso_azimuth, cfg.iso_elevation, cfg.explode)
    write_axo_svg(ws.geometry / "iso.svg", diagram, view, cutaway_height=cfg.cutaway_height, title=cfg.title,
                  stairs_wall_height=cfg.stairs_wall_height)
    if len(levels) > 1 and cfg.explode == 0:
        view2 = AxoView(cfg.iso_azimuth, cfg.iso_elevation, explode_m=2.5)
        write_axo_svg(ws.geometry / "iso_exploded.svg", diagram, view2, cutaway_height=cfg.cutaway_height,
                      title=cfg.title, stairs_wall_height=cfg.stairs_wall_height)

    # ---- meshes -------------------------------------------------------------
    meshes = build_meshes(runs, res, path_level, cfg, camera_height)
    write_glb(ws.geometry / "diagram.glb", meshes)
    write_obj(ws.geometry / "diagram.obj", meshes)

    # ---- debug images -------------------------------------------------------
    if cfg.debug_png:
        for lv in levels:
            if lv.id in grids:
                idx = np.nonzero(path_level == lv.id)[0]
                _debug_png(ws.geometry / f"occupancy_L{lv.id}.png", grids[lv.id], res[idx], tang[idx],
                           widths, idx, [r for r in runs if r.kind == "flat" and r.level == lv.id])
    log.info("geometry written to %s (%d runs, %d levels)", ws.geometry, len(runs), len(levels))
    return diagram


def _run_json(r: Run) -> dict:
    return {
        "id": r.id, "kind": r.kind, "level": r.level, "level_from": r.level_from, "level_to": r.level_to,
        "pieces": [{"p0": p.p0.round(4).tolist(), "p1": p.p1.round(4).tolist(), "left": round(p.left, 3),
                    "right": round(p.right, 3), "axis": p.seg.axis, "length": round(p.length, 3),
                    "left_found": p.left_found, "right_found": p.right_found,
                    "openings": [{"side": o.side, "s0": round(o.s0, 3), "s1": round(o.s1, 3)} for o in p.openings]}
                   for p in r.pieces],
        "left_boundary": r.left_boundary.round(4).tolist(),
        "right_boundary": r.right_boundary.round(4).tolist(),
        "outline": r.outline.round(4).tolist(),
        "walls": [{"a": w.a.round(4).tolist(), "b": w.b.round(4).tolist(), "side": w.side, "piece": w.piece,
                   "z0": round(w.z0, 4), "z1": round(w.z1, 4), "openings": [list(o) for o in w.openings],
                   "solid": [list(s) for s in w.solid_parts()]} for w in r.walls],
    }


def build_meshes(runs: list[Run], path: np.ndarray, path_level: np.ndarray, cfg: GeometryConfig,
                 camera_height: float) -> list[Mesh]:
    floors = Mesh("floor", (0.90, 0.88, 0.84, 1.0))
    stairs = Mesh("stairs", (0.80, 0.76, 0.68, 1.0))
    walls = Mesh("walls", (0.55, 0.55, 0.55, 1.0))
    walls_low = Mesh("walls_cutaway", (0.45, 0.45, 0.47, 1.0))
    route = Mesh("path", (0.85, 0.28, 0.17, 1.0))
    slab = 0.12
    for r in runs:
        for quad in r.piece_quads():
            bottom = quad.copy(); bottom[:, 2] -= slab
            (stairs if r.kind == "stairs" else floors).add_prism(bottom, quad)
        if r.kind == "flat":
            for w in r.walls:
                for t0, t1 in w.solid_parts():
                    a = w.a + (w.b - w.a) * t0; b = w.a + (w.b - w.a) * t1
                    walls.add_box_between(np.r_[a, w.z0], np.r_[b, w.z0], 0.12, w.z0, w.z0 + cfg.wall_height)
                    walls_low.add_box_between(np.r_[a, w.z0], np.r_[b, w.z0], 0.14, w.z0, w.z0 + cfg.cutaway_height)
        else:
            for w in r.walls:   # sloped balustrades beside the flight
                walls_low.add_box_between(np.r_[w.a, w.z0], np.r_[w.b, w.z1], 0.1, 0.0, cfg.stairs_wall_height)
    # path as a thin ribbon 5 cm above the floor (camera height removed)
    pz = path.copy(); pz[:, 2] -= camera_height - 0.05
    for k in range(len(pz) - 1):
        route.add_box_between(pz[k], pz[k + 1], 0.12, 0.0, 0.04)
    return [floors, stairs, walls, walls_low, route]


def _debug_png(path, grid: OccupancyGrid, pts, tang, widths, idx, runs: list[Run]) -> None:
    from PIL import Image, ImageDraw

    occ = grid.counts
    h, w = occ.shape
    scale = max(1, int(900 / max(h, w)))
    img = Image.new("RGB", (w * scale, h * scale), "white")
    px = img.load()
    mx = max(1, int(np.percentile(occ[occ > 0], 90))) if (occ > 0).any() else 1
    for j in range(h):
        for i in range(w):
            c = occ[j, i]
            if c:
                v = int(230 - 200 * min(1.0, c / mx))
                for dy in range(scale):
                    for dx in range(scale):
                        px[i * scale + dx, (h - 1 - j) * scale + dy] = (v, v, v)
    dr = ImageDraw.Draw(img)

    def P(xy):
        i, j = (np.asarray(xy) - grid.origin) / grid.cell
        return (float(i * scale), float((h - j) * scale))

    for k, p in enumerate(pts):
        t = tang[k]; n = np.array([-t[1], t[0]])
        for key, sign, col in (("left_raw", 1, (40, 140, 240)), ("right_raw", -1, (240, 150, 40))):
            d = widths[key][idx[k]]
            if not np.isnan(d):
                q = p[:2] + n * sign * d
                dr.ellipse([P(q)[0] - 2, P(q)[1] - 2, P(q)[0] + 2, P(q)[1] + 2], fill=col)
    if len(pts) > 1:
        dr.line([P(p[:2]) for p in pts], fill=(220, 60, 40), width=2)
    for r in runs:
        ol = [P(q) for q in r.outline]
        dr.polygon(ol, outline=(20, 120, 20))
        for wl in r.walls:
            for t0, t1 in wl.solid_parts():
                a = wl.a + (wl.b - wl.a) * t0; b = wl.a + (wl.b - wl.a) * t1
                dr.line([P(a), P(b)], fill=(0, 0, 0), width=3)
    img.save(path)
