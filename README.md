# gsplat-utils

Turn a walk-through **video** or a **folder of photos** into

1. a **Gaussian splat** (3DGS) you can ship on a **WebGPU website** that glides
   between named camera poses ("studio", "about", "shop", ...), and
2. a **basic 3D building diagram** (corridors, stairs, levels, doorways) that is
   good enough for a wayfinding sign, extracted from the same reconstruction.

Everything is driven by one CLI, `gsu`, working inside a *work directory*:

```
video / photos ──ingest──▶ frames ──sfm──▶ COLMAP model ──train (GPU)──▶ splat ──export──▶ .splat ──web──▶ site
                                              │                                   ▲
                                              └──align──▶ scene.json ──geometry──▶ diagram.json / SVG / GLB
                                                            (uses the splat's gaussians when present, else SfM points)
```

Only `gsu train` needs a GPU. Frame extraction, structure-from-motion (COLMAP /
GLOMAP via `pycolmap`), alignment, the geometry diagram and the website build all
run on a CPU.

## Install

```bash
pip install -e .                 # CPU stages: numpy, scipy, pillow, pycolmap
pip install -e '.[train]'        # + torch + gsplat, on a CUDA machine
pip install -e '.[dev]'          # pytest, trimesh (used by the tests)
```

`ffmpeg`/`ffprobe` must be on your PATH for video input (`pip install imageio-ffmpeg`
also works).

## Quick start (CPU parts)

```bash
# 1. frames: 1400 sharp frames, denser where the camera moves fast
gsu ingest --work work --video video/input.mp4 --frames 1400

# 2. camera poses + sparse points (GLOMAP global mapper, video preset)
gsu sfm --work work                      # add --focal 400 if you know the focal length in px

# 3. gravity, floor, metric scale -> scene.json
gsu align --work work --camera-height 1.5

# 4. corridors / stairs / levels -> diagrams
gsu geometry --work work --title "Building A" --labels examples/labels.example.json

# or all of the above in one go
gsu run --work work --video video/input.mp4 --frames 1400
```

Outputs land in `work/geometry/`:

| file | what |
|---|---|
| `diagram.json` | levels, corridor runs (straight pieces with left/right widths, openings), walls, stairs, path, labels. Metres, +Z up. |
| `plan_L0.svg`, `plan_L1.svg`, ... `plan_all.svg` | top-down plans, one per level |
| `iso.svg`, `iso_exploded.svg` | axonometric "sign" view: floor ribbons, cut-away walls, stairs with balustrades, path, labels |
| `diagram.glb`, `diagram.obj` + `.mtl` | the same as meshes (Y up) for Blender / three.js / Illustrator 3D |
| `path.json` | the smoothed walk path (also used by the website's `route` transition) |
| `occupancy_L*.png` | debug: wall-band points, the path and the detected wall hits per level |

The SVGs import directly into Illustrator/Figma/Inkscape; the diagram is deliberately
coarse (straight corridor pieces, right angles, one width per piece) so it reads as a
sign, not as a survey.

## Photos instead of video

```bash
gsu ingest --work work --images ./photos            # unordered photos: exhaustive matching
gsu ingest --work work --images ./photos --sequential   # photos shot as a walk
gsu sfm --work work --preset photos
```

## Training the splat (GPU machine)

Copy the work directory (only `frames/` and `colmap/` are needed) to a machine with
an NVIDIA GPU:

```bash
pip install -e '.[train]'
gsu train --work work --iters 30000 --strategy default     # or --strategy mcmc --max-gaussians 1500000
gsu export --work work --aligned                            # work/splat/model.splat (+ sidecar .json)
gsu geometry --work work --source splat                     # re-run the diagram on the dense gaussians
```

`gsu train` writes `work/splat/model.ply` in the standard 3DGS layout (loads in
any splat viewer), checkpoints, and `train_log.json`. The world frame is kept in
COLMAP coordinates so cameras, poses and the diagram line up; `gsu export --aligned`
bakes the gravity/metric alignment from `scene.json` into the web model.

## The website (use case 1)

```bash
# name the places visitors can jump to: by video time (s), frame index or image name
gsu poses --work work --set studio 42.5 --set about 71 --set shop frame_003300.jpg
gsu poses --work work --label studio "The Studio" --list
gsu web --work work --title "My space" --serve        # http://localhost:8000
```

`gsu web` copies a dependency-free WebGPU viewer (`gsplat_utils/web/template`) next
to the model and writes `viewer-config.json`. Before training it will happily show the
SfM point cloud so you can test navigation. Press **P** in the viewer to copy the
current viewpoint as JSON for `poses.json`; **H** hides the UI; `#studio` in the URL
deep-links a pose.

### Transition options (the art part)

Defaults live in `poses.json -> transitions.default`; any pose can override them with
its own `"transition": {...}`; the on-page panel changes them live.

| option | values |
|---|---|
| `mode` | `linear` (lerp + slerp), `spline` (Catmull-Rom through `via` points), `arc` (swing out around the midpoint; `arc_height`, `arc_side`), `dolly` (move then turn; `dolly_overlap`, `dolly_order`), `route` (follow the real walk path between the two poses; `route_blend`, `look_ahead`, `route_smooth`) |
| `easing` | `linear`, `smoothstep`, `easeInOutCubic`, `easeInOutQuint`, `easeInOutExpo`, `easeOutBack`, or a cubic bezier `[x1,y1,x2,y2]` |
| `duration_ms`, `duration_per_metre` | fixed time, or time proportional to distance |
| `up_lock` | `true` keeps the horizon level (yaw/pitch interpolation), `false` = full slerp |
| `fov_pulse` | dolly-zoom style field-of-view breathing at mid transition |
| `effect`, `effect_strength` | per-splat effect during the move: `none`, `fade`, `scatter`, `pulse`, `desaturate`, `dissolve` |

`route` is the one that feels like walking through the building: the camera follows the
corridor you filmed instead of cutting through walls. The viewer needs a WebGPU browser
(Chrome/Edge 113+, Safari 18+, recent Firefox) and any static host.

## How the geometry stage works (use case 2)

* **Alignment** (`gsu align`): gravity from the phone's image-down axis averaged over
  all cameras, refined by the floor points; the floor is found as the lower edge of
  the point-height distribution relative to the (horizontally) nearest camera, which
  works per floor of a multi-storey walk. Metric scale = assumed camera height
  (`--camera-height`, default 1.5 m) over that distance, sanity-checked against a
  walking-speed prior (`--walking-speed`, default 1 m/s) when video timestamps exist.
  `--scale` overrides everything.
* **Path**: mis-registered cameras are dropped, the track is smoothed and resampled
  every 0.25 m. Runs with |dz/ds| above `--stair-slope` (0.22) and more than 0.6 m
  of rise are **stairs**; the flat runs between them are grouped into **levels**.
* **Corridors**: per level, points in a wall band (0.35–2.1 m above the floor) go into
  a 0.15 m occupancy grid that counts *how many height bins* a cell covers, so
  handrails, furniture and stray points do not stop a ray while walls do. Left/right
  rays from every path sample give the corridor half-widths; long gaps where no wall
  is hit become **openings** (doors, side corridors). The path is snapped to the two
  dominant directions (`--no-manhattan` to keep raw angles) and each straight piece
  gets one width per side.
* **Diagram**: floor ribbons per run, walls extruded to `--wall-height` (3D files) and
  cut away to `--cutaway` (1.2 m) in the axonometric SVG, stairs as sloped ribbons
  with treads and balustrades, the walk path, and labels from `--labels`, from
  `poses.json` (your website spots double as sign destinations) plus Start/End and
  the stairs.

Sub-models: if the walk breaks (blur, a fast turn) COLMAP returns several models.
`gsu sfm` keeps the largest as `colmap/sparse/0` and the rest in
`colmap/sparse/models/`; `gsu use-model --model colmap/sparse/models/2` switches.
The GLOMAP mapper (default for video) is much better at keeping one model.

## Tips for filming

* Walk slowly, keep the phone upright and at a steady height; turn slowly at corners.
* Avoid long stretches of blank wall in frame; look slightly ahead, not at the floor.
* Overlap the start and end of the walk (or walk back) if you want loop closure.
* Higher resolution helps a lot: 480p footage like the sample needs the `video` preset
  (dense SIFT, GLOMAP) to reconstruct at all.

## Development

```bash
pytest                      # python tests (synthetic building end to end, exports, poses)
node --test                 # viewer transition maths, loaders, sort worker
```

Layout: `gsplat_utils/ingest` (frames), `sfm` (pycolmap), `geometry` (align, path,
occupancy, corridor, diagram, svg, mesh), `train` (gsplat), `export` (PLY / .splat),
`web` (poses, site build, `template/` = the viewer).
