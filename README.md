# gsplat-utils

Turn a walk-through **video** or a **folder of photos** into

1. a **Gaussian splat** (3DGS) you can ship on a **WebGPU website** that glides
   between named camera poses ("studio", "about", "shop", ...), and
2. a **basic 3D building diagram** (corridors, stairs, levels, doorways) that is
   good enough for a wayfinding sign, extracted from the same reconstruction.

Everything is driven by one CLI, `gsu`, working inside a *work directory*:

```
video / photos ──ingest──▶ frames ──sfm──▶ COLMAP model(s) ──train (GPU)──▶ splat ──export──▶ .splat ──web──▶ site
                                              │
                                              ├──align──▶ scene.json            (gravity, floor, metric scale)
                                              └──rectify──▶ work/rectified      (drift-corrected, sub-models stitched)
                                                               └──geometry──▶ diagram.json / plan SVGs / iso SVG / GLB
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

# 4. one metric walk out of the sub-models: scale-drift correction + stitching
gsu rectify --work work

# 5. corridors / stairs / levels -> diagrams
gsu geometry --work work --title "Building A" --labels examples/labels.example.json

# or all of the above in one go
gsu run --work work --video video/input.mp4 --frames 1400
gsu status --work work                   # what each stage produced
```

If the reconstruction breaks into pieces at a blurry turn (`gsu sfm` says so), add every
frame around the break and re-run SfM reusing the existing matches:

```bash
gsu ingest --work work --video video/input.mp4 --add --start 90.5 --end 96.5 --every 1 --drop-blur 0
gsu sfm --work work --keep-database
gsu rectify --work work && gsu geometry --work work
```

Outputs land in `work/geometry/`:

| file | what |
|---|---|
| `diagram.json` | levels, corridor runs (straight pieces with left/right widths, openings), walls, stairs, path, labels. Metres, +Z up. |
| `plan_L0.svg`, `plan_L1.svg`, ... `plan_all.svg` | top-down plans, one per level |
| `iso.svg`, `iso_exploded.svg` | isometric "sign" view: floor slabs, open-well stairs with their void and gallery, halls, path, labels (walls only with `--walls`) |
| `diagram.glb`, `diagram.obj` + `.mtl` | the same as meshes (Y up) for Blender / three.js / Illustrator 3D |
| `path.json` | the smoothed walk path (also used by the website's `route` transition) |
| `occupancy_L*.png` | debug: wall-band points, the path and the detected wall hits per level |

The SVGs import directly into Illustrator/Figma/Inkscape (`iso.svg` is grouped into
one layer per level, plus `path` and `labels`); the diagram is deliberately coarse
(straight corridor pieces, right angles, one width per piece) so it reads as a sign,
not as a survey. Useful knobs for `gsu geometry`:

| flag | effect |
|---|---|
| `--title`, `--labels file.json` | caption and named places (see `examples/labels.example.json`; `poses.json` entries are added automatically) |
| `--explode 2.5` | pull the levels apart vertically in the isometric |
| `--iso-azimuth -45` | rotate the isometric view (degrees) |
| `--cutaway 1.2`, `--wall-height 2.6` | wall height in the SVG / in the 3D meshes |
| `--stairs-style auto|path|simple` | draw stairs along the real flights or as one sloped ribbon |
| `--no-path` | hide the dashed walk path |
| `--cell 0.15`, `--max-width 6`, `--wall-bins 2` | occupancy resolution, furthest wall searched, wall-likeness threshold |
| `--stair-slope 0.22`, `--path-sample 0.25` | stairs detection sensitivity, path resampling step |
| `--no-manhattan` | keep raw corridor angles instead of snapping to the two dominant directions |
| `--source splat` | use the trained gaussians instead of the SfM points |

`node scripts/svg2png.cjs work/geometry/iso.svg` rasterises an SVG with headless
Chromium if you have Playwright installed.

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
* **Rectify** (`gsu rectify`): structure-from-motion on forward motion through a
  textureless corridor is weak in scale: the reconstruction can shrink 20-50x
  along the corridor (we measured exactly that on the sample video). The camera's
  height above the floor is constant, so the floor band tracked through time gives
  the local scale, the path and the points are "unrolled" with it, height is only
  integrated where the raw track is clearly sloped (stairs), and sub-models that
  share no images are registered where their camera tracks interleave in time
  (2D rigid + height offset, snapped to the building grid). Writes
  `work/rectified/`, which `gsu geometry` uses automatically; the raw COLMAP model
  stays untouched for splat training.
* **Corridors**: per level, points in a wall band (0.35–2.1 m above the floor) go into
  a 0.15 m occupancy grid that counts *how many height bins* a cell covers, so
  handrails, furniture and stray points do not stop a ray while walls do. Left/right
  rays from every path sample give the corridor half-widths; long gaps where no wall
  is hit become **openings** (doors, side corridors). The path is snapped to the two
  dominant directions (`--no-manhattan` to keep raw angles) and each straight piece
  gets one width per side.
* **Stairs and halls**: a flight whose walked path encloses an area is an
  **open-well stair** (flights wrapping around a central opening): it is drawn with
  the flights where you walked, the void as a thin wireframe box between the two
  floors and a gallery ring around it on each level (`--stairs-style atrium|path|simple`
  overrides the detection). Landings are merged into their flight. A wide, wandering
  stretch (an entrance hall, the room at the end of the walk) is drawn as a
  rectangle instead of a corridor ribbon.
* **Diagram**: floor slabs per run, stairs as sloped slabs with treads, the walk path
  (not on the flights), labels from `--labels`, from `poses.json` (your website spots
  double as sign destinations) plus Start/End and the stairs; walls are computed but
  only drawn with `--walls` (then extruded to `--wall-height` in the 3D files and cut
  away to `--cutaway` in the isometric).

Sub-models: if the walk breaks (blur, a fast turn) COLMAP returns several models.
`gsu sfm` keeps the largest as `colmap/sparse/0` and the rest in
`colmap/sparse/models/`; `gsu rectify` stitches them for the diagram, and
`gsu use-model --model colmap/sparse/models/2` switches the workspace model (for
training) by hand. The GLOMAP mapper (default for video) is fast and keeps weak
chains together, but drifts in scale on corridors; that is what `rectify` fixes.

## Idealised schematic

```bash
gsu idealize --work work --title "Building A"            # -> work/geometry/ideal/
```

Reads the measured `diagram.json` and redraws the same elements as a clean schematic
on a 0.5 m grid: halls as rectangles, the corridor as a straight slab the same width as
the stair landings (`--corridor-width` to change it), each open-well stair as a square
well (`--void-side`, default the measured median) with flights of `--flight-width` around
it, quarter landings, real steps, a start/end hall drawn as a slab as deep as its stair
block, reaching one landing square into it so the floor ends exactly where the steps start
(`--foyer beside` for the measured rectangle), the route climbing the flights, numbered
markers with a short key instead of text on the drawing (`--labels-inline` to revert). On
the stairs only the walked surfaces are drawn: flights, landings and the stretch of
gallery the route crosses (`--gallery upper|both|none` and `--void-lines` add the full
gallery ring or the well's wireframe). Outputs `plan.svg`, `plan_L*.svg`, `iso.svg`,
`iso_noroute.svg`, `iso_exploded.svg`, `diagram.glb/.obj`, the schematic `diagram.json`
and one isometric per colour theme in `styles/` (`light`, `dark` = off-black with white
lines, `blueprint`, `mono` line art, `paper`, `slate`; `--theme` picks the main one,
`--iso-azimuth` / `--iso-elevation` turn the view).

Geometry knobs, one value per stair in route order where there is a list:
`--stair-sense ccw,cw` (turn direction), `--stair-flights 3,4` (flights per storey;
default from the storey height), `--exit-after 0,4` (the route leaves the stair after
that many flights; 0 = all), `--continue 0,1` (extra flights drawn beyond the exit when
the stair carries on), `--wells measured` (each stair its own well size), `--mid-steps
0.55:4` (a short flight inside the corridor), `--entry-steps 8` (a lower entrance foyer
with a straight flight up to the stair), `--corridor-width`, `--flight-width`. A corridor
meeting a stair continues the walkway along the side of the well it leaves from and runs
straight onto the first flight of the stair it reaches (`--landing straight`, the
default); `--landing inline` instead orders the flights so the last one itself points
down the corridor; `--landing L` meets the near corner with the landing running the full
side. A hall entered from a stair sits flush with the block at the exit
landing, on a side the stair was not entered from, so you step straight off the stair
into it; if the exit is between storeys the hall is drawn at that landing level. The
Start/End point of a base slab sits in line with the foot of the first flight. The sample
sign was made with

```bash
gsu idealize --theme mono --title "Route to G29" --end-label G29 \
    --stair-sense ccw,ccw --stair-flights 3,4 --exit-after 0,4 \
    --iso-azimuth 120 --corridor-max 9 --break-gap 0
```

`--corridor-max 9` draws a longer corridor at 9 m, sliding everything beyond it closer;
with a `--break-gap` above zero the shortened corridor gets a zig-zag break mark and a
gap in slab and route. `--shaft-lines up,both` draws dashed verticals at the four corners
of each stair block (the stairwell), running `--shaft-ext` metres past the stair's top
("up"), bottom ("down") or both.

Drawing rules: touching coplanar surfaces (slabs, landings, the top tread of a flight)
are merged into one outline; the surfaces in a level are ordered for drawing by
separating planes (exact for the axis-aligned boxes used here), so touching steps,
landings and route never paint over each other; the route is drawn as individual dashes
fitted to each straight run (a run starts and ends with a full dash, so corners stay
solid) and disappears where it passes under a higher surface. Edges of the stairs that
lie behind or beneath another surface are drawn as light dashed lines, the architect's
hidden-line convention (`--hidden-lines stairs|all|none`); visibility comes from casting
a ray from points along each edge towards the viewer through every box. A start/end foyer
drawn as the slab under its stair extends `--foyer-ext` metres beyond the block (7 m;
0 = as far as the measured hall).

`--morse "TEXT"` lays the text in Morse along the route instead of the plain dashes
(dot 1 unit, dash 3, gaps 1 within a letter, 3 between letters, 7 between words;
`--morse-unit` sets the dot length on the page, 4 px at 36 px/m, 0 = stretch the message
over the whole route). Marks are placed only
on the visible stretches of the route, so the message pauses where the route passes under
a slab and carries on after it. The log reports the route's visible length, the units
available and roughly how many characters of English that is; `--morse ""` just reports.
A hall entered from a stair gets the same slab as a start foyer.
`python scripts/make_variants.py --work work --out variants/` renders a labelled contact
sheet of angle and geometry variants. This is the drawing to hand to the sign designer;
the measured one is the evidence behind it.

## Example: the sample walk-through

`examples/walkthrough/` holds the diagrams produced from `video/input.mp4` (848x480
handheld phone video, 145 s): entrance hall, open stair up, a 23 m corridor, open
stair down, end hall. Scale comes from the 1.5 m camera-height assumption; corridor
lengths follow the drift-corrected walk. `examples/walkthrough/ideal/` is the
idealised schematic of the same walk.

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
