# Route to G29 — variants and critique

All in the locked style (black line on white, numbered markers, key below) and the locked
geometry: both stairs anticlockwise, three flights up, a full turn of four flights down,
then straight off the landing into the antechamber while the stair carries on down one
more flight. `contact_sheet.png` shows them side by side; each `*.svg` is editable
(layers per level, stairs, path, labels, key).

Made with `python scripts/make_variants.py --work work --out examples/walkthrough/ideal/variants`.

## Angles (eye point 25° unless noted)
| file | note |
|---|---|
| a1_az-45_isometric | classic 45° isometric; the corridor runs at 30° and both stairs show the same two flights foreshortened |
| a2_az-60_base | **the sign**; corridor diagonal, start/end markers clear, the two foreshortened flights of the second stair sit at the back behind the two that read |
| a3_az-70 | less slant on the x axis: the fanned treads of the third flight tighten up |
| a4_az-75_second_stair_reads | the exit flight is seen in profile with its risers, the route loops visibly round the well; the corridor is steeper on the page |
| a5_az-90_frontal | looking straight along the corridor: a frontal axonometric, no depth on the halls — reject |
| a6_el30 | higher eye point; treads show more of their depth everywhere, the drawing is less "model on a table" |
| a7_exploded | levels pulled apart 3 m; still the fastest way to explain "up, across, down" |

## Geometry
| file | note |
|---|---|
| g1_wells_measured | each stair its own well: the second well is the narrow 1.5 × 3 m one the reconstruction found, which makes its short flights steep |
| g2_exit_one_flight_down | the earlier reading of the footage: off after one flight into a room at the landing level, stair continuing three more |
| g3_corridor_2.0m | a wider corridor than the landings; the T-junction at the first stair then needs a step in the outline |
| g4_flights_1.5m | wider flights and landings (corridor follows); bigger stairs, less corridor |
| g5_mid_corridor_4_steps | the four steps up half-way along the corridor (66 s in the video) |
| g6_entry_8_steps | the short entrance flight before the first stair (14 s in the video) |
| g7_wells_4.5m | bigger wells: more legible stairs, less true |
| g8_no_route | structure only, for a version where the route is added by hand |
| g9_foyer_beside | halls as separate rooms beside the stairs: the earlier, weaker reading |

## Critique of the current design
* **A full-turn stair always has two flights that foreshorten.** At a 25° eye point a
  flight descending away from the viewer shows only slivers of its treads (the risers face
  away), so it reads as a fan of plates. At −60° those are the third and fourth flights of
  the second stair, drawn behind the two that read clearly; the route on them is hidden
  where the nearer treads cover it, which is correct but busy. `a4` (−75°) or `a6` (30°)
  trade composition for a clearer second stair.
* **One line weight for everything.** Silhouettes (slab edges) should be heavier than
  internal edges (risers), the route a third weight. The layers in the SVG allow this in
  Illustrator; the renderer could learn it.
* **The antechamber is a box.** Its 9 × 4 m comes from the measured hall; where the G29
  door actually is was not filmed, so the marker sits at the end of a straight run off the
  landing. One extra flight stands for "the stair carries on down"; how far it goes is
  unknown.
* **Scale is a prior, not a measurement**: lengths follow the 1.5 m camera-height
  assumption, the 3.5 m wells are the median of two very different measured wells, and the
  storey height (4.25 m) comes from the drift-corrected climb.
* **Missing cues that matter for wayfinding**: the four steps in the corridor (`g5`) and
  the entrance flight (`g6`) are in the footage but not in the reconstruction (the corridor
  section had collapsed in scale); they stay manual flags until a splat or better footage.
* **What works**: the base-slab foyer under the first stair, the corridor at landing
  width so the top landings are clean Ls, the stair carrying on below the antechamber,
  only walked surfaces drawn, markers off the drawing in a key.

## Recommendation
`a2_az-60_base` for the sign. If the second staircase must be read tread by tread, use
`a4_az-75_second_stair_reads`. Add `--mid-steps 0.55:4 --entry-steps 8` if the sign should
warn about the short flights, and `a7_exploded` where there is room for a second panel.
