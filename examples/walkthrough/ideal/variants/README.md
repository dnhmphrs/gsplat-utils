# Route to G29 — variants and critique

All in the locked style (black line on white, numbered markers, five-step key below;
hidden edges of the stairs as light dashed lines) and the locked geometry: both stairs anticlockwise,
three flights up (south side, back, north side) with the walkway along the west side of
the well running straight on as the bridge onto the first flight of the second stair, a
full turn of four flights down, then straight off the last step into a foyer the same
size as the first, each 5 m long and meeting the first or last step. The bridge is drawn
at 11 m (it is 19 m) without a break mark, and the sign's view is azimuth 120 at a 35°
eye point, from behind the Start: at 25° the second and third flights of the first stair
fall on the same patch of page and their routes cross. `contact_sheet.png`
shows everything side by side, `contact_sheet_angles.png` the front angles and
`contact_sheet_behind.png` the views from the Start side, larger; each `*.svg` is editable
(layers per level, stairs, hidden lines, path, labels, key).

Made with `python scripts/make_variants.py --work work --out examples/walkthrough/ideal/variants`.

## Angles (eye point 35° unless noted)
| file | note |
|---|---|
| a1_az-45_isometric | classic 45° isometric; the corridor runs at 30° and both stairs show the same two flights foreshortened |
| a2_az-60_front | the earlier front view; corridor diagonal, the two foreshortened flights of the second stair at the back |
| a3_az-70 | less slant on the x axis: the fanned treads of the third flight tighten up |
| a4_az-75_second_stair_reads | the exit flight is seen in profile with its risers, the route loops visibly round the well; the corridor is steeper on the page |
| a5_az-90_frontal | looking straight along the corridor: a frontal axonometric, no depth on the halls — reject |
| a6_el30 | higher eye point; treads show more of their depth everywhere, the drawing is less "model on a table" |
| a7_exploded | levels pulled apart 3 m; still the fastest way to explain "up, across, down" |
| a8_el35_true_isometric | the textbook 35.26° isometric at 45°; every axis equally foreshortened, flatter stairs |
| a9_az-30_wide | wide landscape; the bridge nearly horizontal, both stairs seen side-on |
| a11_el20_lower | lower still; most "model on a table", the fewest visible treads |

## From the Start side (eye point 35° unless noted)
Seen the way a reader standing at the sign would see the building: Start nearest, the
route running away up the page to G29. `contact_sheet_behind.png` shows them together.
| file | note |
|---|---|
| b1_az90_from_start | looking straight down the bridge from the first foyer; the bridge is vertical on the page, the two stairs stack above each other |
| b2_az105 | a little to the left of the bridge; the first stair opens up, the second still partly behind the bridge |
| b3_az120_sign | **the sign**: the natural "over the shoulder" view, foyer in front, both wells visible, G29 at the top right |
| b4_az135 | the mirror of the classic isometric from behind; most even, slightly flatter stairs |
| b5_az150 | wide and low; the bridge nearly horizontal across the page |
| b6_az75_over_the_bridge | from the right of the bridge; the second stair hides behind the first |
| b7_az120_el25 / b8_az120_el30 | the sign's view from lower down; at 25° the back flights of the first stair overlap |
| b9_az105_el25 | b2 from lower down |

## Geometry
| file | note |
|---|---|
| g1_wells_measured | each stair its own well: the second well is the narrow 1.5 × 3 m one the reconstruction found, which makes its short flights steep |
| g2_exit_one_flight_down | the earlier reading of the footage: off after one flight into a room at the landing level, with `--continue` drawing the stair carrying on three more |
| g3_corridor_2.0m | a wider corridor than the landings; the T-junction at the first stair then needs a step in the outline |
| g4_flights_1.5m | wider flights and landings (corridor follows); bigger stairs, less corridor |
| g5_mid_corridor_4_steps | the four steps up half-way along the corridor (66 s in the video) |
| g6_entry_8_steps | the short entrance flight before the first stair (14 s in the video) |
| g7_wells_4.5m | bigger wells: more legible stairs, less true |
| g8_no_route | structure only, for a version where the route is added by hand |
| g9_foyer_beside | halls as separate rooms beside the stairs: the earlier, weaker reading |
| g10_landing_L | the earlier reading of the bridge: it meets the near corner of each stair and the top landing runs the full side as an L |
| g13_landing_inline | flights reordered so the last one of the first stair points straight down the bridge (back, north side, west side) |
| g14_solid_two_flights | the start foyer under the first stair, its first two flights a solid mass down to the floor, three treads hinting the second stair carries on |
| g15_all_flights_solid | all three flights of the first stair solid to the floor; the back flight becomes a wall that hides the well |
| g11_no_hidden_lines | hidden edges left out: the two foreshortened flights of the second stair fall back to a fan of plates |
| g12_hidden_lines_all | hidden edges of the slabs too (under the bridge, under the stairs): more information, more clutter |

## Critique of the current design
* **A full-turn stair always has two flights that foreshorten.** At a 25° eye point a
  flight descending away from the viewer shows only slivers of its treads (the risers face
  away). At −60° those are the third and fourth flights of the second stair, drawn behind
  the two that read clearly. The hidden-line dashes now show the covered tread edges, so
  the flights read as stairs rather than a fan of plates (`g11` shows the difference);
  `a4` (−75°) or `a6` (30°) still trade composition for a clearer second stair.
* **One line weight for everything.** Silhouettes (slab edges) should be heavier than
  internal edges (risers), the route a third weight. The layers in the SVG allow this in
  Illustrator; the renderer could learn it.
* **The antechamber is a box.** Its 9 × 4 m comes from the measured hall; where the G29
  door actually is was not filmed, so the marker sits at the end of a straight run off the
  landing. The stair carrying on below it is not drawn (`--continue 0,1` adds a flight).
* **The route from Start passes under the end of the bridge.** The bridge continues the
  last flight, so its first landing sits above the ground-floor corner the Start route
  crosses on its way to the first step; true to an open-well stair, hidden for a metre.
* **Scale is a prior, not a measurement**: lengths follow the 1.5 m camera-height
  assumption, the 3.5 m wells are the median of two very different measured wells, and the
  storey height (4.25 m) comes from the drift-corrected climb.
* **Missing cues that matter for wayfinding**: the four steps in the corridor (`g5`) and
  the entrance flight (`g6`) are in the footage but not in the reconstruction (the corridor
  section had collapsed in scale); they stay manual flags until a splat or better footage.
* **What works**: the base-slab foyer under the first stair, the corridor at landing
  width so the top landings are clean Ls, the stair carrying on below the antechamber,
  only walked surfaces drawn, markers off the drawing in a key.

## Morse option
`--morse "TEXT"` turns the route's dashes into the text in Morse (`--morse-unit 0` stretches
it over the whole route; marks pause where the route is hidden). For this drawing the route
is about 1850 px on the page at 36 px/m with about 1600 px visible: 400 units at a 4 px dot,
roughly 38 characters of English with spaces, 50 at a 3 px dot. The sign itself uses plain
dashes.

## Recommendation
`b3_az120_sign` for the sign, as chosen. `b7`/`b8` if the treads need to read more clearly,
`a7_exploded` where there is room for a second panel.
