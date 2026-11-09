# Route to G29 — variants and critique

All in the locked style (black line on white, numbered markers, key below).
`contact_sheet.png` shows them side by side; each `*.svg` is editable (layers per level,
stairs, path, labels, key).

## Angles
| file | note |
|---|---|
| a1_az-30 | wide landscape; corridor foreshortened, both halls read well |
| a2_az-60_base | current default; corridor diagonal, start/end markers clear of the galleries |
| a3_az-90 | looking along the corridor; stairs stack behind each other — reject |
| a4_az120_other_side | seen from the far side; the route disappears behind the galleries — reject |
| a5_el25_low | low eye point, more "model on a table"; flights read best here |
| a6_el48_high | high eye point, almost a plan; route and halls clearest, stairs flatter |
| a7_exploded | levels pulled apart 3 m; the one that explains "up, across, down" fastest |

## Geometry
| file | note |
|---|---|
| g1_wells_measured | each stair its own well: the second one is the narrow dog-leg the video shows |
| g2_two_flights_dogleg | two flights per storey + half landing, measured wells: closest to the real stairs |
| g3_four_flights | spiral of four: busier, no gain |
| g4_gallery_both_levels | lower gallery rings duplicate the base slabs: clutter |
| g5_flights_only | no gallery/void: cleanest stairs but the corridor floats next to them |
| g6_corridor_1.6m / g7_corridor_2.6m | width is an assumption; 2.0 m sits between them |
| g8_mid_corridor_4_steps | the four steps up half-way along the corridor (66 s in the video) |
| g9_as_filmed_entry8_mid4 | entrance flight (8 steps, 14 s) + corridor steps + measured wells: most faithful |
| g10_foyer_beside | halls as separate rooms beside the stairs: the earlier, weaker reading |
| g11_wells_4.5m | bigger wells: more legible stairs, less true to the second one |
| g12_no_route | structure only, for a version where the route is added by hand |

## Critique of the current design
* **Hierarchy works, weight does not**: one line weight for everything. Silhouettes
  (slab edges) should be heavier than internal edges (step risers, void box), the route a
  third weight. The SVG layers allow this in Illustrator; the renderer could learn it.
* **Two things are inventions**, say so on the sign or hide them: the gallery ring on the
  upper level (the walk never traced it) and three flights per storey. The second stair
  is really a dog-leg (`g2`); the first is a grand open-well stair with half landings and
  three short steps at the top, which no variant shows.
* **Missing cues that matter for wayfinding**: the four steps up in the corridor (`g8`)
  and the short flight at the entrance (`g9`). Both were visible in the footage and
  invisible to the reconstruction (the corridor section of the SfM had collapsed in
  scale). They are now manual flags; measuring them needs better footage or a splat.
* **The route detours** at the top of the first stair (around two sides of the gallery)
  because the drawn stair ends on the far corner. A dog-leg (`g2`) or a stair that starts
  one side later would land the route at the corridor door directly.
* **Scale is a prior, not a measurement**: lengths follow the 1.5 m camera-height
  assumption and the corridor width is a default. The 23 m corridor and the 4.3 m storey
  are consistent with the walking time and a building of this age, no more.
* **Markers 2 and 3 sit in the wells**; placing them at the foot of the first flight
  reads more like "go up here". The key could also carry floor names (Ground / First).
* **What already works**: the base-slab foyers, the single corridor slab, the void
  wireframes giving depth, the key off the drawing, and the exploded variant as the
  teaching view.

## Recommendation
`g2_two_flights_dogleg` geometry with the `a5` or `a6` eye point and, if the building has
them, `--mid-steps 0.55:4 --entry-steps 8`; use `a7_exploded` where the sign has room
for a second panel.
