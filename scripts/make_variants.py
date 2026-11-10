"""Render a family of idealised variants (angles, geometries) in one style and
compose a labelled contact sheet.

    python scripts/make_variants.py --work work --out examples/walkthrough/ideal/variants \
        --theme mono --title "Route to G29" --end-label G29

PNG rendering uses `node scripts/svg2png.cjs` (Playwright) when available.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from dataclasses import replace
from pathlib import Path

import numpy as np

from gsplat_utils.geometry.idealize import IdealConfig, idealize
from gsplat_utils.geometry.svg import AxoView, write_axo_svg

HERE = Path(__file__).resolve().parent

VARIANTS: list[tuple[str, dict, dict]] = [
    # name, IdealConfig overrides, view overrides
    ("a1_az-45_isometric", {}, {"azimuth_deg": -45}),
    ("a2_az-60_front", {}, {"azimuth_deg": -60}),
    ("a3_az-70", {}, {"azimuth_deg": -70}),
    ("a4_az-75_second_stair_reads", {}, {"azimuth_deg": -75}),
    ("a5_az-90_frontal", {}, {"azimuth_deg": -90}),
    ("a6_el30", {}, {"azimuth_deg": -60, "elevation_deg": 30}),
    ("a7_exploded", {}, {"azimuth_deg": -60, "explode_m": 3.0}),
    ("a8_el35_true_isometric", {}, {"azimuth_deg": -45, "elevation_deg": 35.264}),
    ("a9_az-30_wide", {}, {"azimuth_deg": -30}),
    ("a11_el20_lower", {}, {"azimuth_deg": -60, "elevation_deg": 20}),
    # from the Start side, looking along the route the way a reader at the sign would
    ("b1_az90_from_start", {}, {"azimuth_deg": 90}),
    ("b2_az105", {}, {"azimuth_deg": 105}),
    ("b3_az120_sign", {}, {"azimuth_deg": 120}),
    ("b4_az135", {}, {"azimuth_deg": 135}),
    ("b5_az150", {}, {"azimuth_deg": 150}),
    ("b6_az75_over_the_bridge", {}, {"azimuth_deg": 75}),
    ("b7_az120_el25", {}, {"azimuth_deg": 120, "elevation_deg": 25}),
    ("b8_az120_el30", {}, {"azimuth_deg": 120, "elevation_deg": 30}),
    ("b9_az105_el25", {}, {"azimuth_deg": 105, "elevation_deg": 25}),
    ("g1_wells_measured", {"void_mode": "measured"}, {}),
    ("g2_exit_one_flight_down", {"exit_after": [0, 1], "continue_flights": [0, 3]}, {}),
    ("g3_corridor_2.0m", {"corridor_width": 2.0}, {}),
    ("g4_flights_1.5m", {"flight_width": 1.5, "corridor_width": 1.5}, {}),
    ("g5_mid_corridor_4_steps", {"mid_steps": [(0.55, 4)]}, {}),
    ("g6_entry_8_steps", {"entry_steps": 8}, {}),
    ("g7_wells_4.5m", {"void_side": 4.5}, {}),
    ("g8_no_route", {"show_path": False}, {}),
    ("g9_foyer_beside", {"foyer": "beside"}, {}),
    ("g10_landing_L", {"top_landing": "L"}, {}),
    ("g13_landing_inline", {"top_landing": "inline"}, {}),
    ("g14_solid_two_flights", {"foyer": "under", "solid_stairs": [2, 0], "continue_steps": [0, 3]}, {}),
    ("g15_all_flights_solid", {"foyer": "under", "solid_stairs": [3, 0]}, {}),
    ("g11_no_hidden_lines", {"hidden_lines": "none"}, {}),
    ("g12_hidden_lines_all", {"hidden_lines": "all"}, {}),
    # ways of showing the door between the two buildings
    ("t1_door_symbol", {"threshold_style": "door"}, {}),
    ("t2_door_frame", {"threshold_style": "portal"}, {}),
    ("t3_tinted_building", {"threshold_style": "zones"}, {}),
    ("t4_boundary_line", {"threshold_style": "line"}, {}),
    ("t5_door_symbol_no_names", {"threshold_style": "door", "building_names": False}, {}),
    ("t6_no_door", {"threshold": None}, {}),
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", type=Path, default=Path("work"))
    ap.add_argument("--diagram", type=Path, default=None)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--theme", default="mono")
    ap.add_argument("--title", default="Route to G29")
    ap.add_argument("--end-label", default="G29")
    ap.add_argument("--only", nargs="*", default=None, help="variant names to render")
    a = ap.parse_args()
    src = a.diagram or (a.work / "geometry" / "diagram.json")
    d = json.loads(Path(src).read_text())
    a.out.mkdir(parents=True, exist_ok=True)
    # the locked design: both stairs anticlockwise, three flights up (the first at the back of
    # the foyer, the last running into the bridge), the bridge one straight line onto the
    # second stair, a full turn of four flights down, straight off into the antechamber
    base = IdealConfig(theme=a.theme, title=a.title, end_label=a.end_label, all_themes=False,
                       stair_sense=["ccw", "ccw"], stair_flights=[3, 4], exit_after=[0, 4],
                       iso_azimuth=120, iso_elevation=35, corridor_max=11.0, break_gap=0.0,
                       threshold="Medical Science Building|Anatomy Building", threshold_style="zones")
    made = []
    for name, cfg_over, view_over in VARIANTS:
        if a.only and name not in a.only:
            continue
        cfg = replace(base, **cfg_over)
        ideal = idealize(d, cfg)
        view = AxoView(cfg.iso_azimuth, cfg.iso_elevation, cfg.explode)
        for k, v in view_over.items():
            setattr(view, k, v)
        svg = a.out / f"{name}.svg"
        write_axo_svg(svg, ideal, view, title=f"{a.title}  ·  {name.split('_', 1)[1].replace('_', ' ')}",
                      show_path=cfg.show_path, theme=a.theme, label_style=cfg.label_style, slab=cfg.slab,
                      hidden_lines=cfg.hidden_lines)
        made.append(svg)
        print("wrote", svg)
    # PNG renders + contact sheet (optional)
    node = shutil.which("node")
    if node and made:
        try:
            subprocess.run([node, str(HERE / "svg2png.cjs"), *map(str, made)], check=True, capture_output=True)
        except subprocess.CalledProcessError as exc:
            print("png rendering failed (Playwright missing?):", exc.stderr.decode()[-300:])
            return
        try:
            from PIL import Image, ImageDraw
        except ImportError:
            return
        def sheet_of(svgs, name, cols=3, cw=620, ch=560):
            ims = [(s, Image.open(s.with_suffix(".png"))) for s in svgs if s.with_suffix(".png").exists()]
            if not ims:
                return
            rows = (len(ims) + cols - 1) // cols
            sheet = Image.new("RGB", (cols * cw, rows * ch), "white")
            dr = ImageDraw.Draw(sheet)
            for k, (svg, im) in enumerate(ims):
                im = im.copy(); im.thumbnail((cw - 20, ch - 40))
                x, y = (k % cols) * cw + 10, (k // cols) * ch + 30
                sheet.paste(im, (x, y))
                dr.text(((k % cols) * cw + 12, (k // cols) * ch + 8), svg.stem, fill="black")
                dr.rectangle([(k % cols) * cw, (k // cols) * ch, (k % cols + 1) * cw - 1, (k // cols + 1) * ch - 1], outline="#dddddd")
            sheet.save(a.out / name)
            print("contact sheet:", a.out / name, sheet.size)
        sheet_of(made, "contact_sheet.png")
        sheet_of([s for s in made if s.stem.startswith("a")], "contact_sheet_angles.png", cols=2, cw=900, ch=720)
        sheet_of([s for s in made if s.stem.startswith("b")], "contact_sheet_behind.png", cols=2, cw=900, ch=720)
        sheet_of([s for s in made if s.stem.startswith("t")], "contact_sheet_threshold.png", cols=2, cw=900, ch=760)


if __name__ == "__main__":
    main()
