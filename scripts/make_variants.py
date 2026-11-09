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
    ("a1_az-30", {}, {"azimuth_deg": -30}),
    ("a2_az-60_base", {}, {"azimuth_deg": -60}),
    ("a3_az-90", {}, {"azimuth_deg": -90}),
    ("a4_az120_other_side", {}, {"azimuth_deg": 120}),
    ("a5_el25_low", {}, {"azimuth_deg": -60, "elevation_deg": 25}),
    ("a6_el48_high", {}, {"azimuth_deg": -60, "elevation_deg": 48}),
    ("a7_exploded", {}, {"azimuth_deg": -60, "explode_m": 3.0}),
    ("g1_wells_measured", {"void_mode": "measured"}, {}),
    ("g2_two_flights_dogleg", {"flights": 2, "void_mode": "measured"}, {}),
    ("g3_four_flights", {"flights": 4}, {}),
    ("g4_gallery_both_levels", {"gallery": "both"}, {}),
    ("g5_flights_only", {"gallery": "none"}, {}),
    ("g5b_full_gallery_void", {"gallery": "upper", "void_lines": True}, {}),
    ("g6_corridor_1.6m", {"corridor_width": 1.6}, {}),
    ("g7_corridor_2.6m_flights_1.5m", {"corridor_width": 2.6, "flight_width": 1.5}, {}),
    ("g8_mid_corridor_4_steps", {"mid_steps": [(0.55, 4)]}, {}),
    ("g9_as_filmed_entry8_mid4", {"entry_steps": 8, "mid_steps": [(0.55, 4)], "void_mode": "measured"}, {}),
    ("g10_foyer_beside", {"foyer": "beside"}, {}),
    ("g11_wells_4.5m", {"void_side": 4.5}, {}),
    ("g12_no_route", {"show_path": False}, {}),
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
    base = IdealConfig(theme=a.theme, title=a.title, end_label=a.end_label, all_themes=False)
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
                      show_path=cfg.show_path, theme=a.theme, label_style=cfg.label_style, slab=cfg.slab)
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
        ims = [Image.open(s.with_suffix(".png")) for s in made if s.with_suffix(".png").exists()]
        if not ims:
            return
        cols = 3
        cw, ch = 620, 560
        rows = (len(ims) + cols - 1) // cols
        sheet = Image.new("RGB", (cols * cw, rows * ch), "white")
        dr = ImageDraw.Draw(sheet)
        for k, (svg, im) in enumerate(zip(made, ims)):
            im = im.copy(); im.thumbnail((cw - 20, ch - 40))
            x, y = (k % cols) * cw + 10, (k // cols) * ch + 30
            sheet.paste(im, (x, y))
            dr.text(((k % cols) * cw + 12, (k // cols) * ch + 8), svg.stem, fill="black")
            dr.rectangle([(k % cols) * cw, (k // cols) * ch, (k % cols + 1) * cw - 1, (k // cols + 1) * ch - 1], outline="#dddddd")
        sheet.save(a.out / "contact_sheet.png")
        print("contact sheet:", a.out / "contact_sheet.png", sheet.size)


if __name__ == "__main__":
    main()
