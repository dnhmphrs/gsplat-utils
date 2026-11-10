"""SVG output: per-level plans and an axonometric 'sign' view. No dependencies."""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..common import log
from .rectunion import is_axis_rect, loop_is_ccw, rect_of, union_loops

MORSE = {"A": ".-", "B": "-...", "C": "-.-.", "D": "-..", "E": ".", "F": "..-.", "G": "--.", "H": "....", "I": "..",
         "J": ".---", "K": "-.-", "L": ".-..", "M": "--", "N": "-.", "O": "---", "P": ".--.", "Q": "--.-", "R": ".-.",
         "S": "...", "T": "-", "U": "..-", "V": "...-", "W": ".--", "X": "-..-", "Y": "-.--", "Z": "--..",
         "0": "-----", "1": ".----", "2": "..---", "3": "...--", "4": "....-", "5": ".....", "6": "-....", "7": "--...",
         "8": "---..", "9": "----.", ".": ".-.-.-", ",": "--..--", "?": "..--..", "'": ".----.", "!": "-.-.--",
         "/": "-..-.", "(": "-.--.", ")": "-.--.-", "&": ".-...", ":": "---...", ";": "-.-.-.", "=": "-...-",
         "+": ".-.-.", "-": "-....-", "_": "..--.-", '"': ".-..-.", "$": "...-..-", "@": ".--.-.",
         "Ä": ".-.-", "Ö": "---.", "Ü": "..--", "ß": "...--..", "É": "..-..", "Ñ": "--.--"}


def morse_symbols(text: str) -> list[tuple[bool, int]]:
    """Text -> [(mark?, units)] with standard timing: dot 1, dash 3, gap 1 inside a
    character, 3 between characters, 7 between words. Unknown characters are skipped."""
    out: list[tuple[bool, int]] = []
    words = [w for w in text.upper().split() if any(c in MORSE for c in w)]
    for wi, word in enumerate(words):
        chars = [c for c in word if c in MORSE]
        for ci, ch in enumerate(chars):
            code = MORSE[ch]
            for ei, el in enumerate(code):
                out.append((True, 1 if el == "." else 3))
                if ei < len(code) - 1:
                    out.append((False, 1))
            if ci < len(chars) - 1:
                out.append((False, 3))
        if wi < len(words) - 1:
            out.append((False, 7))
    return out


def morse_units(text: str) -> int:
    return sum(n for _, n in morse_symbols(text))

PALETTE = {
    "floor": "#e9e6df", "floor_stroke": "#b9b4a8", "wall": "#2b2b2b", "opening": "#9a9a9a",
    "stairs": "#d9d3c4", "path": "#d9482b", "label": "#111111", "label_dot": "#d9482b",
    "grid": "#f0eee9", "text": "#333333", "muted": "#777777",
}


def _fmt(v: float) -> str:
    return f"{v:.2f}".rstrip("0").rstrip(".") if abs(v) > 1e-9 else "0"


def _poly(pts2: np.ndarray, **attrs) -> str:
    d = " ".join(f"{_fmt(x)},{_fmt(y)}" for x, y in pts2)
    a = " ".join(f'{k.replace("_", "-")}="{v}"' for k, v in attrs.items())
    return f'<polygon points="{d}" {a}/>'


def _polyline(pts2: np.ndarray, **attrs) -> str:
    d = " ".join(f"{_fmt(x)},{_fmt(y)}" for x, y in pts2)
    a = " ".join(f'{k.replace("_", "-")}="{v}"' for k, v in attrs.items())
    return f'<polyline points="{d}" {a}/>'


def _line(a, b, **attrs) -> str:
    at = " ".join(f'{k.replace("_", "-")}="{v}"' for k, v in attrs.items())
    return f'<line x1="{_fmt(a[0])}" y1="{_fmt(a[1])}" x2="{_fmt(b[0])}" y2="{_fmt(b[1])}" {at}/>'


def _text(p, s: str, size: float, **attrs) -> str:
    at = " ".join(f'{k.replace("_", "-")}="{v}"' for k, v in attrs.items())
    s = s.replace("&", "&amp;").replace("<", "&lt;")
    return f'<text x="{_fmt(p[0])}" y="{_fmt(p[1])}" font-size="{_fmt(size)}" {at}>{s}</text>'


# ---------------------------------------------------------------------------
# plan view
# ---------------------------------------------------------------------------
def write_plan_svg(path, diagram: dict, level_id: int | None, bounds: tuple[float, float, float, float],
                   px_per_m: float = 40.0, margin_m: float = 1.5, show_path: bool = True, title: str | None = None,
                   draw_walls: bool = False) -> None:
    """Top-down plan. World x -> right, world y -> up. Draws runs of `level_id`
    (None = all levels, stairs always drawn)."""
    x0, y0, x1, y1 = bounds
    x0 -= margin_m; y0 -= margin_m; x1 += margin_m; y1 += margin_m
    W = (x1 - x0) * px_per_m; H = (y1 - y0) * px_per_m

    def P(xy):
        return ((xy[0] - x0) * px_per_m, (y1 - xy[1]) * px_per_m)

    out = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W:.0f}" height="{H:.0f}" viewBox="0 0 {W:.1f} {H:.1f}" '
           f'font-family="Helvetica, Arial, sans-serif">',
           f'<rect width="100%" height="100%" fill="white"/>']
    # 1 m grid
    gx = math.floor(x0)
    while gx <= x1:
        out.append(_line(P((gx, y0)), P((gx, y1)), stroke=PALETTE["grid"], stroke_width=1)); gx += 1
    gy = math.floor(y0)
    while gy <= y1:
        out.append(_line(P((x0, gy)), P((x1, gy)), stroke=PALETTE["grid"], stroke_width=1)); gy += 1

    runs = diagram["runs"]
    sel = [r for r in runs if level_id is None or r["kind"] == "stairs" or r["level"] == level_id]
    sel = [r for r in sel if r["kind"] != "stairs"] + [r for r in sel if r["kind"] == "stairs"]   # floors under flights
    # floors: coplanar axis-aligned flats merged per height (no borders inside a continuous floor)
    merged: dict[float, list] = {}
    for r in sel:
        if r["kind"] == "mark":
            for pl in r.get("break_marks", []):
                out.append(_polyline(np.array([P(p[:2]) for p in pl]), fill="none", stroke=PALETTE["wall"], stroke_width=1.2,
                                     stroke_linejoin="round"))
            continue
        if r["kind"] == "stairs" and level_id is not None and level_id not in (r["level_from"], r["level_to"]):
            continue
        if r["kind"] == "stairs" and r.get("atrium"):   # gallery ring first, flights on top
            ring_mode = r["atrium"].get("ring_levels", "both")
            upper = max(r["level_from"], r["level_to"])
            if ring_mode in ("upper", "both") and (ring_mode == "both" or level_id is None or level_id == upper):
                for q in r["atrium"].get("ring", []):
                    out.append(_poly(np.array([P(p) for p in q]), fill="#f7f5f0", stroke="#8c8577", stroke_width=0.8))
        thick = r.get("quad_thickness") or [None] * len(r.get("quads") or [])
        rest = []
        for q, th in zip(r.get("quads") or [r["outline"]], thick if len(thick) == len(r.get("quads") or [r["outline"]]) else [None] * len(r.get("quads") or [r["outline"]])):
            qa = np.array(q, float)
            flat = qa.shape[1] == 3 and np.ptp(qa[:, 2]) <= 1e-6
            is_step = r["kind"] == "stairs" and r.get("steps") and th is not None and th < 0.1
            if flat and not is_step and is_axis_rect(qa):
                merged.setdefault(round(float(qa[0, 2]), 3), []).append(rect_of(qa))
            else:
                rest.append(q)
        fill = PALETTE["stairs"] if r["kind"] == "stairs" else PALETTE["floor"]
        for q in rest:
            pts = np.array([P(p) for p in q])
            out.append(_poly(pts, fill=fill, stroke="none" if draw_walls else PALETTE["wall"],
                             stroke_width=1.2, stroke_linejoin="round", fill_opacity=0.95))
        if r["kind"] == "stairs" and not r.get("steps"):
            out.extend(_tread_lines(r, P, px_per_m))
        if r["kind"] == "stairs" and r.get("atrium") and r["atrium"].get("void_lines", True):
            at = r["atrium"]
            c = np.array([P(p) for p in at["corners"]])
            out.append(_poly(c, fill="white", stroke="#4a4a4a", stroke_width=1.2, stroke_dasharray="5 3"))
            # diagonal cross = void (architectural convention)
            out.append(_line(c[0], c[2], stroke="#9a9a9a", stroke_width=0.8))
            out.append(_line(c[1], c[3], stroke="#9a9a9a", stroke_width=0.8))
    for z, rects in sorted(merged.items()):
        loops = union_loops(rects)
        d_parts = ["M " + " L ".join(f"{_fmt(x)},{_fmt(y)}" for x, y in np.array([P(p) for p in lp])) + " Z" for lp in loops]
        out.insert(2, f'<path d="{" ".join(d_parts)}" fill="{PALETTE["floor"]}" fill-rule="evenodd" '
                      f'stroke="{PALETTE["wall"]}" stroke-width="1.2" stroke-linejoin="round" fill-opacity="0.95"/>')
    # walls
    for r in (sel if draw_walls else []):
        if r["kind"] == "stairs":
            continue
        for w in r["walls"]:
            a, b = np.array(w["a"]), np.array(w["b"])
            for t0, t1 in w["solid"]:
                out.append(_line(P(a + (b - a) * t0), P(a + (b - a) * t1), stroke=PALETTE["wall"],
                                 stroke_width=max(2.0, 0.12 * px_per_m), stroke_linecap="square"))
            for t0, t1 in w["openings"]:
                out.append(_line(P(a + (b - a) * t0), P(a + (b - a) * t1), stroke=PALETTE["opening"],
                                 stroke_width=max(1.0, 0.04 * px_per_m), stroke_dasharray="4 3"))
    # path (not inside flights of stairs: the flights themselves show the route)
    if show_path:
        pts = np.array(diagram["path"]["points"])
        lv = diagram["path"].get("level")
        if level_id is not None:
            mask = np.array([l == level_id for l in lv]) if lv else np.ones(len(pts), bool)
        else:
            mask = np.array([l >= 0 for l in lv]) if lv else np.ones(len(pts), bool)
        # draw in contiguous chunks
        chunk = []
        for p, m in zip(pts, mask):
            if m:
                chunk.append(P(p))
            elif chunk:
                out.append(_polyline(np.array(chunk), fill="none", stroke=PALETTE["path"], stroke_width=2,
                                     stroke_dasharray="6 4", stroke_linejoin="round")); chunk = []
        if chunk:
            out.append(_polyline(np.array(chunk), fill="none", stroke=PALETTE["path"], stroke_width=2,
                                 stroke_dasharray="6 4", stroke_linejoin="round"))
    # labels
    fs = max(10.0, 0.35 * px_per_m)
    for lb in diagram.get("labels", []):
        if level_id is not None and lb.get("level", level_id) not in (level_id, -1):
            continue
        p = P(lb["xyz"][:2])
        if lb.get("kind") in ("stairs", "corridor"):
            out.append(_text((p[0] + fs * 0.3, p[1] + fs * 0.35), lb["text"], fs * 0.8, fill=PALETTE["muted"], font_weight="500"))
            continue
        out.append(f'<circle cx="{_fmt(p[0])}" cy="{_fmt(p[1])}" r="{_fmt(fs * 0.35)}" fill="{PALETTE["label_dot"]}"/>')
        out.append(_text((p[0] + fs * 0.6, p[1] + fs * 0.35), lb["text"], fs, fill=PALETTE["label"], font_weight="600"))
    # scale bar + title
    sb = P((x0 + 0.5, y0 + 0.6)); sb2 = P((x0 + 5.5, y0 + 0.6))
    out.append(_line(sb, sb2, stroke=PALETTE["text"], stroke_width=3))
    out.append(_text((sb[0], sb[1] - 6), "5 m", fs * 0.9, fill=PALETTE["text"]))
    name = title or (f"Level {level_id}" if level_id is not None else "All levels")
    out.append(_text((12, fs * 1.6), name, fs * 1.4, fill=PALETTE["text"], font_weight="700"))
    note = diagram.get("note", "")
    if note:
        out.append(_text((12, fs * 1.6 + fs * 1.2), note, fs * 0.8, fill=PALETTE["muted"]))
    out.append("</svg>")
    Path(path).write_text("\n".join(out), encoding="utf-8")


def _tread_lines(r: dict, P, px_per_m: float, every_m: float = 0.3) -> list[str]:
    out = []
    L = np.array(r["left_boundary"]); R = np.array(r["right_boundary"])
    for j in range(len(L) - 1):
        seg_len = np.linalg.norm(L[j + 1, :2] - L[j, :2])
        n = int(seg_len / every_m)
        for k in range(1, n):
            t = k / n
            a = L[j] + (L[j + 1] - L[j]) * t
            b = R[j] + (R[j + 1] - R[j]) * t
            out.append(_line(P(a[:2]), P(b[:2]), stroke="#8c8577", stroke_width=1))
    return out


def _emit_morse_route(faces, view, ppm, S, lw, runs_, text, unit_px, cam_h, lvl_off, off_for_z, layer_for,
                      stairs_layer_for_z, all_pts) -> bool:
    """Lay `text` in Morse along the route: marks are drawn only where the route is visible,
    and the message pauses while the route passes under a slab (hidden stretches carry no
    symbols). `unit_px` <= 0 stretches the message over the whole visible route. Logs the
    route's capacity; returns False (nothing drawn) when the text has no Morse characters."""
    step_px = 1.0
    samples = []                     # (run index, 3D point incl. raise/offset, layer, flag)
    for ri, (a, b, flag) in enumerate(runs_):
        oa = lvl_off(flag) if flag >= 0 else off_for_z(float(a[2]))
        ob = lvl_off(flag) if flag >= 0 else off_for_z(float(b[2]))
        a3 = np.array([a[0], a[1], a[2] + 0.06 + oa]); b3 = np.array([b[0], b[1], b[2] + 0.06 + ob])
        p2, _ = view.project(np.vstack([a3, b3])); all_pts.append(p2)
        L_px = float(np.linalg.norm((p2[1] - p2[0]) * ppm))
        n = max(1, int(np.ceil(L_px / step_px)))
        for k in range(n + 1):
            t = k / n
            samples.append((ri, a3 + (b3 - a3) * t, a[2] + (b[2] - a[2]) * t, flag))
    if len(samples) < 2:
        return False
    P = np.array([s_[1] for s_ in samples])
    vis = faces.visible(P)
    p2all, _ = view.project(P); p2all = p2all * ppm
    # arc length of each step (between consecutive samples of the same run), visible or not
    seg_len = np.zeros(len(samples)); seg_vis = np.zeros(len(samples), bool)
    for i in range(len(samples) - 1):
        if samples[i][0] == samples[i + 1][0]:
            seg_len[i] = float(np.linalg.norm(p2all[i + 1] - p2all[i]))
            seg_vis[i] = bool(vis[i] and vis[i + 1])
    vis_len = np.where(seg_vis, seg_len, 0.0)
    cum_vis = np.concatenate([[0.0], np.cumsum(vis_len)])          # visible arc at each sample
    total_vis = float(cum_vis[-1]); total_all = float(seg_len.sum())
    symbols = morse_symbols(text)
    n_units = sum(n for _, n in symbols)
    if unit_px <= 0:                                   # stretch the message over the visible route
        unit_px = total_vis / max(n_units, 1)
    cap_units = total_vis / unit_px
    log.info("route on the page: %.0f px, %.0f px visible; Morse unit %.1f px -> %d units available "
             "(about %d characters of English with spaces)%s", total_all, total_vis, unit_px, int(cap_units),
             int(cap_units / 10.5), f"; message needs {n_units} units ({len(''.join(text.split()))} characters)" if symbols else "")
    if not symbols:
        return False
    need = n_units * unit_px
    if need > total_vis + 1e-6:
        log.warning("Morse message does not fit on the visible route: %d units needed, %d available",
                    n_units, int(cap_units))
    elif need < 0.6 * total_vis:
        log.info("the message uses %.0f%% of the visible route; --morse-unit 0 stretches it to fill",
                 100 * need / total_vis)
    # place each mark over the visible arc interval [s0, s1)
    s = 0.0
    for mark, n in symbols:
        s0, s1 = s, s + n * unit_px
        s = s1
        if not mark:
            continue
        i0 = int(np.searchsorted(cum_vis, s0, side="right") - 1)
        i1 = int(np.searchsorted(cum_vis, s1, side="left"))
        i0 = max(0, min(i0, len(samples) - 1)); i1 = max(i0 + 1, min(i1, len(samples) - 1))
        # contiguous visible stretches within [i0, i1] of the same run -> one line each
        j = i0
        while j < i1:
            if not seg_vis[j]:
                j += 1; continue
            k = j
            while k + 1 < i1 and seg_vis[k + 1] and samples[k + 1][0] == samples[j][0]:
                k += 1
            pa, pb = p2all[j], p2all[k + 1]
            if j == i0 and cum_vis[j] < s0:              # trim the first step to the symbol start
                f = (s0 - cum_vis[j]) / max(seg_len[j], 1e-9); pa = p2all[j] + (p2all[j + 1] - p2all[j]) * f
            if k + 1 == i1 and cum_vis[k + 1] > s1:     # trim the last step to the symbol end
                f = (s1 - cum_vis[k]) / max(seg_len[k], 1e-9); pb = p2all[k] + (p2all[k + 1] - p2all[k]) * f
            mid = samples[j]
            ly = layer_for(mid[1][:2], float(mid[2]), mid[3])
            box = (float(min(P[j][0], P[k + 1][0])), float(min(P[j][1], P[k + 1][1])), float(min(P[j][2], P[k + 1][2])),
                   float(max(P[j][0], P[k + 1][0])), float(max(P[j][1], P[k + 1][1])), float(max(P[j][2], P[k + 1][2])))
            _, d = view.project(np.vstack([P[j], P[k + 1]]))
            faces.add(ly, float(d.mean()) + 0.04,
                      _line(pa, pb, stroke=S["path"], stroke_width=lw * 1.8, stroke_linecap="butt"), boxes=[box], kind="route")
            j = k + 1
    return True


# ---------------------------------------------------------------------------
# axonometric view (clean architectural isometric)
# ---------------------------------------------------------------------------
ISO_ELEVATION = math.degrees(math.atan(1 / math.sqrt(2)))   # 35.264..: true isometric

THEMES = {
    "light": dict(bg="#ffffff", line="#1f1f1f", line_w=1.0, floor_top="#f3f1ec", floor_side_a="#d7d3ca",
                  floor_side_b="#c6c1b7", stairs_top="#ebe7de", riser="#d3ccbe", ring="#f7f5f0", void="#4a4a4a",
                  path="#d9482b", label="#111111", muted="#6f6f6f", title="#111111", key_line="#d9d9d9",
                  marker_fill="#d9482b", marker_text="#ffffff", tread="#8f8a80", rail="#dcd8cf", hidden="#a39e94"),
    "dark": dict(bg="#121212", line="#f2f2f2", line_w=1.0, floor_top="#1b1b1b", floor_side_a="#262626",
                 floor_side_b="#202020", stairs_top="#222222", riser="#2e2e2e", ring="#181818", void="#8c8c8c",
                 path="#ff7a4d", label="#f2f2f2", muted="#9a9a9a", title="#ffffff", key_line="#333333",
                 marker_fill="#ff7a4d", marker_text="#121212", tread="#8a8a8a", rail="#2e2e2e", hidden="#5d5d5d"),
    "blueprint": dict(bg="#163a7a", line="#eaf2ff", line_w=1.0, floor_top="#1d4a96", floor_side_a="#2a5cb0",
                      floor_side_b="#1a418a", stairs_top="#2457a8", riser="#2f66bd", ring="#1b4590", void="#9ec1ff",
                      path="#ffd166", label="#ffffff", muted="#bcd0f5", title="#ffffff", key_line="#3b65b5",
                      marker_fill="#ffd166", marker_text="#163a7a", tread="#9ec1ff", rail="#2f66bd", hidden="#7f9fd6"),
    "mono": dict(bg="#ffffff", line="#000000", line_w=1.1, floor_top="#ffffff", floor_side_a="#ffffff",
                 floor_side_b="#ffffff", stairs_top="#ffffff", riser="#ffffff", ring="#ffffff", void="#000000",
                 path="#000000", label="#000000", muted="#555555", title="#000000", key_line="#000000",
                 marker_fill="#000000", marker_text="#ffffff", tread="#000000", rail="#ffffff", hidden="#8c8c8c"),
    "paper": dict(bg="#f4efe6", line="#3b3128", line_w=1.0, floor_top="#efe8da", floor_side_a="#d9cfbc",
                  floor_side_b="#cbbfa9", stairs_top="#e6dcc8", riser="#d4c7ae", ring="#f1ebdf", void="#6b5d4f",
                  path="#b8432f", label="#2b241e", muted="#7a6d60", title="#2b241e", key_line="#cdbfa8",
                  marker_fill="#b8432f", marker_text="#ffffff", tread="#8b7d6b", rail="#d9cfbc", hidden="#a89c8a"),
    "slate": dict(bg="#1e2430", line="#e6ebf2", line_w=1.0, floor_top="#2a3140", floor_side_a="#363e4f",
                  floor_side_b="#2f3747", stairs_top="#323a4a", riser="#3d465a", ring="#262d3b", void="#8fa0bb",
                  path="#5ad1b9", label="#e6ebf2", muted="#9aa7bb", title="#ffffff", key_line="#3d465a",
                  marker_fill="#5ad1b9", marker_text="#1e2430", tread="#8fa0bb", rail="#3d465a", hidden="#6e7a90"),
}
AXO_STYLE = THEMES["light"]


@dataclass
class AxoView:
    azimuth_deg: float = -45.0
    elevation_deg: float = ISO_ELEVATION
    explode_m: float = 0.0          # extra vertical spacing per level
    px_per_m: float = 36.0

    def axes(self):
        az = math.radians(self.azimuth_deg); el = math.radians(self.elevation_deg)
        u = np.array([-math.sin(az), math.cos(az), 0.0])                                      # screen right
        v = np.array([-math.sin(el) * math.cos(az), -math.sin(el) * math.sin(az), math.cos(el)])  # screen up
        w = np.array([math.cos(el) * math.cos(az), math.cos(el) * math.sin(az), math.sin(el)])    # toward viewer
        return u, v, w

    def project(self, p: np.ndarray, level_offset: float = 0.0):
        u, v, w = self.axes()
        p = np.asarray(p, float).reshape(-1, 3).copy()
        p[:, 2] += level_offset
        return np.c_[p @ u, -(p @ v)], p @ w


def _box_sep(a, b, w, eps: float = 1e-6) -> int:
    """Painter's relation of two axis-aligned boxes (x0, y0, z0, x1, y1, z1) seen along
    `w` (unit vector towards the viewer): -1 = a is behind b (draw a first), +1 = b is
    behind a, 0 = undecided (interpenetrating, or separated both ways = no overlap on
    screen). A plane separating the boxes puts the one on the viewer's side in front."""
    votes = set()
    for k in range(3):
        if abs(w[k]) < 1e-9:
            continue
        if a[3 + k] <= b[k] + eps:        # a entirely on the -k side of b
            votes.add(-1 if w[k] > 0 else 1)
        elif b[3 + k] <= a[k] + eps:      # b entirely on the -k side of a
            votes.add(1 if w[k] > 0 else -1)
    return votes.pop() if len(votes) == 1 else 0


def _sb_overlap(a, b, tol: float = 1e-4) -> bool:
    return a[0] < b[2] - tol and b[0] < a[2] - tol and a[1] < b[3] - tol and b[1] < a[3] - tol


@dataclass
class _Item:
    depth: float
    svg: str
    members: list           # [(box3d, screen_bbox)] of the solids this fragment draws
    sbox: tuple | None      # screen bbox of all members
    edges: list             # [(a3, b3)] top-face edges, for the hidden-line pass
    kind: str               # "stairs" | "flat" | ""


class _Faces:
    """Collects svg fragments per layer. Within a layer the fragments are ordered
    far-to-near: fragments that carry axis-aligned boxes get an exact order from
    separating planes (topological sort), the centroid depth only breaks ties, so
    touching step boxes, landings and route dashes never paint over each other."""

    def __init__(self, view: "AxoView"):
        self.view = view
        self.w = view.axes()[2]
        self.layers: dict[str, list[_Item]] = {}

    def add(self, layer: str, depth: float, svg: str, boxes=None, edges=None, kind: str = "") -> None:
        members = []
        for bx in boxes or []:
            x0, y0, z0, x1, y1, z1 = bx
            pts = np.array([[x, y, z] for x in (x0, x1) for y in (y0, y1) for z in (z0, z1)])
            p2, _ = self.view.project(pts)
            members.append((tuple(float(v) for v in bx),
                            (float(p2[:, 0].min()), float(p2[:, 1].min()), float(p2[:, 0].max()), float(p2[:, 1].max()))))
        sbox = None
        if members:
            arr = np.array([m[1] for m in members])
            sbox = (float(arr[:, 0].min()), float(arr[:, 1].min()), float(arr[:, 2].max()), float(arr[:, 3].max()))
        self.layers.setdefault(layer, []).append(_Item(depth, svg, members, sbox, list(edges or []), kind))

    def _boxes(self):
        items = [it for lst in self.layers.values() for it in lst]
        allboxes = [(np.array(bx, float), sb, id(it)) for it in items for bx, sb in it.members]
        if not allboxes:
            return items, None, None, None
        B = np.array([b[0] for b in allboxes])            # (M, 6)
        SB = np.array([b[1] for b in allboxes])           # (M, 4) screen bboxes
        owner = np.array([b[2] for b in allboxes])
        return items, B, SB, owner

    def _hidden_mask(self, P: np.ndarray, B: np.ndarray, sel: np.ndarray) -> np.ndarray:
        """P: (N,3) points, sel: indices into B of candidate boxes -> (N,) bool: the ray from
        the point towards the viewer passes through some box (slab test)."""
        if len(sel) == 0:
            return np.zeros(len(P), bool)
        w = np.asarray(self.w, float)
        l_, h_ = B[sel, :3], B[sel, 3:]                                    # (M,3)
        s_in = np.full((len(P), len(sel)), -np.inf); s_out = np.full((len(P), len(sel)), np.inf)
        for k in range(3):
            if abs(w[k]) < 1e-9:
                inside = (P[:, None, k] >= l_[None, :, k] - 1e-9) & (P[:, None, k] <= h_[None, :, k] + 1e-9)
                s_out = np.where(inside, s_out, -np.inf)
                continue
            s1 = (l_[None, :, k] - P[:, None, k]) / w[k]; s2 = (h_[None, :, k] - P[:, None, k]) / w[k]
            s_in = np.maximum(s_in, np.minimum(s1, s2)); s_out = np.minimum(s_out, np.maximum(s1, s2))
        hit = np.maximum(s_in, 0.0) < s_out - 1e-6
        return hit.any(axis=1)

    def visible(self, P: np.ndarray) -> np.ndarray:
        """(N,) bool: which of the 3D points are not hidden behind any solid."""
        _, B, _, _ = self._boxes()
        if B is None:
            return np.ones(len(P), bool)
        out = np.ones(len(P), bool)
        for i in range(0, len(P), 512):                   # chunked: N x M arrays
            out[i:i + 512] = ~self._hidden_mask(P[i:i + 512], B, np.arange(len(B)))
        return out

    def hidden_segments(self, kinds: tuple = ("stairs",), step: float = 0.03):
        """Parts of the items' top-face edges that lie behind some solid, as 3D segments.
        A point is hidden when the ray from it towards the viewer passes through a box
        (slab test); sampled along each edge and refined by bisection at the ends."""
        items, B, SB, owner = self._boxes()
        if B is None:
            return []

        def hidden_mask(P: np.ndarray, sel: np.ndarray) -> np.ndarray:
            return self._hidden_mask(P, B, sel)

        out = []
        for it in items:
            if it.kind not in kinds or not it.edges:
                continue
            for a3, b3 in it.edges:
                a3 = np.asarray(a3, float); b3 = np.asarray(b3, float)
                L = float(np.linalg.norm(b3 - a3))
                if L < 1e-6:
                    continue
                p2, _ = self.view.project(np.vstack([a3, b3]))
                eb = (p2[:, 0].min(), p2[:, 1].min(), p2[:, 0].max(), p2[:, 1].max())
                cand = np.where((owner != id(it)) & (SB[:, 0] < eb[2] + 1e-6) & (SB[:, 2] > eb[0] - 1e-6) &
                                (SB[:, 1] < eb[3] + 1e-6) & (SB[:, 3] > eb[1] - 1e-6))[0]
                if len(cand) == 0:
                    continue
                n = max(2, int(np.ceil(L / step)) + 1)
                ts = np.linspace(0.0, 1.0, n)
                P = a3[None, :] + (b3 - a3)[None, :] * ts[:, None]
                hm = hidden_mask(P, cand)
                if not hm.any():
                    continue

                def refine(t_vis, t_hid):
                    for _ in range(10):
                        tm = 0.5 * (t_vis + t_hid)
                        if hidden_mask((a3 + (b3 - a3) * tm)[None, :], cand)[0]:
                            t_hid = tm
                        else:
                            t_vis = tm
                    return t_hid
                i = 0
                while i < n:
                    if not hm[i]:
                        i += 1; continue
                    j = i
                    while j + 1 < n and hm[j + 1]:
                        j += 1
                    t0 = ts[i] if i == 0 else refine(ts[i - 1], ts[i])
                    t1 = ts[j] if j == n - 1 else refine(ts[j + 1], ts[j])
                    if t1 - t0 > 1e-4:
                        out.append((a3 + (b3 - a3) * t0, a3 + (b3 - a3) * t1))
                    i = j + 1
        return out

    def _relation(self, A: _Item, B: _Item) -> int:
        if A.sbox is None or B.sbox is None or not _sb_overlap(A.sbox, B.sbox):
            return 0
        # the route is an overlay on its own stair: never hidden by the treads and landings
        # it runs across (a higher storey, drawn in a later layer, still covers it)
        if {A.kind, B.kind} == {"route", "stairs"}:
            return -1 if A.kind == "stairs" else 1
        votes = set()
        for ba, sa in A.members:
            if not _sb_overlap(sa, B.sbox):
                continue
            for bb, sb in B.members:
                if not _sb_overlap(sa, sb):
                    continue
                r = _box_sep(ba, bb, self.w)
                if r:
                    votes.add(r)
        return votes.pop() if len(votes) == 1 else 0

    def _order(self, items: list[_Item]) -> list[_Item]:
        import heapq
        n = len(items)
        succ = [[] for _ in range(n)]
        indeg = [0] * n
        for i in range(n):
            if items[i].sbox is None:
                continue
            for j in range(i + 1, n):
                r = self._relation(items[i], items[j])
                if r == -1:
                    succ[i].append(j); indeg[j] += 1
                elif r == 1:
                    succ[j].append(i); indeg[i] += 1
        heap = [(items[i].depth, i) for i in range(n) if indeg[i] == 0]
        heapq.heapify(heap)
        done = [False] * n
        out: list[_Item] = []
        remaining = set(range(n))
        while len(out) < n:
            if not heap:                        # cycle: release the farthest remaining fragment
                i = min(remaining, key=lambda k: items[k].depth)
                heapq.heappush(heap, (items[i].depth, i))
            _, i = heapq.heappop(heap)
            if done[i]:
                continue
            done[i] = True; remaining.discard(i); out.append(items[i])
            for j in succ[i]:
                indeg[j] -= 1
                if indeg[j] <= 0 and not done[j]:
                    heapq.heappush(heap, (items[j].depth, j))
        return out

    def emit(self, order: list[str]) -> list[str]:
        out = []
        for name in order:
            items = self.layers.get(name)
            if not items:
                continue
            out.append(f'<g id="{name}">')
            out += [it.svg for it in self._order(items)]
            out.append("</g>")
        return out


def _quad_depth(d: np.ndarray) -> float:
    return float(d.mean())


def _side_tone(normal_xy: np.ndarray, view: AxoView, a: str, b: str) -> str:
    """Two-tone shading by which way a vertical face points relative to the viewer."""
    u, _, w = view.axes()
    n = np.array([normal_xy[0], normal_xy[1], 0.0])
    return a if (n @ u) >= 0 else b


def write_axo_svg(path, diagram: dict, view: AxoView, cutaway_height: float = 1.2, title: str | None = None,
                  shade: bool = True, stairs_wall_height: float = 0.9, slab: float = 0.15,
                  show_path: bool = True, draw_walls: bool = False, theme: str = "light",
                  label_style: str = "inline", path_on_stairs: bool = True, hidden_lines: str = "stairs",
                  path_morse: str | None = None, morse_unit_px: float = 4.0) -> None:
    """Clean isometric building diagram: floor slabs, stairs (sloped slabs with
    tread lines, or real steps when a run carries per-quad thickness), the
    walk path, labels inline or as numbered markers with a key. Grouped per
    level for editing. `theme` is one of THEMES."""
    S = dict(THEMES.get(theme, THEMES["light"]))
    lw = S["line_w"]
    levels = {lv["id"]: lv for lv in diagram["levels"]}
    faces = _Faces(view)
    ppm = view.px_per_m
    cam_h = diagram.get("camera_height_m", 1.5)

    def lvl_off(level_id: int) -> float:
        return view.explode_m * max(level_id, 0)

    level_floors = sorted((lv["floor_z"], lid) for lid, lv in levels.items())

    def off_for_z(z_floor: float) -> float:
        """Exploded offset for an arbitrary floor height (interpolated between levels)."""
        if view.explode_m == 0 or len(level_floors) < 2:
            return 0.0
        if z_floor <= level_floors[0][0]:
            return lvl_off(level_floors[0][1])
        for (za, la), (zb, lb) in zip(level_floors[:-1], level_floors[1:]):
            if za <= z_floor <= zb:
                t = (z_floor - za) / max(zb - za, 1e-9)
                return lvl_off(la) + (lvl_off(lb) - lvl_off(la)) * t
        return lvl_off(level_floors[-1][1])

    def run_layer(r: dict) -> str:
        if r["kind"] == "stairs":
            lo = min(r["level_from"], r["level_to"]) if r["level_from"] >= 0 and r["level_to"] >= 0 else max(r["level_from"], r["level_to"], 0)
            return f"stairs-L{lo}"
        return f"level-L{r['level']}"

    def stairs_layer_for_z(z: float) -> str:
        """Stairs geometry sits just above the highest floor that is not above it."""
        below = [(lv["floor_z"], lid) for lid, lv in levels.items() if lv["floor_z"] <= z + 1e-6]
        if not below:
            return "stairs-Lsub"
        return f"stairs-L{max(below)[1]}"

    all_pts = []

    def P(pts3, off=0.0):
        p2, d = view.project(pts3, off)
        all_pts.append(p2)
        return p2 * ppm, d

    def poly(p2, fill, stroke=S["line"], w=lw, extra=None):
        kw = dict(fill=fill, stroke=stroke, stroke_width=w, stroke_linejoin="round")
        if extra:
            kw.update(extra)
        return _poly(p2, **kw)

    def slab_faces(layer, quad_top, offs, top_fill, thickness, side_fills=None, kind="flat"):
        """Top face + the visible side faces of a 4-point slab (l0,l1,r1,r0); `offs`
        are per-vertex exploded offsets."""
        q = np.asarray(quad_top, float).copy()
        offs = np.asarray(offs, float)
        q[:, 2] += offs
        bottom = q.copy(); bottom[:, 2] -= thickness
        p2, d = view.project(q); all_pts.append(p2)
        parts = [poly(p2 * ppm, top_fill)]
        sa, sb = side_fills or (S["floor_side_a"], S["floor_side_b"])
        _, _, wv = view.axes()
        for k in range(4):
            k2 = (k + 1) % 4
            e = q[k2, :2] - q[k, :2]
            if np.linalg.norm(e) < 1e-6:
                continue
            n = np.array([e[1], -e[0]]); n /= np.linalg.norm(n)
            if np.array([n[0], n[1], 0.0]) @ wv <= 0:
                continue
            side = np.array([q[k], q[k2], bottom[k2], bottom[k]])
            p2s, _ = view.project(side)
            parts.append(poly(p2s * ppm, _side_tone(n, view, sa, sb)))
        box = (float(q[:, 0].min()), float(q[:, 1].min()), float(bottom[:, 2].min()),
               float(q[:, 0].max()), float(q[:, 1].max()), float(q[:, 2].max()))
        edges = [(q[k].copy(), q[(k + 1) % 4].copy()) for k in range(4)]
        faces.add(layer, _quad_depth(d) - 0.03, "".join(parts), boxes=[box], edges=edges, kind=kind)

    # coplanar axis-aligned flat surfaces are merged per height so touching
    # slabs, landings and walkways share one outline (no borders inside)
    flat_groups: dict[tuple, dict] = {}

    def collect_flat(quad: np.ndarray, off: float, layer: str, layer_rank: int, thickness: float, fill: str) -> bool:
        if not is_axis_rect(quad) or np.ptp(quad[:, 2]) > 1e-6:
            return False
        key = (round(float(quad[0, 2]), 3), round(float(off), 3), round(float(thickness), 3))
        g = flat_groups.setdefault(key, {"rects": [], "layer": layer, "rank": layer_rank, "fill": fill,
                                         "z": float(quad[0, 2]), "off": float(off), "thickness": float(thickness),
                                         "stairs": False})
        g["rects"].append(rect_of(quad))
        if fill == S["stairs_top"]:
            g["stairs"] = True
        if layer_rank > g["rank"]:
            g["layer"], g["rank"], g["fill"] = layer, layer_rank, fill
        return True

    def layer_rank_of(layer: str) -> int:
        # a merged surface belongs to its floor's layer when any member is a floor slab;
        # only groups made of stair landings alone stay in a stairs layer
        kind, _, num = layer.partition("-L")
        try:
            n = int(num)
        except ValueError:
            n = 0
        return n * 2 + (1 if kind == "level" else 0)

    for r in diagram["runs"]:
        layer = run_layer(r)
        if r["kind"] == "mark":
            # break mark: a zig-zag across the corridor where it is drawn shorter than it is
            off = lvl_off(r["level"])
            for pl in r.get("break_marks", []):
                P3 = np.array(pl, float); P3[:, 2] += off
                p2, d = view.project(P3); all_pts.append(p2)
                box = (float(P3[:, 0].min()), float(P3[:, 1].min()), float(P3[:, 2].min()),
                       float(P3[:, 0].max()), float(P3[:, 1].max()), float(P3[:, 2].max()))
                faces.add(layer, float(d.mean()) + 0.03,
                          _polyline(p2 * ppm, fill="none", stroke=S["line"], stroke_width=lw, stroke_linejoin="round"), boxes=[box])
            continue
        quads = [np.array(q, float) for q in (r.get("quads") or [])]
        if not quads:
            L = np.array(r["left_boundary"]); R = np.array(r["right_boundary"])
            quads = [np.array([L[j], L[j + 1], R[j + 1], R[j]]) for j in range(len(L) - 1)]
        thick = r.get("quad_thickness") or [slab] * len(quads)
        if r["kind"] == "stairs":
            o0, o1 = lvl_off(r["level_from"]), lvl_off(r["level_to"])
            z_lo = min(q[:, 2].min() for q in quads); z_hi = max(q[:, 2].max() for q in quads)
            lo_, hi_ = (o0, o1) if o1 >= o0 else (o1, o0)
            span = max(z_hi - z_lo, 1e-6)

            def off_at(z):
                return lo_ + (hi_ - lo_) * (z - z_lo) / span

            def at_hi(_r):
                return True
            is_steps = bool(r.get("steps"))
            for quad, th in zip(quads, thick):
                offs = np.array([off_at(z) for z in quad[:, 2]])
                layer = stairs_layer_for_z(float(quad[:, 2].min()))
                if is_steps and (abs(th - slab) < 1e-6 or th >= 0.3) and np.ptp(quad[:, 2]) <= 1e-6:
                    # landing / walkway: merge with the floor it belongs to
                    zq = float(quad[0, 2])
                    lid = r["level_to"] if abs(zq - (z_hi if r["level_to"] >= 0 and at_hi(r) else z_lo)) < 1e-6 else -1
                    target_layer = f"level-L{r['level_to']}" if (r["level_to"] >= 0 and abs(zq - z_hi) < 1e-6 and r["level_to"] > r["level_from"]) else \
                                   f"level-L{r['level_from']}" if (r["level_from"] >= 0 and abs(zq - z_hi) < 1e-6 and r["level_from"] > r["level_to"]) else layer
                    if collect_flat(quad, float(offs[0]), target_layer, layer_rank_of(target_layer), th, S["stairs_top"]):
                        continue
                slab_faces(layer, quad, offs, S["stairs_top"], th,
                           side_fills=(S["riser"], S["riser"]) if is_steps else None, kind="stairs")
                if not is_steps and abs(quad[1, 2] - quad[0, 2]) > 1e-6:   # tread lines on sloped slabs
                    seg_len = np.linalg.norm(quad[1, :2] - quad[0, :2]); m = int(seg_len / 0.3)
                    for k in range(1, m):
                        t = k / m
                        a = quad[0] + (quad[1] - quad[0]) * t; b = quad[3] + (quad[2] - quad[3]) * t
                        pa, da = view.project(a, off_at(a[2])); pb, _ = view.project(b, off_at(b[2]))
                        faces.add(layer, float(da[0]) - 0.01, _line(pa[0] * ppm, pb[0] * ppm, stroke=S["tread"], stroke_width=lw * 0.8))
            if r.get("atrium"):
                at = r["atrium"]
                corners = np.array(at["corners"])
                lo_layer = f"level-L{min(r['level_from'], r['level_to'])}" if r["level_from"] >= 0 else layer
                hi_layer = f"level-L{max(r['level_from'], r['level_to'])}" if r["level_to"] >= 0 else layer
                oz0 = lvl_off(r["level_from"]) if at["z0"] <= at["z1"] else lvl_off(r["level_to"])
                oz1 = lvl_off(r["level_to"]) if at["z0"] <= at["z1"] else lvl_off(r["level_from"])
                z_lo2, z_hi2 = min(at["z0"], at["z1"]), max(at["z0"], at["z1"])
                ring_levels = ((z_lo2, lo_layer, oz0), (z_hi2, hi_layer, oz1))
                if at.get("ring_levels") == "upper":
                    ring_levels = (ring_levels[1],)
                elif at.get("ring_levels") == "none":
                    ring_levels = ()
                for z, ly, oz in ring_levels:
                    for q in at.get("ring", []):
                        quad = np.c_[np.array(q), np.full(4, z)]
                        slab_faces(ly, quad, [oz] * 4, S["ring"], slab)
                bot = np.c_[corners, np.full(4, z_lo2 + oz0)]
                top = np.c_[corners, np.full(4, z_hi2 + oz1)]
                pb, db = view.project(bot); pt, dt = view.project(top)
                all_pts.append(pb); all_pts.append(pt)
                for k in (range(4) if at.get("void_lines", True) else ()):
                    k2 = (k + 1) % 4
                    faces.add(layer, float(db[[k, k2]].mean()) + 0.05, _line(pb[k] * ppm, pb[k2] * ppm, stroke=S["void"], stroke_width=lw * 0.9))
                    faces.add(layer, float(dt[[k, k2]].mean()) + 0.05, _line(pt[k] * ppm, pt[k2] * ppm, stroke=S["void"], stroke_width=lw * 0.9))
                    faces.add(layer, float((db[k] + dt[k]) / 2) + 0.05, _line(pb[k] * ppm, pt[k] * ppm, stroke=S["void"], stroke_width=lw * 0.9))
            # dashed verticals at the block corners (stairwell), split at the floors so each
            # piece is ordered with the geometry of its own storey
            for sl in r.get("shaft_lines", []):
                x, y, zlo, zhi = (float(v) for v in sl)
                cuts = sorted({zlo, zhi} | {lv["floor_z"] for lv in levels.values() if zlo < lv["floor_z"] < zhi})
                cum = 0.0
                for za, zb in zip(cuts[:-1], cuts[1:]):
                    oa, ob = off_for_z(za), off_for_z(zb)
                    A = np.array([x, y, za + oa]); B = np.array([x, y, zb + ob])
                    p2, d = view.project(np.vstack([A, B])); all_pts.append(p2)
                    faces.add(stairs_layer_for_z(za + 1e-6), float(d.mean()) + 0.02,
                              _line(p2[0] * ppm, p2[1] * ppm, stroke=S["line"], stroke_width=lw * 0.8,
                                    stroke_dasharray="5 4", stroke_dashoffset=f"{-cum:.2f}", stroke_linecap="butt"),
                              boxes=[(x, y, za + oa, x, y, zb + ob)])
                    cum += float(np.linalg.norm((p2[1] - p2[0]) * ppm))
            for w in (r["walls"] if draw_walls else []):   # balustrades
                a, b = np.array(w["a"]), np.array(w["b"])
                quad = np.array([[*a, w["z0"] + o0], [*b, w["z1"] + o1], [*b, w["z1"] + o1 + stairs_wall_height],
                                 [*a, w["z0"] + o0 + stairs_wall_height]])
                p2, d = P(quad)
                faces.add(layer, _quad_depth(d), poly(p2, S["rail"], extra={"fill_opacity": 0.9}))
        else:
            off = lvl_off(r["level"])
            for quad, th in zip(quads, thick):
                if not collect_flat(quad, off, layer, layer_rank_of(layer), th, S["floor_top"]):
                    slab_faces(layer, quad, [off] * 4, S["floor_top"], th)
            for w in (r["walls"] if draw_walls else []):
                a, b = np.array(w["a"]), np.array(w["b"])
                z0 = w["z0"]
                e = b - a
                nrm = np.array([-e[1], e[0]]); nrm /= max(np.linalg.norm(nrm), 1e-9)
                if w["side"] == "right":
                    nrm = -nrm
                for t0, t1 in w["solid"]:
                    pa = a + (b - a) * t0; pb = a + (b - a) * t1
                    quad = np.array([[*pa, z0], [*pb, z0], [*pb, z0 + cutaway_height], [*pa, z0 + cutaway_height]])
                    p2, d = P(quad, off)
                    tone = _side_tone(nrm, view, S["floor_side_a"], S["floor_side_b"]) if shade else S["floor_side_a"]
                    faces.add(layer, _quad_depth(d), poly(p2, tone))
                for t0, t1 in w["openings"]:
                    pa = a + (b - a) * t0; pb = a + (b - a) * t1
                    p2, d = P(np.array([[*pa, z0], [*pb, z0]]), off)
                    faces.add(layer, _quad_depth(d), _line(p2[0], p2[1], stroke=S["muted"], stroke_width=lw,
                                                            stroke_dasharray="3 3"))
    # merged flats: one outline per connected coplanar surface, side faces on its boundary
    _, _, wv = view.axes()
    for g in flat_groups.values():
        loops = union_loops(g["rects"])
        if not loops:
            continue
        z = g["z"] + g["off"]
        allp = np.vstack(loops)
        p3 = np.c_[allp, np.full(len(allp), z)]
        p2all, dall = view.project(p3); all_pts.append(p2all)
        depth = float(dall.min())      # farthest point: whatever rests on the slab is drawn after it
        d_parts = []
        for lp in loops:
            p2, _ = view.project(np.c_[lp, np.full(len(lp), z)])
            d_parts.append("M " + " L ".join(f"{_fmt(x * ppm)},{_fmt(y * ppm)}" for x, y in p2) + " Z")
        parts = [f'<path d="{" ".join(d_parts)}" fill="{g["fill"]}" fill-rule="evenodd" stroke="{S["line"]}" '
                 f'stroke-width="{lw}" stroke-linejoin="round"/>']
        for lp in loops:
            n_ = len(lp)
            for k in range(n_):
                a, b = lp[k], lp[(k + 1) % n_]
                e = b - a
                nrm = np.array([e[1], -e[0]]) if loop_is_ccw(lp) else np.array([-e[1], e[0]])
                nrm = nrm / max(np.linalg.norm(nrm), 1e-9)
                if np.array([nrm[0], nrm[1], 0.0]) @ wv <= 0:
                    continue
                side = np.array([[*a, z], [*b, z], [*b, z - g["thickness"]], [*a, z - g["thickness"]]])
                p2s, _ = view.project(side)
                parts.append(poly(p2s * ppm, _side_tone(nrm, view, S["floor_side_a"], S["floor_side_b"])))
        boxes = [(x0, y0, z - g["thickness"], x1, y1, z) for (x0, y0, x1, y1) in g["rects"]]
        edges = [(np.r_[lp[k], z], np.r_[lp[(k + 1) % len(lp)], z]) for lp in loops for k in range(len(lp))]
        faces.add(g["layer"], depth - 0.03, "".join(parts), boxes=boxes, edges=edges,
                  kind="stairs" if g["stairs"] else "flat")

    # path: every dash is its own short solid line placed in the level / stairs layer of
    # its height and ordered with the geometry, so a higher surface drawn later hides the
    # stretch of route beneath it. Dashes are fitted to each straight run so a run starts
    # and ends with a full dash: corners stay solid.
    if show_path:
        pts = np.array(diagram["path"]["points"])
        lv = diagram["path"].get("level") or [0] * len(pts)
        stairs_runs = [r for r in diagram["runs"] if r["kind"] == "stairs"]
        stairs_boxes = []
        for r in stairs_runs:
            q = np.vstack([np.array(qq) for qq in r["quads"]]) if r.get("quads") else np.vstack([np.array(r["left_boundary"]), np.array(r["right_boundary"])])
            stairs_boxes.append((q[:, 0].min() - 0.5, q[:, 1].min() - 0.5, q[:, 0].max() + 0.5, q[:, 1].max() + 0.5))

        def layer_for(xy, z_top: float, flag: int) -> str:
            if flag >= 0:
                return f"level-L{flag}"
            for x0, y0, x1, y1 in stairs_boxes:
                if x0 <= xy[0] <= x1 and y0 <= xy[1] <= y1:
                    return stairs_layer_for_z(z_top + 1e-6)      # a dash belongs with the highest tread it touches
            return "path"
        # straight runs (same direction, same level flag) in floor-relative coordinates
        runs_: list[list] = []        # [a3, b3, flag]
        for (p0, l0), (p1, l1) in zip(zip(pts[:-1], lv[:-1]), zip(pts[1:], lv[1:])):
            if l0 == -9:                    # pen up: no segment out of a break point
                continue
            if l1 == -9:
                l1 = l0
            if not (l0 >= 0 or path_on_stairs) and not (l1 >= 0):
                continue
            flag = l0 if l0 == l1 else (max(l0, l1) if min(l0, l1) >= 0 else -1)
            a = np.array([p0[0], p0[1], p0[2] - cam_h]); b = np.array([p1[0], p1[1], p1[2] - cam_h])
            L = float(np.linalg.norm(b - a))
            if L < 1e-6:
                continue
            dvec = (b - a) / L
            if runs_ and runs_[-1][2] == flag and np.linalg.norm(runs_[-1][1] - a) < 1e-6:
                prev_d = runs_[-1][1] - runs_[-1][0]; prev_d = prev_d / max(np.linalg.norm(prev_d), 1e-9)
                if abs(float(prev_d @ dvec) - 1.0) < 1e-6:
                    runs_[-1][1] = b
                    continue
            runs_.append([a, b, flag])
        if path_morse is not None and _emit_morse_route(faces, view, ppm, S, lw, runs_, path_morse, morse_unit_px, cam_h,
                                                        lvl_off, off_for_z, layer_for, stairs_layer_for_z, all_pts):
            runs_ = []                                    # the message replaces the plain dashes
        dash_px, gap_px = 6.0, 4.0
        for a, b, flag in runs_:
            oa = lvl_off(flag) if flag >= 0 else off_for_z(float(a[2]))
            ob = lvl_off(flag) if flag >= 0 else off_for_z(float(b[2]))
            a3 = np.array([a[0], a[1], a[2] + 0.06 + oa]); b3 = np.array([b[0], b[1], b[2] + 0.06 + ob])
            p2, _ = view.project(np.vstack([a3, b3])); all_pts.append(p2)
            L_px = float(np.linalg.norm((p2[1] - p2[0]) * ppm))
            if L_px < 1e-3:
                continue
            n_d = max(1, int(round((L_px + gap_px) / (dash_px + gap_px))))
            period = (L_px + gap_px) / n_d
            dash = max(period - gap_px, min(L_px, 2.0)) if n_d > 1 else L_px
            for k in range(n_d):
                t0 = (k * period) / L_px; t1 = min(1.0, (k * period + dash) / L_px)
                sa = a3 + (b3 - a3) * t0; sb = a3 + (b3 - a3) * t1
                q2, d = view.project(np.vstack([sa, sb]))
                za = a[2] + (b[2] - a[2]) * t0; zb = a[2] + (b[2] - a[2]) * t1
                ly = layer_for(0.5 * (sa + sb)[:2], max(za, zb), flag)
                box = (float(min(sa[0], sb[0])), float(min(sa[1], sb[1])), float(min(sa[2], sb[2])),
                       float(max(sa[0], sb[0])), float(max(sa[1], sb[1])), float(max(sa[2], sb[2])))
                faces.add(ly, float(d.mean()) + 0.04,
                          _line(q2[0] * ppm, q2[1] * ppm, stroke=S["path"], stroke_width=lw * 1.8, stroke_linecap="butt"),
                          boxes=[box], kind="route")
    # labels
    fs = max(11.0, 0.4 * ppm)
    key_rows = []
    labels = list(diagram.get("labels", []))
    if label_style == "key":
        def _rank(lb):
            role = lb.get("role") or ("start" if lb["text"] == "Start" else "end" if lb["text"] == "End" else "")
            return 0 if role == "start" else 2 if role == "end" else 1
        order = sorted(range(len(labels)), key=lambda i: (_rank(labels[i]), i))
        labels = [labels[i] for i in order]
    for n, lb in enumerate(labels, start=1):
        p = np.array(lb["xyz"], float)
        off = lvl_off(lb.get("level", 0)) if lb.get("level", 0) >= 0 else off_for_z(lb.get("floor_z", p[2]))
        base = p.copy(); base[2] = lb.get("floor_z", p[2])
        is_stairs = lb.get("kind") in ("stairs", "corridor")
        if label_style == "key":
            top = base.copy(); top[2] += 1.3
            pb, _ = P(base, off); pt, _ = P(top, off)
            pb, pt = pb[0], pt[0]
            faces.add("labels", 0.0, _line(pb, pt, stroke=S["line"], stroke_width=lw))
            rr = fs * 0.62
            faces.add("labels", 0.0, f'<circle cx="{_fmt(pt[0])}" cy="{_fmt(pt[1] - rr)}" r="{_fmt(rr)}" fill="{S["marker_fill"]}" stroke="{S["bg"]}" stroke-width="{lw}"/>')
            faces.add("labels", 0.0, _text((pt[0], pt[1] - rr + fs * 0.36), str(n), fs * 0.95, fill=S["marker_text"],
                                           font_weight="700", text_anchor="middle"))
            key_rows.append((n, lb["text"]))
            all_pts.append(np.array([(pt - [0, 2 * rr]) / ppm]))
        else:
            top = base.copy(); top[2] += (cutaway_height if draw_walls else 0.0) + 1.0
            pb, _ = P(base, off); pt, _ = P(top, off)
            pb, pt = pb[0], pt[0]
            col = S["muted"] if is_stairs else S["label"]
            faces.add("labels", 0.0, _line(pb, pt, stroke=col, stroke_width=lw))
            if not is_stairs:
                faces.add("labels", 0.0, f'<circle cx="{_fmt(pb[0])}" cy="{_fmt(pb[1])}" r="{_fmt(fs * 0.25)}" fill="{S["path"]}"/>')
            faces.add("labels", 0.0, _text((pt[0] + 5, pt[1] - 3), lb["text"], fs * (0.8 if is_stairs else 1.0), fill=col,
                                           font_weight="500" if is_stairs else "700"))
    # level tags: at the left-most point of each level's floor geometry, below it (inline style only)
    per_level: dict[int, list[np.ndarray]] = {}
    for r in diagram["runs"]:
        if r["kind"] != "flat" or label_style == "key":
            continue
        pts3 = np.vstack([np.array(q) for q in r["quads"]]) if r.get("quads") else np.vstack([np.array(r["left_boundary"]), np.array(r["right_boundary"])])
        p2, _ = view.project(pts3, lvl_off(r["level"]))
        per_level.setdefault(r["level"], []).append(p2 * ppm)
    for lid, chunks in per_level.items():
        lv_ = levels.get(lid)
        if lv_ is None:
            continue
        A_ = np.vstack(chunks)
        k = int(np.argmin(A_[:, 0]))
        faces.add("labels", 0.0, _text((A_[k, 0] - fs * 0.3, A_[k, 1] + fs * 1.6), lv_["name"], fs * 0.85,
                                       fill=S["muted"], text_anchor="end", font_weight="500"))
        all_pts.append(np.array([[A_[k, 0] / ppm - fs * 3.5 / ppm, A_[k, 1] / ppm + fs * 2 / ppm]]))

    A = np.vstack(all_pts) * ppm
    # margins: generous at the sides, tight above so the drawing sits high on the page with
    # the title close over it, the key below
    pad = fs * 2.0
    x0 = A[:, 0].min() - pad; x1 = A[:, 0].max() + pad
    y0 = A[:, 1].min() - fs * 1.0; y1 = A[:, 1].max() + fs * 1.5
    top_extra = fs * 2.0 if title else fs * 0.5
    key_h = (len(key_rows) + 0.5) * fs * 1.5 + fs if key_rows else 0.0
    W, H = x1 - x0, (y1 - y0) + top_extra + key_h
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W:.0f}" height="{H:.0f}" '
           f'viewBox="{x0:.1f} {y0 - top_extra:.1f} {W:.1f} {H:.1f}" font-family="Helvetica Neue, Helvetica, Arial, sans-serif">',
           f'<rect x="{x0:.1f}" y="{y0 - top_extra:.1f}" width="{W:.1f}" height="{H:.1f}" fill="{S["bg"]}"/>']
    # draw order: by floor height (a landing-level hall between two storeys sorts between them),
    # each stairs layer right after the level it rises from
    def layer_height(name: str) -> float:
        kind, _, num = name.partition("-L")
        if num == "sub":
            return -1e9
        try:
            lid = int(num)
        except ValueError:
            return 1e9
        zf = levels.get(lid, {}).get("floor_z", 0.0)
        return zf + (0.001 if kind == "stairs" else 0.0)
    # hidden edges of the stairs (treads, landings) as light dashed lines: the architect's
    # convention for what lies behind or beneath another surface
    if hidden_lines != "none":
        kinds = ("stairs",) if hidden_lines == "stairs" else ("stairs", "flat")
        for a3, b3 in faces.hidden_segments(kinds):
            p2, _ = view.project(np.vstack([a3, b3]))
            faces.add("hidden", 0.0, _line(p2[0] * ppm, p2[1] * ppm, stroke=S["hidden"], stroke_width=lw * 0.55,
                                           stroke_dasharray=f"{lw * 2.2:.2f} {lw * 2.2:.2f}", stroke_linecap="butt"))
    order_layers = sorted([k for k in faces.layers if k not in ("path", "labels", "hidden")], key=layer_height)
    order_layers += ["hidden", "path", "labels"]
    out += faces.emit(order_layers)
    if title:
        out.append(_text((x0 + fs, y0 - fs * 0.5), title, fs * 1.3, fill=S["title"], font_weight="700"))
    if key_rows:
        out.append('<g id="key">')
        ky = y1 + fs * 1.2
        out.append(_line((x0 + fs, ky - fs * 0.6), (x1 - fs, ky - fs * 0.6), stroke=S["key_line"], stroke_width=lw))
        rr = fs * 0.62
        for i, (n, text) in enumerate(key_rows):
            cy = ky + fs * 0.6 + i * fs * 1.5
            cx = x0 + fs + rr
            out.append(f'<circle cx="{_fmt(cx)}" cy="{_fmt(cy)}" r="{_fmt(rr)}" fill="{S["marker_fill"]}"/>')
            out.append(_text((cx, cy + fs * 0.36), str(n), fs * 0.95, fill=S["marker_text"], font_weight="700", text_anchor="middle"))
            out.append(_text((cx + rr + fs * 0.7, cy + fs * 0.36), text, fs, fill=S["label"], font_weight="500"))
        out.append("</g>")
    out.append("</svg>")
    Path(path).write_text("\n".join(out), encoding="utf-8")
