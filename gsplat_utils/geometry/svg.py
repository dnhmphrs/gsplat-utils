"""SVG output: per-level plans and an axonometric 'sign' view. No dependencies."""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

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
    # floors
    for r in sel:
        if r["kind"] == "stairs" and level_id is not None and level_id not in (r["level_from"], r["level_to"]):
            continue
        if r["kind"] == "stairs" and r.get("atrium"):   # gallery ring first, flights on top
            upper_only = r["atrium"].get("ring_levels") == "upper"
            upper = max(r["level_from"], r["level_to"])
            if not upper_only or level_id is None or level_id == upper:
                for q in r["atrium"].get("ring", []):
                    out.append(_poly(np.array([P(p) for p in q]), fill="#f7f5f0", stroke="#8c8577", stroke_width=0.8))
        fill = PALETTE["stairs"] if r["kind"] == "stairs" else PALETTE["floor"]
        for q in r.get("quads") or [r["outline"]]:
            pts = np.array([P(p) for p in q])
            out.append(_poly(pts, fill=fill, stroke="none" if draw_walls else PALETTE["wall"],
                             stroke_width=1.2, stroke_linejoin="round", fill_opacity=0.95))
        if r["kind"] == "stairs":
            out.extend(_tread_lines(r, P, px_per_m))
            if r.get("atrium"):
                at = r["atrium"]
                c = np.array([P(p) for p in at["corners"]])
                out.append(_poly(c, fill="white", stroke="#4a4a4a", stroke_width=1.2, stroke_dasharray="5 3"))
                # diagonal cross = void (architectural convention)
                out.append(_line(c[0], c[2], stroke="#9a9a9a", stroke_width=0.8))
                out.append(_line(c[1], c[3], stroke="#9a9a9a", stroke_width=0.8))
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
        if lb.get("kind") == "stairs":
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


# ---------------------------------------------------------------------------
# axonometric view (clean architectural isometric)
# ---------------------------------------------------------------------------
ISO_ELEVATION = math.degrees(math.atan(1 / math.sqrt(2)))   # 35.264..: true isometric

AXO_STYLE = {
    "line": "#1f1f1f", "line_w": 1.0,
    "floor_top": "#f3f1ec", "floor_side_a": "#d7d3ca", "floor_side_b": "#c6c1b7",
    "wall_a": "#e4e1da", "wall_b": "#cdc8be", "wall_top": "#f7f6f2",
    "stairs_top": "#ebe7de", "tread": "#8f8a80", "rail": "#dcd8cf",
    "path": "#d9482b", "label": "#111111", "muted": "#6f6f6f", "leader": "#1f1f1f",
    "ring": "#f7f5f0", "void": "#4a4a4a",
}


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


class _Faces:
    """Collects svg fragments with a depth key per layer; layers are emitted in
    order, each sorted far-to-near (painter's algorithm)."""

    def __init__(self):
        self.layers: dict[str, list[tuple[float, str]]] = {}

    def add(self, layer: str, depth: float, svg: str) -> None:
        self.layers.setdefault(layer, []).append((depth, svg))

    def emit(self, order: list[str]) -> list[str]:
        out = []
        for name in order:
            items = self.layers.get(name)
            if not items:
                continue
            items.sort(key=lambda t: t[0])
            out.append(f'<g id="{name}">')
            out += [svg for _, svg in items]
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
                  show_path: bool = True, draw_walls: bool = False) -> None:
    """Clean isometric building diagram: floor slabs, cut-away walls, stairs with
    treads and balustrades, the walk path, labels. Grouped per level for editing."""
    S = AXO_STYLE
    lw = S["line_w"]
    levels = {lv["id"]: lv for lv in diagram["levels"]}
    faces = _Faces()
    ppm = view.px_per_m

    def lvl_off(level_id: int) -> float:
        return view.explode_m * max(level_id, 0)

    def run_layer(r: dict) -> str:
        if r["kind"] == "stairs":
            lo = min(r["level_from"], r["level_to"]) if r["level_from"] >= 0 and r["level_to"] >= 0 else max(r["level_from"], r["level_to"], 0)
            return f"stairs-L{lo}"
        return f"level-L{r['level']}"

    all_pts = []

    def P(pts3, off=0.0):
        p2, d = view.project(pts3, off)
        all_pts.append(p2)
        return p2 * ppm, d

    def poly(p2, fill, stroke=S["line"], w=lw, extra=""):
        return _poly(p2, fill=fill, stroke=stroke, stroke_width=w, stroke_linejoin="round") if not extra else \
            _poly(p2, fill=fill, stroke=stroke, stroke_width=w, stroke_linejoin="round", **extra)

    def slab_faces(layer: str, quad_top: np.ndarray, off0: float, off1: float, top_fill: str, thickness: float):
        """Top face + the visible side faces of a 4-point slab (l0,l1,r1,r0)."""
        q = np.asarray(quad_top, float).copy()
        offs = np.array([off0, off1, off1, off0])
        q[:, 2] += offs
        bottom = q.copy(); bottom[:, 2] -= thickness
        p2, d = view.project(q); all_pts.append(p2)
        faces.add(layer, _quad_depth(d) - 0.03, poly(p2 * ppm, top_fill))
        # sides: 4 edges; draw only those facing the viewer
        _, _, wv = view.axes()
        for k in range(4):
            k2 = (k + 1) % 4
            e = q[k2, :2] - q[k, :2]
            if np.linalg.norm(e) < 1e-6:
                continue
            n = np.array([e[1], -e[0]])          # outward for a CCW (seen from above) quad
            n /= np.linalg.norm(n)
            if np.array([n[0], n[1], 0.0]) @ wv <= 0:
                continue
            side = np.array([q[k], q[k2], bottom[k2], bottom[k]])
            p2s, ds = view.project(side)
            tone = _side_tone(n, view, S["floor_side_a"], S["floor_side_b"])
            faces.add(layer, _quad_depth(ds) - 0.02, poly(p2s * ppm, tone))

    for r in diagram["runs"]:
        layer = run_layer(r)
        quads = [np.array(q, float) for q in (r.get("quads") or [])]
        if not quads:
            L = np.array(r["left_boundary"]); R = np.array(r["right_boundary"])
            quads = [np.array([L[j], L[j + 1], R[j + 1], R[j]]) for j in range(len(L) - 1)]
        n = len(quads)
        if r["kind"] == "stairs":
            o0, o1 = lvl_off(r["level_from"]), lvl_off(r["level_to"])
            z_lo = min(q[:, 2].min() for q in quads); z_hi = max(q[:, 2].max() for q in quads)
            span = max(z_hi - z_lo, 1e-6)

            def off_at(z):   # exploded offset interpolated by height along the flight
                t = (z - z_lo) / span if o1 >= o0 else 1.0 - (z - z_lo) / span
                lo_, hi_ = (o0, o1) if o1 >= o0 else (o1, o0)
                return lo_ + (hi_ - lo_) * t
            for quad in quads:
                offs = np.array([off_at(z) for z in quad[:, 2]])
                slab_faces(layer, quad, offs[0], offs[1], S["stairs_top"], slab) if np.allclose(offs, offs[0]) else \
                    slab_faces(layer, quad, offs[0], offs[1], S["stairs_top"], slab)
                # treads on sloped quads along the walking direction l0 -> l1
                if abs(quad[1, 2] - quad[0, 2]) > 1e-6:
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
                for z, ly, oz in ring_levels:
                    for q in at.get("ring", []):
                        quad = np.c_[np.array(q), np.full(4, z)]
                        slab_faces(ly, quad, oz, oz, S["ring"], slab)
                bot = np.c_[corners, np.full(4, z_lo2 + oz0)]
                top = np.c_[corners, np.full(4, z_hi2 + oz1)]
                pb, db = view.project(bot); pt, dt = view.project(top)
                all_pts.append(pb); all_pts.append(pt)
                for k in range(4):
                    k2 = (k + 1) % 4
                    faces.add(layer, float(db[[k, k2]].mean()) + 0.05, _line(pb[k] * ppm, pb[k2] * ppm, stroke=S["void"], stroke_width=lw * 0.9))
                    faces.add(layer, float(dt[[k, k2]].mean()) + 0.05, _line(pt[k] * ppm, pt[k2] * ppm, stroke=S["void"], stroke_width=lw * 0.9))
                    faces.add(layer, float((db[k] + dt[k]) / 2) + 0.05, _line(pb[k] * ppm, pt[k] * ppm, stroke=S["void"], stroke_width=lw * 0.9))
            for w in (r["walls"] if draw_walls else []):   # balustrades
                a, b = np.array(w["a"]), np.array(w["b"])
                quad = np.array([[*a, w["z0"] + o0], [*b, w["z1"] + o1], [*b, w["z1"] + o1 + stairs_wall_height],
                                 [*a, w["z0"] + o0 + stairs_wall_height]])
                p2, d = P(quad)
                faces.add(layer, _quad_depth(d), poly(p2, S["rail"], extra={"fill_opacity": 0.9}))
        else:
            off = lvl_off(r["level"])
            for quad in quads:
                slab_faces(layer, quad, off, off, S["floor_top"], slab)
            for w in (r["walls"] if draw_walls else []):
                a, b = np.array(w["a"]), np.array(w["b"])
                z0 = w["z0"]
                e = b - a
                nrm = np.array([-e[1], e[0]]); nrm /= max(np.linalg.norm(nrm), 1e-9)
                if w["side"] == "right":
                    nrm = -nrm            # points into the corridor
                for t0, t1 in w["solid"]:
                    pa = a + (b - a) * t0; pb = a + (b - a) * t1
                    quad = np.array([[*pa, z0], [*pb, z0], [*pb, z0 + cutaway_height], [*pa, z0 + cutaway_height]])
                    p2, d = P(quad, off)
                    tone = _side_tone(nrm, view, S["wall_a"], S["wall_b"]) if shade else S["wall_a"]
                    faces.add(layer, _quad_depth(d), poly(p2, tone))
                    th = 0.1
                    top = np.array([[*(pa - nrm * th), z0 + cutaway_height], [*(pb - nrm * th), z0 + cutaway_height],
                                    [*pb, z0 + cutaway_height], [*pa, z0 + cutaway_height]])
                    p2t, dt = P(top, off)
                    faces.add(layer, _quad_depth(dt) + 0.001, poly(p2t, S["wall_top"]))
                for t0, t1 in w["openings"]:
                    pa = a + (b - a) * t0; pb = a + (b - a) * t1
                    p2, d = P(np.array([[*pa, z0], [*pb, z0]]), off)
                    faces.add(layer, _quad_depth(d), _line(p2[0], p2[1], stroke=S["muted"], stroke_width=lw,
                                                            stroke_dasharray="3 3"))
    # path, drawn per flat stretch (the flights of stairs show the route themselves)
    if show_path:
        pts = np.array(diagram["path"]["points"])
        lv = diagram["path"].get("level") or [0] * len(pts)
        chunk = []
        for p, l in list(zip(pts, lv)) + [(None, None)]:
            if p is not None and l >= 0:
                chunk.append((p, l))
                continue
            if len(chunk) >= 2:
                pp = np.array([c[0] for c in chunk]); offs = np.array([lvl_off(c[1]) for c in chunk])
                pp[:, 2] = pp[:, 2] - diagram.get("camera_height_m", 1.5) + 0.06 + offs
                p2, d = P(pp)
                faces.add("path", 0.0, _polyline(p2, fill="none", stroke=S["path"], stroke_width=lw * 1.8,
                                                 stroke_dasharray="6 4", stroke_linejoin="round", stroke_linecap="round"))
            chunk = []
    # labels
    fs = max(11.0, 0.4 * ppm)
    for lb in diagram.get("labels", []):
        p = np.array(lb["xyz"], float)
        off = lvl_off(lb.get("level", 0)) if lb.get("level", 0) >= 0 else 0.0
        base = p.copy(); base[2] = lb.get("floor_z", p[2])
        top = base.copy(); top[2] += (cutaway_height if draw_walls else 0.0) + 1.0
        pb, _ = P(base, off); pt, _ = P(top, off)
        pb, pt = pb[0], pt[0]
        is_stairs = lb.get("kind") == "stairs"
        col = S["muted"] if is_stairs else S["label"]
        faces.add("labels", 0.0, _line(pb, pt, stroke=col, stroke_width=lw))
        if not is_stairs:
            faces.add("labels", 0.0, f'<circle cx="{_fmt(pb[0])}" cy="{_fmt(pb[1])}" r="{_fmt(fs * 0.25)}" fill="{S["path"]}"/>')
        faces.add("labels", 0.0, _text((pt[0] + 5, pt[1] - 3), lb["text"], fs * (0.8 if is_stairs else 1.0), fill=col,
                                       font_weight="500" if is_stairs else "700"))
    # level tags: at the left-most point of each level's floor geometry
    per_level: dict[int, list[np.ndarray]] = {}
    for r in diagram["runs"]:
        if r["kind"] != "flat":
            continue
        pts3 = np.vstack([np.array(r["left_boundary"]), np.array(r["right_boundary"])])
        p2, _ = view.project(pts3, lvl_off(r["level"]))
        per_level.setdefault(r["level"], []).append(p2 * ppm)
    for lid, chunks in per_level.items():
        lv_ = levels.get(lid)
        if lv_ is None:
            continue
        A_ = np.vstack(chunks)
        k = int(np.argmin(A_[:, 0]))          # left-most point of the level, tag below-left of it
        faces.add("labels", 0.0, _text((A_[k, 0] - fs * 0.3, A_[k, 1] + fs * 1.6), lv_["name"], fs * 0.85,
                                       fill=S["muted"], text_anchor="end", font_weight="500"))
        all_pts.append(np.array([[A_[k, 0] / ppm - fs * 3.5 / ppm, A_[k, 1] / ppm + fs * 2 / ppm]]))

    A = np.vstack(all_pts) * ppm
    pad = fs * 3
    x0, y0 = A.min(axis=0) - pad; x1, y1 = A.max(axis=0) + pad
    top_extra = fs * 2.5 if title else fs
    W, H = x1 - x0, (y1 - y0) + top_extra
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W:.0f}" height="{H:.0f}" '
           f'viewBox="{x0:.1f} {y0 - top_extra:.1f} {W:.1f} {H:.1f}" font-family="Helvetica Neue, Helvetica, Arial, sans-serif">',
           f'<rect x="{x0:.1f}" y="{y0 - top_extra:.1f}" width="{W:.1f}" height="{H:.1f}" fill="white"/>']
    level_ids = sorted(levels)
    order = []
    for lid in level_ids:
        order += [f"level-L{lid}", f"stairs-L{lid}"]
    order += [k for k in faces.layers if k not in order and k not in ("path", "labels")]
    order += ["path", "labels"]
    out += faces.emit(order)
    if title:
        out.append(_text((x0 + 12, y0 - fs * 1.0), title, fs * 1.3, fill=S["label"], font_weight="700"))
    out.append("</svg>")
    Path(path).write_text("\n".join(out), encoding="utf-8")
