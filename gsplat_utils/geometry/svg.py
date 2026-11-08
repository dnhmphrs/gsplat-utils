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
                   px_per_m: float = 40.0, margin_m: float = 1.5, show_path: bool = True, title: str | None = None) -> None:
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
        fill = PALETTE["stairs"] if r["kind"] == "stairs" else PALETTE["floor"]
        pts = np.array([P(p) for p in r["outline"]])
        out.append(_poly(pts, fill=fill, stroke="none", fill_opacity=0.95))
        if r["kind"] == "stairs":
            out.extend(_tread_lines(r, P, px_per_m))
    # walls
    for r in sel:
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
    # path
    if show_path:
        pts = np.array(diagram["path"]["points"])
        if level_id is not None:
            lv = diagram["path"].get("level")
            mask = np.array([l == level_id or l == -1 for l in lv]) if lv else np.ones(len(pts), bool)
        else:
            mask = np.ones(len(pts), bool)
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
# axonometric view
# ---------------------------------------------------------------------------
@dataclass
class AxoView:
    azimuth_deg: float = -55.0
    elevation_deg: float = 35.0
    explode_m: float = 0.0          # extra vertical spacing per level
    px_per_m: float = 36.0

    def axes(self):
        az = math.radians(self.azimuth_deg); el = math.radians(self.elevation_deg)
        u = np.array([-math.sin(az), math.cos(az), 0.0])                             # screen right
        v = np.array([-math.sin(el) * math.cos(az), -math.sin(el) * math.sin(az), math.cos(el)])  # screen up
        w = np.array([math.cos(el) * math.cos(az), math.cos(el) * math.sin(az), math.sin(el)])    # toward viewer
        return u, v, w

    def project(self, p: np.ndarray, level_offset: float = 0.0):
        u, v, w = self.axes()
        p = np.asarray(p, float).reshape(-1, 3).copy()
        p[:, 2] += level_offset
        return np.c_[p @ u, -(p @ v)], p @ w


def write_axo_svg(path, diagram: dict, view: AxoView, cutaway_height: float = 1.2, title: str | None = None,
                  shade: bool = True, stairs_wall_height: float = 0.9) -> None:
    """Axonometric 'exploded' building diagram: floor ribbons, low cut-away walls, stairs, path, labels."""
    levels = {lv["id"]: lv for lv in diagram["levels"]}
    faces: list[tuple[float, str]] = []     # (depth, svg)

    def lvl_off(level_id: int) -> float:
        return view.explode_m * max(level_id, 0)

    u, v, wv = view.axes()
    light = np.array([0.3, -0.5, 0.8]); light /= np.linalg.norm(light)

    def shade_color(normal: np.ndarray, base=(0.86, 0.86, 0.84)) -> str:
        if not shade:
            return "#b0aca3"
        lam = 0.55 + 0.45 * max(0.0, float(normal @ light))
        r, g, b = [int(255 * min(1.0, c * lam)) for c in base]
        return f"#{r:02x}{g:02x}{b:02x}"

    all_pts = []
    for r in diagram["runs"]:
        off = lvl_off(r["level"] if r["kind"] == "flat" else min(r["level_from"], r["level_to"]) if r["level_from"] >= 0 else 0)
        L = np.array(r["left_boundary"]); R = np.array(r["right_boundary"])
        n = len(L) - 1
        for j in range(n):
            quad = np.array([L[j], L[j + 1], R[j + 1], R[j]])
            if r["kind"] == "stairs":
                # interpolate exploded offset along the flight
                tq = np.array([j / n, (j + 1) / n, (j + 1) / n, j / n])
                o0, o1 = lvl_off(r["level_from"]), lvl_off(r["level_to"])
                quad = quad.copy(); quad[:, 2] += o0 + (o1 - o0) * tq
                p2, d = view.project(quad)
                faces.append((float(d.mean()) - 0.01, _poly(p2 * view.px_per_m, fill=PALETTE["stairs"], stroke="#8c8577", stroke_width=1)))
                # treads
                seg_len = np.linalg.norm(L[j + 1, :2] - L[j, :2]); m = int(seg_len / 0.3)
                for k in range(1, m):
                    t = k / m
                    a = quad[0] + (quad[1] - quad[0]) * t; b = quad[3] + (quad[2] - quad[3]) * t
                    pa, da = view.project(a); pb, _ = view.project(b)
                    faces.append((float(da[0]), _line(pa[0] * view.px_per_m, pb[0] * view.px_per_m, stroke="#8c8577", stroke_width=1)))
            else:
                p2, d = view.project(quad, off)
                faces.append((float(d.mean()) - 0.02, _poly(p2 * view.px_per_m, fill=PALETTE["floor"], stroke=PALETTE["floor_stroke"], stroke_width=0.8)))
            all_pts.append(p2)
        if r["kind"] == "stairs":
            o0, o1 = lvl_off(r["level_from"]), lvl_off(r["level_to"])
            for w in r["walls"]:
                a, b = np.array(w["a"]), np.array(w["b"])
                quad = np.array([[*a, w["z0"] + o0], [*b, w["z1"] + o1], [*b, w["z1"] + o1 + stairs_wall_height],
                                 [*a, w["z0"] + o0 + stairs_wall_height]])
                p2, d = view.project(quad)
                faces.append((float(d.mean()), _poly(p2 * view.px_per_m, fill="#c9c3b6", fill_opacity=0.85,
                                                     stroke=PALETTE["wall"], stroke_width=0.8)))
                all_pts.append(p2)
        # walls (flat runs only): vertical faces of height cutaway_height
        if r["kind"] == "flat":
            for w in r["walls"]:
                a, b = np.array(w["a"]), np.array(w["b"])
                z0 = w["z0"]
                for t0, t1 in w["solid"]:
                    pa = a + (b - a) * t0; pb = a + (b - a) * t1
                    quad = np.array([[*pa, z0], [*pb, z0], [*pb, z0 + cutaway_height], [*pa, z0 + cutaway_height]])
                    e = pb - pa
                    normal = np.array([-e[1], e[0], 0.0]); normal /= max(np.linalg.norm(normal), 1e-9)
                    if w["side"] == "right":
                        normal = -normal
                    # the wall faces the corridor (inwards); shade by |n . light|
                    p2, d = view.project(quad, off)
                    faces.append((float(d.mean()), _poly(p2 * view.px_per_m, fill=shade_color(np.abs(normal)), stroke=PALETTE["wall"], stroke_width=0.9)))
                    # top edge darker for crispness
                    faces.append((float(d.mean()) + 0.005, _line(p2[3] * view.px_per_m, p2[2] * view.px_per_m, stroke=PALETTE["wall"], stroke_width=1.6)))
                    all_pts.append(p2)
                for t0, t1 in w["openings"]:
                    pa = a + (b - a) * t0; pb = a + (b - a) * t1
                    quad = np.array([[*pa, z0], [*pb, z0]])
                    p2, d = view.project(quad, off)
                    faces.append((float(d.mean()), _line(p2[0] * view.px_per_m, p2[1] * view.px_per_m, stroke=PALETTE["opening"], stroke_width=1.2, stroke_dasharray="4 3")))
    # path
    pts = np.array(diagram["path"]["points"])
    lv = diagram["path"].get("level") or [0] * len(pts)
    offs = np.array([lvl_off(l) if l >= 0 else 0.0 for l in lv])
    # stairs samples: interpolate the offset between neighbours
    if view.explode_m:
        idx = np.arange(len(lv)); good = np.array([l >= 0 for l in lv])
        if good.any() and (~good).any():
            offs[~good] = np.interp(idx[~good], idx[good], offs[good])
    pp = pts.copy(); pp[:, 2] = pp[:, 2] - diagram.get("camera_height_m", 1.5) + 0.05 + offs
    p2, d = view.project(pp)
    faces.append((float(d.max()) + 0.5, _polyline(p2 * view.px_per_m, fill="none", stroke=PALETTE["path"], stroke_width=2.2,
                                                   stroke_dasharray="7 4", stroke_linejoin="round")))
    all_pts.append(p2)
    # labels
    fs = max(11.0, 0.4 * view.px_per_m)
    label_svgs = []
    for lb in diagram.get("labels", []):
        p = np.array(lb["xyz"], float)
        off = lvl_off(lb.get("level", 0)) if lb.get("level", 0) >= 0 else 0.0
        base = p.copy(); base[2] = lb.get("floor_z", p[2])
        top = base.copy(); top[2] += cutaway_height + 0.8
        pb, _ = view.project(base, off); pt, _ = view.project(top, off)
        pb = pb[0] * view.px_per_m; pt = pt[0] * view.px_per_m
        is_stairs = lb.get("kind") == "stairs"
        label_svgs.append(_line(pb, pt, stroke=PALETTE["muted"] if is_stairs else PALETTE["label"], stroke_width=1))
        if not is_stairs:
            label_svgs.append(f'<circle cx="{_fmt(pb[0])}" cy="{_fmt(pb[1])}" r="{_fmt(fs * 0.28)}" fill="{PALETTE["label_dot"]}"/>')
        label_svgs.append(_text((pt[0] + 4, pt[1] - 2), lb["text"], fs * (0.8 if is_stairs else 1.0),
                                fill=PALETTE["muted"] if is_stairs else PALETTE["label"], font_weight="700" if not is_stairs else "500"))
        all_pts.append(np.array([pt / view.px_per_m, pb / view.px_per_m]))
    # level name tags at the start of each level's first run
    for r in diagram["runs"]:
        if r["kind"] != "flat":
            continue
        lv_ = levels.get(r["level"])
        if lv_ is None or lv_.get("_tagged"):
            continue
        lv_["_tagged"] = True
        L = np.array(r["left_boundary"])
        p2, _ = view.project(L[0], lvl_off(r["level"]))
        q = p2[0] * view.px_per_m
        label_svgs.append(_text((q[0] - fs * 0.4, q[1] + fs * 1.3), lv_["name"], fs * 0.9, fill=PALETTE["muted"], text_anchor="end"))

    A = np.vstack(all_pts) * view.px_per_m
    pad = fs * 3
    x0, y0 = A.min(axis=0) - pad; x1, y1 = A.max(axis=0) + pad
    W, H = x1 - x0, y1 - y0 + fs * 2
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W:.0f}" height="{H:.0f}" viewBox="{x0:.1f} {y0 - fs * 2:.1f} {W:.1f} {H:.1f}" '
           f'font-family="Helvetica, Arial, sans-serif">', f'<rect x="{x0:.1f}" y="{y0 - fs * 2:.1f}" width="{W:.1f}" height="{H:.1f}" fill="white"/>']
    faces.sort(key=lambda f: f[0])
    out += [s for _, s in faces]
    out += label_svgs
    if title:
        out.append(_text((x0 + 12, y0 - fs * 0.6), title, fs * 1.3, fill=PALETTE["text"], font_weight="700"))
    out.append("</svg>")
    Path(path).write_text("\n".join(out), encoding="utf-8")
