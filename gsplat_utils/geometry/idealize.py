"""`gsu idealize`: a clean schematic from a measured diagram.json.

The measured diagram keeps every wobble of the walk. For a sign you want the
*idea* of the building: the same elements in the same arrangement, but drawn as
regular shapes on a grid: square open-well stairs with a full set of flights and
a gallery, a straight corridor of standard width, halls as tidy rectangles, a
centreline route. This module turns diagram.json into such a schematic (same
schema, so the plan/isometric writers draw it) plus a GLB.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..common import log, read_json, write_json
from ..paths import Workspace
from .mesh import Mesh, write_glb, write_obj
from .stairs import enclosed_area
from .svg import AxoView, write_axo_svg, write_plan_svg


@dataclass
class IdealConfig:
    grid: float = 0.5                 # everything snaps to this (m)
    corridor_width: float | None = None   # None = same as the flight/landing width
    flight_width: float = 1.25
    step_rise: float = 0.17           # riser height of the drawn steps (m)
    step_run: float = 0.28            # tread depth of straight (non-well) flights (m)
    void_mode: str = "median"         # "median": one square well for all stairs; "measured": each stair its own (rounded)
    flights: int | None = None        # flights per storey around a well (None = from the storey height)
    gallery: str = "traversed"        # "traversed" (only the walked stretch) | "upper" | "both" | "none"
    void_lines: bool = False          # wireframe box of the open well
    iso_elevation: float = 25.0
    mid_steps: list = field(default_factory=list)   # [(fraction_along_corridor, n_steps)] short flights inside corridors
    stair_sense: list = field(default_factory=list)  # per stair in route order: "ccw" | "cw" | "auto" (measured)
    exit_after: list = field(default_factory=list)   # per stair: flights the route takes before leaving (0 = all)
    stair_flights: list = field(default_factory=list)  # per stair: flights per storey (0 = from the height / --flights)
    continue_flights: list = field(default_factory=list)  # per stair: extra flights drawn beyond the route's exit
    top_landing: str = "straight"     # "straight": the corridor continues the walkway along the exit side of the well,
                                      # flights ordered from where you enter; "inline": the last flight itself runs in
                                      # the corridor's direction; "L": the corridor meets the near corner and the
                                      # landing runs the full side
    hidden_lines: str = "stairs"      # hidden edges as light dashed lines: "stairs" | "all" | "none"
    foyer_ext: float | None = 7.0     # how far a base-slab foyer extends beyond its stair block (None = from the measured hall)
    path_morse: str | None = None     # lay this text in Morse along the route instead of plain dashes
    morse_unit_px: float = 4.0        # length of a Morse dot on the page (px at 36 px/m)

    def __post_init__(self):
        if self.corridor_width is None:
            self.corridor_width = self.flight_width
    entry_steps: int = 0              # straight flight between a lower entrance foyer and the first stair
    foyer: str = "base"               # "base": a start/end hall is a slab as deep as its stair block, flush against it,
                                      # the Start/End in line with the first flight; "beside": the measured rectangle
    theme: str = "light"
    all_themes: bool = True           # also write styles/iso_<theme>.svg for every theme
    label_style: str = "key"          # numbered markers + key, or "inline"
    start_label: str = "Start"
    end_label: str = "End"
    flight_rise: float = 1.5          # typical rise of one flight between landings (m)
    void_side: float | None = None    # None = median of the measured wells, snapped
    hall_min: float = 4.0
    slab: float = 0.15
    explode: float = 0.0
    title: str | None = None
    show_path: bool = True
    iso_azimuth: float = -60.0        # view direction; -60 keeps the start/end markers clear of the galleries


def _snap(v: float, g: float) -> float:
    return float(np.round(v / g) * g)


def _rot_sense(xy: np.ndarray) -> int:
    """+1 counter-clockwise, -1 clockwise (plan view, y up) for the walked loop."""
    if len(xy) < 3:
        return 1
    x, y = xy[:, 0], xy[:, 1]
    a = 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))
    return 1 if a >= 0 else -1


@dataclass
class Element:
    kind: str                      # "hall" | "corridor" | "stairs"
    run: dict
    level: int = 0
    level_from: int = -1
    level_to: int = -1
    centre: np.ndarray = field(default_factory=lambda: np.zeros(2))
    size: np.ndarray = field(default_factory=lambda: np.zeros(2))   # halls / stair blocks
    axis: str = "x"                # corridors
    a: np.ndarray = field(default_factory=lambda: np.zeros(2))      # corridor start/end (centreline)
    b: np.ndarray = field(default_factory=lambda: np.zeros(2))
    sense: int = 1                 # stairs rotation sense
    void: float = 3.5              # kept for compatibility: mean well side
    void_size: np.ndarray = field(default_factory=lambda: np.array([3.5, 3.5]))
    m_centre: np.ndarray | None = None    # measured void centre
    m_entry: np.ndarray | None = None     # measured first/last stairs sample (xy)
    m_exit: np.ndarray | None = None
    base_of: "Element | None" = None      # hall that is the slab flush against this staircase
    term: np.ndarray | None = None        # Start/End point on a base slab
    exit_after: int = 0                   # stairs: route leaves after this many flights (0 = all)
    exit_from: "Element | None" = None    # hall entered part-way down/up a stair
    flights: int = 0                      # stairs: flights per storey (0 = default)
    cont: int = 0                         # stairs: extra flights drawn beyond the exit
    entry_pt: np.ndarray | None = None    # hall at a stair exit: point on its edge where the route comes in
    n_fl: int = 3                         # stairs: flights per storey actually drawn


def _elements(d: dict, cfg: IdealConfig) -> list[Element]:
    els = []
    sides = []
    for r in d["runs"]:
        if r["kind"] == "stairs" and r.get("atrium"):
            c = np.array(r["atrium"]["corners"])
            sides += [float(c[:, 0].max() - c[:, 0].min()), float(c[:, 1].max() - c[:, 1].min())]
    void = cfg.void_side or (_snap(float(np.median(sides)), cfg.grid) if sides else 3.5)
    void = max(void, 2.0)
    pts = np.array(d["path"]["points"]); lv = np.array(d["path"]["level"])
    floor = {lv_["id"]: _snap(lv_["floor_z"] - d["levels"][0]["floor_z"], 0.25) for lv_ in d["levels"]}
    for r in d["runs"]:
        q = np.array([p for quad in r["quads"] for p in quad])[:, :2]
        if r["kind"] == "stairs":
            e = Element("stairs", r, level_from=r["level_from"], level_to=r["level_to"], void=void)
            e.m_entry = np.array(r["pieces"][0]["p0"][:2], float)
            e.m_exit = np.array(r["pieces"][-1]["p1"][:2], float)
            e.void_size = np.array([void, void])
            if r.get("atrium"):
                c = np.array(r["atrium"]["corners"])
                e.centre = c.mean(axis=0)
                e.m_centre = c.mean(axis=0)
                if cfg.void_mode == "measured" and not cfg.void_side:
                    sz = c.max(axis=0) - c.min(axis=0)
                    e.void_size = np.maximum(np.array([_snap(sz[0], cfg.grid), _snap(sz[1], cfg.grid)]), 1.0)
                    e.void = float(e.void_size.mean())
                # rotation sense from the walked samples of this run
                i0 = int(min(p["i0"] for p in r["pieces"])) if "i0" in r["pieces"][0] else None
                seg = pts[(lv == -1)]
                # samples between the neighbouring flat runs: approximate by nearest to the void
                near = seg[np.linalg.norm(seg[:, :2] - e.centre, axis=1) < void * 1.5]
                e.sense = _rot_sense(near[:, :2]) if len(near) >= 3 else 1
            else:
                e.centre = q.mean(axis=0)
            e.size = e.void_size + 2 * cfg.flight_width
            k_st = sum(1 for x in els if x.kind == "stairs")
            if k_st < len(cfg.stair_sense) and cfg.stair_sense[k_st] in ("cw", "ccw"):
                e.sense = -1 if cfg.stair_sense[k_st] == "cw" else 1
            if k_st < len(cfg.exit_after):
                e.exit_after = int(cfg.exit_after[k_st])
            if k_st < len(cfg.stair_flights):
                e.flights = int(cfg.stair_flights[k_st])
            if k_st < len(cfg.continue_flights):
                e.cont = int(cfg.continue_flights[k_st])
            e.n_fl = _n_flights(e, floor.get(e.level_from, 0.0), floor.get(e.level_to, 0.0), cfg)
            els.append(e)
        elif r.get("hall"):
            e = Element("hall", r, level=r["level"])
            lo, hi = q.min(axis=0), q.max(axis=0)
            e.centre = 0.5 * (lo + hi)
            e.size = np.maximum(np.array([_snap(hi[0] - lo[0], 1.0), _snap(hi[1] - lo[1], 1.0)]), cfg.hall_min)
            els.append(e)
        else:
            p0 = np.array(r["pieces"][0]["p0"][:2]); p1 = np.array(r["pieces"][-1]["p1"][:2])
            e = Element("corridor", r, level=r["level"])
            dv = p1 - p0
            e.axis = "x" if abs(dv[0]) >= abs(dv[1]) else "y"
            e.a, e.b = p0, p1
            e.centre = 0.5 * (p0 + p1)
            els.append(e)
    return els


def _exit_dir(els: list[Element], i: int) -> np.ndarray | None:
    """Direction of the corridor that leaves element i, if the next element is one."""
    if i + 1 < len(els) and els[i + 1].kind == "corridor":
        nb = els[i + 1]
        dv = np.asarray(nb.b, float) - np.asarray(nb.a, float)
        if nb.axis == "y":
            return np.array([0.0, 1.0 if dv[1] >= 0 else -1.0])
        return np.array([1.0 if dv[0] >= 0 else -1.0, 0.0])
    return None


def _layout(els: list[Element], cfg: IdealConfig, terminals: tuple[np.ndarray, np.ndarray] | None = None) -> None:
    """Snap centres to the grid, straighten corridors, make neighbours abut.
    With cfg.foyer == "base", a hall at the start/end of the walk that sits next
    to a staircase becomes a slab as deep as the stair block, flush against it on
    the Start/End side; both ends get the same length for balance."""
    g = cfg.grid
    for e in els:
        e.centre = np.array([_snap(e.centre[0], g), _snap(e.centre[1], g)])
    for e in els:
        if e.kind == "corridor":
            if e.axis == "y":
                x = _snap(0.5 * (e.a[0] + e.b[0]), g)
                e.a = np.array([x, _snap(e.a[1], g)]); e.b = np.array([x, _snap(e.b[1], g)])
            else:
                y = _snap(0.5 * (e.a[1] + e.b[1]), g)
                e.a = np.array([_snap(e.a[0], g), y]); e.b = np.array([_snap(e.b[0], g), y])
    # a stair entered or left by a corridor: slide the block along the shared side so the
    # corridor is flush with the block edge at the start (entry) / end (exit) of the landing
    # side in the walking sense; the landing then runs the full side and the join is clean
    def corners_of_side(side: int, sense_: int):
        ccw = {0: (0, 1), 1: (1, 2), 2: (2, 3), 3: (3, 0)}[side]
        return ccw if sense_ > 0 else (ccw[1], ccw[0])
    corner_sign = {0: (-1, -1), 1: (1, -1), 2: (1, 1), 3: (-1, 1)}
    for i, e in enumerate(els):
        if e.kind != "stairs":
            continue
        for nb, role in ((els[i - 1] if i > 0 else None, "entry"), (els[i + 1] if i + 1 < len(els) else None, "exit")):
            if nb is None or nb.kind != "corridor":
                continue
            cp = nb.b if role == "entry" else nb.a            # corridor end touching the block
            far = nb.a if role == "entry" else nb.b           # the other end: tells which side it comes from
            d = far - e.centre
            if nb.axis == "y":
                side = 2 if d[1] > 0 else 0
            else:
                side = 1 if d[0] > 0 else 3
            c0, c1 = corners_of_side(side, e.sense)
            if cfg.top_landing == "L":
                corner = c0 if role == "entry" else c1        # arrive at the near corner, walk the side
            else:
                corner = c1 if role == "entry" else c0        # in line with the flight: no walk along the side
            k = 1 if side in (1, 3) else 0                    # axis along the side
            sgn = corner_sign[corner][k]
            half_c = cfg.corridor_width / 2
            # corridor outer edge on the corner's side == block edge on that side
            e.centre[k] = _snap(cp[k] + sgn * half_c - sgn * e.size[k] / 2, g)
    # start/end halls as slabs flush against their staircase: after the blocks have found
    # their final place against the corridor, so the slab really sits under the stair
    if cfg.foyer == "base" and terminals is not None:
        bases = []
        for i, e in enumerate(els):
            if e.kind != "hall" or i not in (0, len(els) - 1):
                continue
            nb = els[i + 1] if i == 0 and len(els) > 1 else els[i - 1] if i == len(els) - 1 and len(els) > 1 else None
            if nb is None or nb.kind != "stairs":
                continue
            if i == len(els) - 1 and nb.exit_after:
                continue        # the route leaves the stair part-way: the hall sits beside it, not under it
            term = terminals[0] if i == 0 else terminals[1]
            d = term - nb.centre
            k = int(np.argmax(np.abs(d)))
            sign = 1 if d[k] >= 0 else -1
            edge = nb.centre[k] + sign * nb.size[k] / 2
            ext = cfg.foyer_ext if cfg.foyer_ext else max(cfg.hall_min, abs(term[k] - edge) + 2.0)
            bases.append((e, nb, k, sign, ext))
        if bases:
            ext = _snap(max(b[4] for b in bases), 1.0)      # same extension on both ends: balance
            w = cfg.flight_width
            for e, nb, k, sign, _ in bases:
                # the slab is as deep as the stair block, runs `ext` out from the block face and one
                # landing square into it, so the foyer ends exactly where the first flight starts
                face = nb.centre[k] + sign * nb.size[k] / 2
                e.size = nb.size.copy(); e.size[k] = ext + w
                e.centre = nb.centre.copy(); e.centre[k] = face + sign * (ext - w) / 2
                e.base_of = nb
                # the Start/End point sits 1.5 m in from the far end of the slab, clear of the bridge
                t = nb.centre.copy(); t[k] = face + sign * max(ext - 1.5, ext / 2)
                if e is els[0]:
                    # the Start sits in line with the foot of the first flight, so the route walks
                    # straight onto the stair instead of along the side of the well
                    side_pt = nb.centre.copy(); side_pt[k] = nb.centre[k] + sign * nb.size[k]
                    t[1 - k] = _corner_xy(nb, _first_corner(nb, side_pt, _exit_dir(els, els.index(nb)), cfg), cfg.flight_width)[1 - k]
                e.term = t
    # halls next to a stair block must not overlap it: push them out along the main offset axis
    for i, e in enumerate(els):
        if e.kind != "hall" or e.base_of is not None:
            continue
        for j in (i - 1, i + 1):
            if 0 <= j < len(els) and els[j].kind == "stairs":
                s = els[j]
                lo_s, hi_s = s.centre - s.size / 2, s.centre + s.size / 2
                lo_e, hi_e = e.centre - e.size / 2, e.centre + e.size / 2
                overlap = np.minimum(hi_s, hi_e) - np.maximum(lo_s, lo_e)
                if np.all(overlap > 0):
                    d = e.centre - s.centre
                    k = int(np.argmax(np.abs(d)))              # push along the dominant offset axis
                    sign = 1 if d[k] >= 0 else -1
                    e.centre[k] = s.centre[k] + sign * (s.size[k] / 2 + e.size[k] / 2)
                    e.centre[k] = _snap(e.centre[k], g)
    # corridors run from the edge of one block to the edge of the next
    for i, e in enumerate(els):
        if e.kind != "corridor":
            continue
        prev_ = els[i - 1] if i > 0 else None
        next_ = els[i + 1] if i + 1 < len(els) else None
        k = 1 if e.axis == "y" else 0
        for nb, end in ((prev_, "a"), (next_, "b")):
            if nb is None or nb.kind == "corridor":
                continue
            p = getattr(e, end).copy()
            far = getattr(e, "b" if end == "a" else "a")
            lo, hi = nb.centre - nb.size / 2, nb.centre + nb.size / 2
            other = 1 - k
            # keep the corridor inside the block's extent on the other axis, flush with the edge
            # it arrives at (the far end of the corridor tells which side that is)
            p[other] = float(np.clip(p[other], lo[other] + cfg.corridor_width / 2, hi[other] - cfg.corridor_width / 2))
            p[k] = lo[k] if far[k] < nb.centre[k] else hi[k]
            setattr(e, end, p)
            # the other end shares the same coordinate on the other axis
            q = getattr(e, "b" if end == "a" else "a").copy(); q[other] = p[other]
            setattr(e, "b" if end == "a" else "a", q)


def _corner_xy(block: Element, corner: int, w: float) -> np.ndarray:
    """Centre of a corner landing square: 0=SW 1=SE 2=NE 3=NW."""
    sgn = {0: (-1, -1), 1: (1, -1), 2: (1, 1), 3: (-1, 1)}[corner]
    return block.centre + np.array(sgn, float) * (block.size / 2 - w / 2)


def _n_flights(e: Element, z0: float, z1: float, cfg: IdealConfig) -> int:
    n = int(e.flights) if e.flights else int(cfg.flights) if cfg.flights else int(np.clip(round(abs(z1 - z0) / cfg.flight_rise), 2, 4))
    return int(np.clip(n, 1, 4))


_WALK_CCW = {0: np.array([1.0, 0.0]), 1: np.array([0.0, 1.0]), 2: np.array([-1.0, 0.0]), 3: np.array([0.0, -1.0])}


def _first_corner(block: Element, arrive_from: np.ndarray, exit_dir: np.ndarray | None = None,
                  cfg: IdealConfig | None = None) -> int:
    """Corner the first flight leaves from. Normally you arrive on the side facing
    `arrive_from`, walk it in the turning sense and climb from its far corner. With
    `--landing inline` and a corridor leaving in `exit_dir`, the LAST flight is chosen to
    run in that direction (so the corridor continues it without a turn) and the first
    flight follows from the flight count."""
    if exit_dir is not None and cfg is not None and cfg.top_landing == "inline":
        d = np.asarray(exit_dir, float); d = d / max(np.linalg.norm(d), 1e-9)
        last_side = max(range(4), key=lambda sd: float((_WALK_CCW[sd] * block.sense) @ d))
        entry_side = (last_side - (block.n_fl - 1) * block.sense) % 4
        return _corners_of_side(entry_side, block.sense)[0]
    side = _side_of(block, np.asarray(arrive_from, float))
    return _corners_of_side(side, block.sense)[1]


def _rect(c: np.ndarray, size: np.ndarray, z: float) -> np.ndarray:
    hx, hy = size / 2
    return np.array([[c[0] - hx, c[1] - hy, z], [c[0] + hx, c[1] - hy, z], [c[0] + hx, c[1] + hy, z], [c[0] - hx, c[1] + hy, z]])


def _side_of(block: Element, p: np.ndarray) -> int:
    """0=south(-y) 1=east(+x) 2=north(+y) 3=west(-x): which side of the block faces point p."""
    d = p - block.centre
    if abs(d[0]) * block.size[1] > abs(d[1]) * block.size[0]:
        return 1 if d[0] > 0 else 3
    return 2 if d[1] > 0 else 0


def _stair_block(e: Element, entry: np.ndarray | None, exit_: np.ndarray | None, z0: float, z1: float,
                 cfg: IdealConfig, exit_dir: np.ndarray | None = None) -> tuple[list[np.ndarray], list[tuple[np.ndarray, np.ndarray]], dict, list[np.ndarray]]:
    """Flights (quads with z), landings (corner squares), atrium dict, route points.

    The flights climb around the well from the entry side to the exit side in the
    walked rotation sense; corners are quarter landings; the remaining sides are
    gallery. Returns (flight_quads, landing_quads, atrium, route_pts)."""
    w = cfg.flight_width
    hx, hy = e.void_size[0] / 2, e.void_size[1] / 2
    c = e.centre
    # corner squares (landings) indexed by corner: 0=SW 1=SE 2=NE 3=NW ; sides: 0=S 1=E 2=N 3=W
    corner_xy = {0: c + [-hx - w / 2, -hy - w / 2], 1: c + [hx + w / 2, -hy - w / 2],
                 2: c + [hx + w / 2, hy + w / 2], 3: c + [-hx - w / 2, hy + w / 2]}

    def side_strip(side: int, za: float, zb: float):
        # centreline start/end of the strip along the ccw walking direction and its outward normal
        if side == 0:
            p0, p1 = c + [-hx, -hy - w / 2], c + [hx, -hy - w / 2]; n = np.array([0, -1.0])
        elif side == 1:
            p0, p1 = c + [hx + w / 2, -hy], c + [hx + w / 2, hy]; n = np.array([1.0, 0])
        elif side == 2:
            p0, p1 = c + [hx, hy + w / 2], c + [-hx, hy + w / 2]; n = np.array([0, 1.0])
        else:
            p0, p1 = c + [-hx - w / 2, hy], c + [-hx - w / 2, -hy]; n = np.array([-1.0, 0])
        return p0, p1, n
    sense = e.sense
    exit_side = _side_of(e, exit_) if exit_ is not None else 0

    def corners_of_side(side: int, sense_: int):
        # corner at the start and end of `side` when walking with sense_ (ccw = +1)
        ccw = {0: (0, 1), 1: (1, 2), 2: (2, 3), 3: (3, 0)}[side]
        return ccw if sense_ > 0 else (ccw[1], ccw[0])
    # you arrive on one side of the well (the landing), walk it in the turning sense and the
    # first flight leaves from the far corner of that side
    e.n_fl = _n_flights(e, z0, z1, cfg)
    if entry is not None:
        first_corner = _first_corner(e, entry, exit_dir, cfg)
    elif e.m_entry is not None and e.m_centre is not None:
        dm = e.m_entry - e.m_centre
        first_corner = {(-1, -1): 0, (1, -1): 1, (1, 1): 2, (-1, 1): 3}[(1 if dm[0] >= 0 else -1, 1 if dm[1] >= 0 else -1)]
    else:
        first_corner = 0
    entry_side = next(sd for sd in range(4) if corners_of_side(sd, sense)[0] == first_corner)
    if exit_ is None:
        exit_side = (entry_side + 2) % 4
    # a storey is climbed in 2-4 flights of ~1.5 m around the well (or as configured)
    n_fl = e.n_fl
    order = [(entry_side + k * sense) % 4 for k in range(n_fl)]
    dz = (z1 - z0) / n_fl
    flights, landings, route = [], [], []
    thick = []
    seen_landings = set()
    z = z0
    n_route = int(e.exit_after) if e.exit_after and 0 < e.exit_after < n_fl else n_fl
    route.append(np.r_[corner_xy[first_corner], z0])
    for fi, side in enumerate(order):
        c0, c1 = corners_of_side(side, sense)
        p0, p1, n = side_strip(side, z, z + dz)
        if sense < 0:
            p0, p1 = p1, p0
        # quad along walking direction: left/right offsets by w/2 across n
        l0, l1 = p0 + n * (w / 2), p1 + n * (w / 2)
        r0, r1 = p0 - n * (w / 2), p1 - n * (w / 2)
        if sense < 0:
            l0, r0, l1, r1 = r0, l0, r1, l1
        # real steps: flat tread boxes stacked along the walking direction
        n_steps = max(3, int(round(abs(dz) / cfg.step_rise)))
        rise = dz / n_steps
        for k in range(n_steps):
            t0, t1 = k / n_steps, (k + 1) / n_steps
            a0 = l0 + (l1 - l0) * t0; a1 = l0 + (l1 - l0) * t1
            b0 = r0 + (r1 - r0) * t0; b1 = r0 + (r1 - r0) * t1
            ztop = z + (k + 1) * rise
            flights.append(np.array([np.r_[a0, ztop], np.r_[a1, ztop], np.r_[b1, ztop], np.r_[b0, ztop]]))
            thick.append(abs(rise))
        thick[-1] = cfg.slab          # the top tread is the landing surface: merge with it
        for cc, zz in ((c0, z), (c1, z + dz)):
            key = (cc, round(zz, 3))
            if key not in seen_landings:
                seen_landings.add(key)
                landings.append(_rect(corner_xy[cc], np.array([w, w]), zz))
        if fi < n_route:
            route.append(np.r_[p0, z + max(rise, 0)]); route.append(np.r_[p1, z + dz + max(-rise, 0)])
        z += dz
    last_corner = corners_of_side(order[n_route - 1], sense)[1]
    z_exit = z0 + n_route * dz
    route.append(np.r_[corner_xy[last_corner], z_exit])
    # the stair carries on beyond where the route leaves: extra flights, no route
    for k in range(int(e.cont)):
        side = (order[n_route - 1] + (k + 1) * sense) % 4
        c0, c1 = corners_of_side(side, sense)
        zc0 = z_exit + k * dz
        p0, p1, n = side_strip(side, zc0, zc0 + dz)
        if sense < 0:
            p0, p1 = p1, p0
        l0, l1 = p0 + n * (w / 2), p1 + n * (w / 2)
        r0, r1 = p0 - n * (w / 2), p1 - n * (w / 2)
        if sense < 0:
            l0, r0, l1, r1 = r0, l0, r1, l1
        n_steps = max(3, int(round(abs(dz) / cfg.step_rise)))
        rise = dz / n_steps
        for k2 in range(n_steps):
            t0, t1 = k2 / n_steps, (k2 + 1) / n_steps
            a0 = l0 + (l1 - l0) * t0; a1 = l0 + (l1 - l0) * t1
            b0 = r0 + (r1 - r0) * t0; b1 = r0 + (r1 - r0) * t1
            ztop = zc0 + (k2 + 1) * rise
            flights.append(np.array([np.r_[a0, ztop], np.r_[a1, ztop], np.r_[b1, ztop], np.r_[b0, ztop]]))
            thick.append(abs(rise))
        thick[-1] = cfg.slab
        for cc, zz in ((c0, zc0), (c1, zc0 + dz)):
            key = (cc, round(zz, 3))
            if key not in seen_landings:
                seen_landings.add(key)
                landings.append(_rect(corner_xy[cc], np.array([w, w]), zz))
    # continue along the gallery (at z1) from the last landing to the exit side, the short way round
    def gallery_walk(sense_):
        last_side = order[n_route - 1]
        pts_, side_, corner_ = [], (last_side + sense_) % 4 if sense_ == sense else last_side, last_corner
        for _ in range(4):
            if side_ == exit_side:
                break
            c_next = corners_of_side(side_, sense_)[1]
            pts_.append(np.r_[corner_xy[c_next], z_exit])
            corner_ = c_next
            side_ = (side_ + sense_) % 4
        return pts_, corner_
    g_fwd, c_fwd = gallery_walk(sense)
    g_back, c_back = gallery_walk(-sense)
    gallery, corner = (g_fwd, c_fwd) if len(g_fwd) <= len(g_back) else (g_back, c_back)
    # exit: along the exit side's gallery centreline to the exit point, then straight out
    if exit_ is not None:
        ex = np.array(exit_, float)
        ex = np.clip(ex, e.centre - e.size / 2 + w / 2, e.centre + e.size / 2 - w / 2)
        last = gallery[-1][:2] if gallery else corner_xy[corner]
        # exit side is horizontal (S/N) -> walk along x first at the gallery's y, else along y
        if exit_side in (0, 2):
            gallery.append(np.r_[ex[0], last[1], z_exit])
        else:
            gallery.append(np.r_[last[0], ex[1], z_exit])
        gallery.append(np.r_[ex, z_exit])
    # entry: from the entry point straight to the first landing, right-angled
    if entry is not None:
        en = np.array(entry, float)
        en = np.clip(en, e.centre - e.size / 2 + w / 2, e.centre + e.size / 2 - w / 2)
        first = corner_xy[first_corner]
        if entry_side in (0, 2):
            route = [np.r_[en, z0], np.r_[first[0], en[1], z0]] + route
        else:
            route = [np.r_[en, z0], np.r_[en[0], first[1], z0]] + route
    route_gallery = gallery
    void_corners = _rect(c, e.void_size, 0.0)[:, :2]
    from .stairs import gallery_ring
    ring_mode = cfg.gallery if cfg.gallery in ("upper", "both", "none") else "none"
    atrium = {"corners": void_corners.round(4).tolist(), "z0": float(z0), "z1": float(z1), "flight_width": w,
              "ring": [np.round(q, 4).tolist() for q in gallery_ring(void_corners, w)] if ring_mode != "none" else [],
              "inferred_ring": True, "ring_levels": ring_mode, "void_lines": bool(cfg.void_lines)}
    # traversed gallery: slabs only under the upper-level legs the route actually walks
    walkways = []
    if cfg.gallery == "traversed":
        floor_zs = {z0, z1}
        legs = []
        if entry is not None:
            legs.append(route[:3])                      # landing you arrive on (at z0)
        legs.append([route[-1]] + route_gallery)        # landing you leave from (at z_exit)
        for leg in legs:
            for a_, b_ in zip(leg[:-1], leg[1:]):
                zz = float(a_[2])
                if min(abs(zz - z0), abs(zz - z1), abs(zz - z_exit)) > 1e-6:
                    continue
                a2, b2 = np.asarray(a_[:2], float), np.asarray(b_[:2], float)
                dv = b2 - a2; L = float(np.linalg.norm(dv))
                if L < 1e-6:
                    continue
                dv /= L; nn = np.array([-dv[1], dv[0]])
                # a leg running along a side of the well becomes the full landing of that side
                along_x = abs(dv[1]) < 1e-6
                on_side = None
                if along_x and abs(abs(a2[1] - c[1]) - (hy + w / 2)) < 1e-6:
                    on_side = 0 if a2[1] < c[1] else 2
                elif (not along_x) and abs(abs(a2[0] - c[0]) - (hx + w / 2)) < 1e-6:
                    on_side = 3 if a2[0] < c[0] else 1
                if on_side is not None:
                    ca, cb = corners_of_side(on_side, 1)
                    pa, pb = corner_xy[ca], corner_xy[cb]
                    dd = (pb - pa) / max(np.linalg.norm(pb - pa), 1e-9)
                    a2e, b2e = pa - dd * w / 2, pb + dd * w / 2
                    nn = np.array([-dd[1], dd[0]])
                else:
                    a2e, b2e = a2 - dv * w / 2, b2 + dv * w / 2
                walkways.append(np.array([np.r_[a2e + nn * w / 2, zz], np.r_[b2e + nn * w / 2, zz],
                                          np.r_[b2e - nn * w / 2, zz], np.r_[a2e - nn * w / 2, zz]]))
    e._z_exit = z_exit
    return flights, landings + walkways, atrium, (route, route_gallery), thick


def _straight_flight(a: np.ndarray, b: np.ndarray, width: float, za: float, zb: float, cfg: IdealConfig):
    """Step boxes for a straight flight from xy a (floor za) to xy b (floor zb). Returns (quads, thickness)."""
    a = np.asarray(a, float); b = np.asarray(b, float)
    d = b - a; L = float(np.linalg.norm(d)); d = d / max(L, 1e-9); n = np.array([-d[1], d[0]])
    n_steps = max(2, int(round(abs(zb - za) / cfg.step_rise)))
    rise = (zb - za) / n_steps
    quads, thick = [], []
    for k in range(n_steps):
        t0, t1 = k / n_steps, (k + 1) / n_steps
        p0 = a + d * L * t0; p1 = a + d * L * t1
        ztop = za + (k + 1) * rise
        quads.append(np.array([np.r_[p0 + n * width / 2, ztop], np.r_[p1 + n * width / 2, ztop],
                               np.r_[p1 - n * width / 2, ztop], np.r_[p0 - n * width / 2, ztop]]))
        thick.append(abs(rise))
    return quads, thick


def _stairs_run(rid: int, quads, thick, level_from: int, level_to: int, cam_h: float) -> dict:
    """A plain stairs run (straight flight) for the diagram dict."""
    return {"id": rid, "kind": "stairs", "hall": False, "level": -1, "level_from": level_from, "level_to": level_to,
            "pieces": [{"p0": [*quads[0][0][:2], quads[0][0][2] + cam_h], "p1": [*quads[-1][1][:2], quads[-1][1][2] + cam_h],
                        "left": 0.0, "right": 0.0, "axis": "", "length": float(np.linalg.norm(quads[-1][1][:2] - quads[0][0][:2])),
                        "left_found": True, "right_found": True, "openings": []}],
            "left_boundary": [q[0].tolist() for q in quads] + [quads[-1][1].tolist()],
            "right_boundary": [q[3].tolist() for q in quads] + [quads[-1][2].tolist()],
            "outline": np.vstack([q[:, :2] for q in quads]).tolist(),
            "quads": [q.tolist() for q in quads], "quad_thickness": list(thick), "steps": True, "atrium": None, "walls": []}


def _corners_of_side(side: int, sense_: int):
    ccw = {0: (0, 1), 1: (1, 2), 2: (2, 3), 3: (3, 0)}[side]
    return ccw if sense_ > 0 else (ccw[1], ccw[0])


def place_exit_hall(stair: Element, hall: Element, entry: np.ndarray, corridor_from: np.ndarray, cfg: IdealConfig,
                    z0: float, z1: float, corridor_axis: str | None = None) -> None:
    """Put `hall` (entered part-way down/up `stair`) flush against the block at the exit
    corner, on the side facing away from the corridor the stair was entered from."""
    w = cfg.flight_width
    landing_side = _side_of(stair, np.asarray(entry, float))
    first_corner = _first_corner(stair, entry)
    entry_side = next(sd for sd in range(4) if _corners_of_side(sd, stair.sense)[0] == first_corner)
    n_fl = _n_flights(stair, z0, z1, cfg)
    n_route = int(stair.exit_after) if stair.exit_after and 0 < stair.exit_after < n_fl else n_fl
    order = [(entry_side + k * stair.sense) % 4 for k in range(n_fl)]
    exit_corner = _corners_of_side(order[n_route - 1], stair.sense)[1]
    sides_of_corner = {0: (0, 3), 1: (0, 1), 2: (1, 2), 3: (2, 3)}[exit_corner]   # SW: S,W ; SE: S,E ; NE: E,N ; NW: N,W
    normals = {0: np.array([0, -1.0]), 1: np.array([1.0, 0]), 2: np.array([0, 1.0]), 3: np.array([-1.0, 0])}
    free = [sd for sd in sides_of_corner if sd != landing_side]     # not the side the corridor arrives on
    if len(free) == 1:
        side = free[0]
    else:
        away = stair.centre - np.asarray(corridor_from, float)      # direction away from where the corridor is
        if corridor_axis in ("x", "y"):
            away[0 if corridor_axis == "x" else 1] = 0.0           # only the lateral offset of the corridor counts
        if np.linalg.norm(away) < 1e-6:
            away = normals[free[-1]]
        side = max(free, key=lambda sd: float(normals[sd] @ away))
    n = normals[side]
    k = 0 if side in (1, 3) else 1
    # the same slab as a start foyer: as deep as the block, `foyer_ext` out from the block face
    # and one landing square into it, so the stair ends exactly where the hall begins
    ext = cfg.foyer_ext if cfg.foyer_ext else max(cfg.hall_min, float(hall.size[k]))
    face = stair.centre[k] + n[k] * stair.size[k] / 2
    hall.size = stair.size.copy(); hall.size[k] = ext + w
    hall.centre = stair.centre.copy(); hall.centre[k] = face + n[k] * (ext - w) / 2
    landing = _corner_xy(stair, exit_corner, w)
    entry_pt = landing.copy(); entry_pt[k] = face
    hall.entry_pt = entry_pt
    hall.term = entry_pt.copy(); hall.term[k] = face + n[k] * max(ext - 1.5, ext / 2)   # straight ahead, 1.5 m from the end
    hall.base_of = None
    hall.exit_from = stair


def _manhattan(a: np.ndarray, b: np.ndarray, z: float) -> list[np.ndarray]:
    """Right-angled route from a to b: first along x, then along y."""
    mid = np.array([b[0], a[1]])
    pts = [np.r_[a, z]]
    if np.linalg.norm(mid - a) > 1e-6:
        pts.append(np.r_[mid, z])
    pts.append(np.r_[b, z])
    return pts


def idealize(d: dict, cfg: IdealConfig) -> dict:
    els = _elements(d, cfg)
    ppts = np.array(d["path"]["points"])
    _layout(els, cfg, terminals=(ppts[0, :2].copy(), ppts[-1, :2].copy()))
    levels = {lv["id"]: lv for lv in d["levels"]}
    floor = {lid: _snap(lv["floor_z"] - d["levels"][0]["floor_z"], 0.25) for lid, lv in levels.items()}
    cam_h = d.get("camera_height_m", 1.5)
    runs = []
    path_pts: list[np.ndarray] = []
    path_lv: list[int] = []
    labels = []
    z_bias = 0.0                      # raised by short flights inside corridors (carried to the next stair)
    entry_rise = cfg.entry_steps * cfg.step_rise if cfg.entry_steps > 0 else 0.0
    exit_level_z: float | None = None  # floor height of the hall after a stair left part-way
    extra_levels: list[dict] = []
    corridor_idx = [k for k, e in enumerate(els) if e.kind == "corridor"]
    for i, e in enumerate(els):
        prev_ = els[i - 1] if i > 0 else None
        next_ = els[i + 1] if i + 1 < len(els) else None
        rid = len(runs)
        if e.kind == "hall":
            z = floor[e.level]
            if exit_level_z is not None:
                # the route left the previous stair part-way: this hall is at that landing level
                lid = max(floor) + 1
                floor[lid] = exit_level_z
                extra_levels.append({"id": lid, "name": f"Landing level ({exit_level_z:+.1f} m)", "floor_z": exit_level_z,
                                     "ceiling_z": exit_level_z + 3.0, "camera_z": exit_level_z + cam_h})
                e.level = lid
                z = exit_level_z
                exit_level_z = None
            quad = _rect(e.centre, e.size, z)
            runs.append({"id": rid, "kind": "flat", "hall": True, "level": e.level, "level_from": -1, "level_to": -1,
                         "pieces": [{"p0": [*(e.centre - [e.size[0] / 2, 0]), z + cam_h], "p1": [*(e.centre + [e.size[0] / 2, 0]), z + cam_h],
                                     "left": e.size[1] / 2, "right": e.size[1] / 2, "axis": "x", "length": float(e.size[0]),
                                     "left_found": True, "right_found": True, "openings": []}],
                         "left_boundary": quad[[0, 1]].tolist(), "right_boundary": quad[[3, 2]].tolist(),
                         "outline": quad[:, :2].tolist(), "quads": [quad.tolist()], "atrium": None, "walls": []})
            # route through the hall: right-angled, from where we come in to where we leave
            lo_h, hi_h = e.centre - e.size / 2, e.centre + e.size / 2
            if e.exit_from is not None:
                st = e.exit_from
                if e.entry_pt is not None:
                    edge = np.asarray(e.entry_pt, float).copy()
                else:
                    dd = e.centre - st.centre; kk = int(np.argmax(np.abs(dd)))
                    edge = e.centre.copy(); edge[kk] = st.centre[kk] + (1 if dd[kk] >= 0 else -1) * st.size[kk] / 2
                    edge[1 - kk] = float(np.clip(edge[1 - kk], lo_h[1 - kk] + 0.75, hi_h[1 - kk] - 0.75))
                for p in _manhattan(edge, e.term if e.term is not None else e.centre, z + cam_h):
                    path_pts.append(p); path_lv.append(e.level)
                continue
            if e.base_of is not None:
                # base slab under a staircase: the route runs from the Start/End point along the
                # slab centreline to the stair block (the stair itself adds the rest)
                nb = e.base_of
                d_ = e.term - nb.centre
                k = int(np.argmax(np.abs(d_)))
                sign = 1 if d_[k] >= 0 else -1
                edge = nb.centre.copy(); edge[k] = nb.centre[k] + sign * nb.size[k] / 2
                edge[1 - k] = float(e.term[1 - k])          # same row as the Start/End point
                if prev_ is None and entry_rise > 0:
                    # lower entrance foyer + straight flight up to the stair (whose landing stays at level z)
                    foyer = e.centre.copy(); fsize = e.size.copy()
                    zf = z - entry_rise
                    runs[-1]["quads"] = [_rect(foyer, fsize, zf).tolist()]
                    runs[-1]["outline"] = _rect(foyer, fsize, zf)[:, :2].tolist()
                    runs[-1]["left_boundary"] = _rect(foyer, fsize, zf)[[0, 1]].tolist()
                    runs[-1]["right_boundary"] = _rect(foyer, fsize, zf)[[3, 2]].tolist()
                    run_len = cfg.entry_steps * cfg.step_run
                    fa = edge.copy(); fa[k] = edge[k] + sign * run_len
                    fq, ft = _straight_flight(fa, edge, cfg.corridor_width, zf, z, cfg)
                    runs.append(_stairs_run(len(runs), fq, ft, e.level, e.level, cam_h))
                    for p in _manhattan(np.asarray(e.term, float), fa, zf + cam_h):
                        path_pts.append(p); path_lv.append(e.level)
                    path_pts.append(np.r_[fa, zf + cam_h + entry_rise * 0.0]); path_lv.append(-1)
                    path_pts.append(np.r_[edge, z + cam_h]); path_lv.append(-1)
                    path_pts.append(np.r_[edge, z + cam_h]); path_lv.append(e.level)
                    continue
                zc = z + cam_h
                seq = [e.term, edge] if prev_ is None else [edge, e.term]
                for a_, b_ in zip(seq[:-1], seq[1:]):
                    for p in _manhattan(np.asarray(a_, float), np.asarray(b_, float), zc):
                        path_pts.append(p); path_lv.append(e.level)
                continue

            def edge_point(other: Element) -> np.ndarray:
                # point on the hall boundary facing `other`, at other's coordinate on the shared axis
                d = other.centre - e.centre
                k = int(np.argmax(np.abs(d) - e.size / 2))      # axis along which `other` lies outside
                p = np.array(other.centre, float)
                if other.kind == "corridor":
                    p = np.array(other.a if np.linalg.norm(other.a - e.centre) < np.linalg.norm(other.b - e.centre) else other.b, float)
                p[k] = hi_h[k] if d[k] > 0 else lo_h[k]
                p[1 - k] = float(np.clip(p[1 - k], lo_h[1 - k] + 0.75, hi_h[1 - k] - 0.75))
                return p
            pin = edge_point(prev_) if prev_ is not None else e.centre
            pout = edge_point(next_) if next_ is not None else e.centre
            seq = [pin]
            if prev_ is None or next_ is None:
                seq.append(e.centre)
            seq.append(pout)
            zc = z + cam_h
            for a_, b_ in zip(seq[:-1], seq[1:]):
                for p in _manhattan(np.asarray(a_, float), np.asarray(b_, float), zc):
                    path_pts.append(p); path_lv.append(e.level)
        elif e.kind == "corridor":
            z = floor[e.level] + z_bias
            dv = e.b - e.a; L = float(np.linalg.norm(dv)); dv = dv / max(L, 1e-9); n = np.array([-dv[1], dv[0]])
            hw = cfg.corridor_width / 2
            # short flights inside this corridor (fraction along it, number of steps; negative = down)
            steps_here = [(f_, ns) for f_, ns in cfg.mid_steps] if corridor_idx and i == corridor_idx[0] else []
            cuts = [(0.0, 0)] + sorted(steps_here) + [(1.0, 0)]
            zz = z
            for (f0, n0), (f1, n1) in zip(cuts[:-1], cuts[1:]):
                a_ = e.a + dv * L * f0; b_ = e.a + dv * L * f1
                if n0:
                    run_len = abs(n0) * cfg.step_run
                    a_ = a_ + dv * run_len / 2
                if n1:
                    run_len = abs(n1) * cfg.step_run
                    b_ = b_ - dv * run_len / 2
                quad = np.array([np.r_[a_ + n * hw, zz], np.r_[b_ + n * hw, zz], np.r_[b_ - n * hw, zz], np.r_[a_ - n * hw, zz]])
                runs.append({"id": len(runs), "kind": "flat", "hall": False, "level": e.level, "level_from": -1, "level_to": -1,
                             "pieces": [{"p0": [*a_, zz + cam_h], "p1": [*b_, zz + cam_h], "left": hw, "right": hw, "axis": e.axis,
                                         "length": float(np.linalg.norm(b_ - a_)), "left_found": True, "right_found": True, "openings": []}],
                             "left_boundary": quad[[0, 1]].tolist(), "right_boundary": quad[[3, 2]].tolist(),
                             "outline": quad[:, :2].tolist(), "quads": [quad.tolist()], "atrium": None, "walls": []})
                for p in (np.r_[a_, zz + cam_h], np.r_[b_, zz + cam_h]):
                    path_pts.append(p); path_lv.append(e.level)
                if n1:
                    rise = n1 * cfg.step_rise
                    fa = b_; fb = b_ + dv * abs(n1) * cfg.step_run
                    fq, ft = _straight_flight(fa, fb, cfg.corridor_width, zz, zz + rise, cfg)
                    runs.append(_stairs_run(len(runs), fq, ft, e.level, e.level, cam_h))
                    path_pts.append(np.r_[fa, zz + cam_h]); path_lv.append(-1)
                    path_pts.append(np.r_[fb, zz + rise + cam_h]); path_lv.append(-1)
                    zz += rise
                    labels.append({"text": f"{abs(n1)} steps {'up' if rise > 0 else 'down'}", "xyz": [*(0.5 * (fa + fb)), zz - rise],
                                   "level": e.level, "floor_z": float(zz - rise), "kind": "stairs"})
            z_bias = zz - floor[e.level]
        else:
            z0, z1 = floor.get(e.level_from, 0.0) + z_bias, floor.get(e.level_to, 0.0)
            z_bias = 0.0
            entry = prev_.centre if prev_ is not None else None
            if prev_ is not None and prev_.kind == "corridor":
                entry = prev_.b
            exit_dir = _exit_dir(els, i)
            if prev_ is not None and prev_.base_of is e:
                entry = e.centre.copy(); dd = prev_.term - e.centre; kk = int(np.argmax(np.abs(dd)))
                entry[kk] = e.centre[kk] + (1 if dd[kk] >= 0 else -1) * e.size[kk] / 2
                entry[1 - kk] = _corner_xy(e, _first_corner(e, entry, exit_dir, cfg), cfg.flight_width)[1 - kk]
            exit_ = next_.centre if next_ is not None else None
            if next_ is not None and next_.kind == "corridor":
                exit_ = next_.a
            if next_ is not None and next_.base_of is e:
                exit_ = e.centre.copy(); dd = next_.term - e.centre; kk = int(np.argmax(np.abs(dd)))
                exit_[kk] = e.centre[kk] + (1 if dd[kk] >= 0 else -1) * e.size[kk] / 2
            if e.exit_after and next_ is not None and next_.kind == "hall" and entry is not None:
                is_corr = prev_ is not None and prev_.kind == "corridor"
                corridor_from = prev_.b if is_corr else np.asarray(entry, float)
                place_exit_hall(e, next_, entry, corridor_from, cfg, z0, z1, corridor_axis=prev_.axis if is_corr else None)
                exit_ = np.asarray(next_.entry_pt, float).copy()
            flights, landings, atrium, (route, gallery), thick = _stair_block(e, entry, exit_, z0, z1, cfg, exit_dir=exit_dir)
            if e.exit_after and abs(e._z_exit - z1) > 1e-6:
                exit_level_z = float(e._z_exit)
            quads = flights + landings
            quad_thickness = thick + [cfg.slab] * len(landings)
            runs.append({"id": rid, "kind": "stairs", "hall": False, "level": -1, "level_from": e.level_from, "level_to": e.level_to,
                         "pieces": [{"p0": [*q[0][:2], q[0][2] + cam_h], "p1": [*q[1][:2], q[1][2] + cam_h], "left": cfg.flight_width / 2,
                                     "right": cfg.flight_width / 2, "axis": "", "length": float(np.linalg.norm(q[1][:2] - q[0][:2])),
                                     "left_found": True, "right_found": True, "openings": []} for q in flights],
                         "left_boundary": [q[0].tolist() for q in flights] + [flights[-1][1].tolist()],
                         "right_boundary": [q[3].tolist() for q in flights] + [flights[-1][2].tolist()],
                         "outline": np.vstack([q[:, :2] for q in flights]).tolist(),
                         "quads": [q.tolist() for q in quads], "quad_thickness": quad_thickness, "steps": True,
                         "atrium": atrium, "walls": []})
            # route: entry leg (visible) -> flights (hidden) -> gallery to the exit (visible)
            n_entry = 3 if entry is not None else 1
            for p in route[:n_entry]:
                path_pts.append(np.r_[p[:2], p[2] + cam_h]); path_lv.append(e.level_from)
            for p in route[n_entry - 1:]:
                path_pts.append(np.r_[p[:2], p[2] + cam_h]); path_lv.append(-1)
            for p in [route[-1]] + gallery:
                path_pts.append(np.r_[p[:2], p[2] + cam_h]); path_lv.append(-2 if exit_level_z is not None else e.level_to)
            # marker where you start climbing / descending: the entry corner of the flights
            start_pt = route[3] if (entry is not None and len(route) > 3) else route[0]
            n_floors = abs(int(e.level_to) - int(e.level_from)) if (e.level_to >= 0 and e.level_from >= 0) else 1
            floors = {1: "one floor", 2: "two floors", 3: "three floors"}.get(n_floors, f"{n_floors} floors")
            labels.append({"text": f"Stairs {'up' if z1 > z0 else 'down'} {floors}", "xyz": [float(start_pt[0]), float(start_pt[1]), float(z0)],
                           "level": int(e.level_from), "floor_z": float(z0), "kind": "stairs"})
    # -2 marks route points on a landing-level hall created after a part-way exit
    if extra_levels:
        lid = extra_levels[-1]["id"]
        path_lv = [lid if l == -2 else l for l in path_lv]
    else:
        path_lv = [-1 if l == -2 else l for l in path_lv]
    pts = np.array(path_pts)
    labels.insert(0, {"text": cfg.start_label, "xyz": [float(pts[0, 0]), float(pts[0, 1]), float(pts[0, 2] - cam_h)], "level": int(path_lv[0]), "floor_z": float(pts[0, 2] - cam_h), "role": "start"})
    labels.insert(1, {"text": cfg.end_label, "xyz": [float(pts[-1, 0]), float(pts[-1, 1]), float(pts[-1, 2] - cam_h)], "level": int(path_lv[-1]), "floor_z": float(pts[-1, 2] - cam_h), "role": "end"})
    # user labels (anything that is not Start/End/stairs) keep their text, snapped to the nearest route point
    for lb in d.get("labels", []):
        if lb.get("kind") == "stairs" or lb["text"] in ("Start", "End"):
            continue
        p = np.array(lb["xyz"][:2])
        k = int(np.argmin(np.linalg.norm(pts[:, :2] - p, axis=1)))
        labels.append({"text": lb["text"], "xyz": [float(pts[k, 0]), float(pts[k, 1]), float(pts[k, 2] - cam_h)],
                       "level": int(path_lv[k]) if path_lv[k] >= 0 else 0, "floor_z": float(pts[k, 2] - cam_h)})
    out = {
        "frame": "ideal", "units": "m", "camera_height_m": cam_h, "plan_rotation_deg": 0.0,
        "note": f"idealised schematic on a {cfg.grid} m grid from the measured diagram",
        "levels": [{"id": lid, "name": lv["name"], "floor_z": floor[lid], "ceiling_z": floor[lid] + 3.0, "camera_z": floor[lid] + cam_h}
                   for lid, lv in levels.items()] + extra_levels,
        "runs": runs,
        "path": {"step": None, "points": pts.round(4).tolist(), "level": path_lv},
        "labels": labels,
        "config": cfg.__dict__ | {"title": cfg.title},
        "source": "idealized",
    }
    return out


def _ideal_meshes(ideal: dict, slab: float) -> list[Mesh]:
    floors = Mesh("floor", (0.90, 0.88, 0.84, 1.0))
    stairs = Mesh("stairs", (0.80, 0.76, 0.68, 1.0))
    atrium = Mesh("atrium", (0.35, 0.35, 0.37, 1.0))
    for r in ideal["runs"]:
        th = r.get("quad_thickness") or [slab] * len(r["quads"])
        for q, t in zip(r["quads"], th):
            top = np.array(q); bottom = top.copy(); bottom[:, 2] -= t
            (stairs if r["kind"] == "stairs" else floors).add_prism(bottom, top)
        if r.get("atrium"):
            at = r["atrium"]; corners = np.array(at["corners"])
            for z in (at["z0"], at["z1"]):
                for q in at["ring"]:
                    top = np.c_[np.array(q), np.full(4, z)]; bottom = top.copy(); bottom[:, 2] -= slab
                    floors.add_prism(bottom, top)
                for k in range(4):
                    a, b = corners[k], corners[(k + 1) % 4]
                    atrium.add_box_between(np.r_[a, z], np.r_[b, z], 0.06, 0.0, 0.06)
            for k in range(4):
                c = corners[k]
                atrium.add_box_between(np.r_[c - [0.03, 0], min(at["z0"], at["z1"])], np.r_[c + [0.03, 0], min(at["z0"], at["z1"])],
                                       0.06, 0.0, abs(at["z1"] - at["z0"]))
    return [floors, stairs, atrium]


def run_idealize(ws: Workspace, cfg: IdealConfig, diagram_path: os.PathLike | str | None = None,
                 out_dir: os.PathLike | str | None = None) -> dict:
    src = Path(diagram_path) if diagram_path else ws.geometry / "diagram.json"
    d = read_json(src)
    ideal = idealize(d, cfg)
    out = Path(out_dir) if out_dir else ws.geometry / "ideal"
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "diagram.json", ideal)
    allxy = np.vstack([np.array(q)[:, :2] for r in ideal["runs"] for q in r["quads"]])
    bounds = (allxy[:, 0].min(), allxy[:, 1].min(), allxy[:, 0].max(), allxy[:, 1].max())
    write_plan_svg(out / "plan.svg", ideal, None, bounds, title=cfg.title or "Plan", show_path=cfg.show_path)
    for lv in ideal["levels"]:
        write_plan_svg(out / f"plan_L{lv['id']}.svg", ideal, lv["id"], bounds, title=lv["name"], show_path=cfg.show_path)
    from .svg import THEMES

    view = AxoView(cfg.iso_azimuth, cfg.iso_elevation, explode_m=cfg.explode)
    kw = dict(title=cfg.title, slab=cfg.slab, label_style=cfg.label_style, hidden_lines=cfg.hidden_lines,
              path_morse=cfg.path_morse, morse_unit_px=cfg.morse_unit_px)
    write_axo_svg(out / "iso.svg", ideal, view, show_path=cfg.show_path, theme=cfg.theme, **kw)
    write_axo_svg(out / "iso_noroute.svg", ideal, view, show_path=False, theme=cfg.theme, **kw)
    if len(ideal["levels"]) > 1 and cfg.explode == 0:
        write_axo_svg(out / "iso_exploded.svg", ideal, AxoView(cfg.iso_azimuth, cfg.iso_elevation, explode_m=3.0),
                      show_path=cfg.show_path, theme=cfg.theme, **kw)
    if cfg.all_themes:
        (out / "styles").mkdir(exist_ok=True)
        for name in THEMES:
            write_axo_svg(out / "styles" / f"iso_{name}.svg", ideal, view, show_path=cfg.show_path, theme=name, **kw)
    meshes = _ideal_meshes(ideal, cfg.slab)
    write_glb(out / "diagram.glb", meshes)
    write_obj(out / "diagram.obj", meshes)
    log.info("idealised diagram written to %s (%d elements)", out, len(ideal["runs"]))
    return ideal
