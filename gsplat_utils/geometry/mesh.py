"""Tiny mesh toolkit: boxes/prisms from quads, flat-shaded GLB and OBJ writers (no deps)."""
from __future__ import annotations

import json
import struct
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass
class Mesh:
    name: str
    color: tuple[float, float, float, float]
    vertices: list[np.ndarray] = field(default_factory=list)   # list of (3,) arrays
    faces: list[tuple[int, int, int]] = field(default_factory=list)

    def add_quad(self, a, b, c, d) -> None:
        """Quad a-b-c-d (counter-clockwise seen from the outside)."""
        i = len(self.vertices)
        self.vertices += [np.asarray(a, float), np.asarray(b, float), np.asarray(c, float), np.asarray(d, float)]
        self.faces += [(i, i + 1, i + 2), (i, i + 2, i + 3)]

    def add_prism(self, quad_bottom: np.ndarray, quad_top: np.ndarray) -> None:
        """Closed hexahedron from a bottom and a top 4-gon (same winding, bottom CCW seen from above)."""
        b, t = quad_bottom, quad_top
        self.add_quad(t[0], t[1], t[2], t[3])          # top
        self.add_quad(b[3], b[2], b[1], b[0])          # bottom
        for k in range(4):
            k2 = (k + 1) % 4
            self.add_quad(b[k], b[k2], t[k2], t[k])    # sides
        # fix winding so that normals point outwards
        self._orient_last(12)

    def add_box_between(self, a: np.ndarray, b: np.ndarray, width: float, z0: float, z1: float,
                        za1: float | None = None, zb1: float | None = None) -> None:
        """Box whose centreline runs from xy a to xy b (can be sloped: z at a / b)."""
        a = np.asarray(a, float); b = np.asarray(b, float)
        d = b[:2] - a[:2]
        L = np.linalg.norm(d)
        if L < 1e-9:
            return
        n = np.array([-d[1], d[0]]) / L * (width / 2)
        za0, zb0 = (a[2] if len(a) > 2 else z0), (b[2] if len(b) > 2 else z0)
        if za1 is None:
            za1 = za0 + (z1 - z0)
        if zb1 is None:
            zb1 = zb0 + (z1 - z0)
        bottom = np.array([np.r_[a[:2] + n, za0], np.r_[b[:2] + n, zb0], np.r_[b[:2] - n, zb0], np.r_[a[:2] - n, za0]])
        top = bottom.copy(); top[:, 2] = [za1, zb1, zb1, za1]
        self.add_prism(bottom, top)

    def _orient_last(self, n_faces: int) -> None:
        V = np.array(self.vertices)
        fs = self.faces[-n_faces:]
        idx = np.array(fs).ravel()
        centre = V[idx].mean(axis=0)
        fixed = []
        for f in fs:
            p0, p1, p2 = V[f[0]], V[f[1]], V[f[2]]
            nrm = np.cross(p1 - p0, p2 - p0)
            if nrm @ ((p0 + p1 + p2) / 3 - centre) < 0:
                f = (f[0], f[2], f[1])
            fixed.append(f)
        self.faces[-n_faces:] = fixed

    def arrays(self) -> tuple[np.ndarray, np.ndarray]:
        return np.array(self.vertices, dtype=np.float32).reshape(-1, 3), np.array(self.faces, dtype=np.uint32).reshape(-1, 3)

    def flat_shaded(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Unindexed triangles with per-face normals."""
        V, F = self.arrays()
        tri = V[F]                                   # (M,3,3)
        n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
        n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)
        pos = tri.reshape(-1, 3)
        nrm = np.repeat(n, 3, axis=0).astype(np.float32)
        idx = np.arange(len(pos), dtype=np.uint32)
        return pos.astype(np.float32), nrm, idx


def _z_up_to_y_up(p: np.ndarray) -> np.ndarray:
    """(x, y, z) with z up  ->  (x, z, -y) with y up (glTF / most DCC tools)."""
    out = np.empty_like(p)
    out[:, 0] = p[:, 0]
    out[:, 1] = p[:, 2]
    out[:, 2] = -p[:, 1]
    return out


def write_glb(path, meshes: list[Mesh], y_up: bool = True) -> None:
    """Minimal glTF 2.0 binary with one node per mesh and flat PBR materials."""
    buffer = bytearray()
    buffer_views = []
    accessors = []
    gltf_meshes = []
    materials = []
    nodes = []

    def add_view(data: bytes, target: int) -> int:
        while len(buffer) % 4:
            buffer.extend(b"\x00")
        buffer_views.append({"buffer": 0, "byteOffset": len(buffer), "byteLength": len(data), "target": target})
        buffer.extend(data)
        return len(buffer_views) - 1

    for m in meshes:
        if not m.faces:
            continue
        pos, nrm, idx = m.flat_shaded()
        if y_up:
            pos = _z_up_to_y_up(pos); nrm = _z_up_to_y_up(nrm)
        pv = add_view(pos.tobytes(), 34962)
        nv = add_view(nrm.tobytes(), 34962)
        iv = add_view(idx.tobytes(), 34963)
        accessors.append({"bufferView": pv, "componentType": 5126, "count": len(pos), "type": "VEC3",
                          "min": pos.min(axis=0).tolist(), "max": pos.max(axis=0).tolist()})
        accessors.append({"bufferView": nv, "componentType": 5126, "count": len(nrm), "type": "VEC3"})
        accessors.append({"bufferView": iv, "componentType": 5125, "count": len(idx), "type": "SCALAR"})
        materials.append({"name": m.name, "pbrMetallicRoughness": {"baseColorFactor": list(m.color),
                          "metallicFactor": 0.0, "roughnessFactor": 0.9},
                          "doubleSided": True, **({"alphaMode": "BLEND"} if m.color[3] < 1 else {})})
        base = len(accessors) - 3
        gltf_meshes.append({"name": m.name, "primitives": [{"attributes": {"POSITION": base, "NORMAL": base + 1},
                            "indices": base + 2, "material": len(materials) - 1}]})
        nodes.append({"name": m.name, "mesh": len(gltf_meshes) - 1})
    gltf = {
        "asset": {"version": "2.0", "generator": "gsplat-utils"},
        "scene": 0, "scenes": [{"nodes": list(range(len(nodes)))}], "nodes": nodes,
        "meshes": gltf_meshes, "materials": materials, "accessors": accessors,
        "bufferViews": buffer_views, "buffers": [{"byteLength": len(buffer)}],
    }
    js = json.dumps(gltf, separators=(",", ":")).encode()
    while len(js) % 4:
        js += b" "
    while len(buffer) % 4:
        buffer.extend(b"\x00")
    total = 12 + 8 + len(js) + 8 + len(buffer)
    with open(path, "wb") as f:
        f.write(struct.pack("<III", 0x46546C67, 2, total))
        f.write(struct.pack("<II", len(js), 0x4E4F534A)); f.write(js)
        f.write(struct.pack("<II", len(buffer), 0x004E4942)); f.write(bytes(buffer))


def write_obj(path, meshes: list[Mesh], y_up: bool = True) -> None:
    path = Path(path)
    mtl_path = path.with_suffix(".mtl")
    with open(mtl_path, "w") as mf:
        for m in meshes:
            r, g, b, a = m.color
            mf.write(f"newmtl {m.name}\nKd {r:.4f} {g:.4f} {b:.4f}\nd {a:.3f}\nillum 1\n\n")
    with open(path, "w") as f:
        f.write(f"# gsplat-utils building diagram ({'Y up' if y_up else 'Z up'}, metres)\n")
        f.write(f"mtllib {mtl_path.name}\n")
        off = 1
        for m in meshes:
            V, F = m.arrays()
            if len(F) == 0:
                continue
            if y_up:
                V = _z_up_to_y_up(V)
            f.write(f"o {m.name}\nusemtl {m.name}\n")
            for v in V:
                f.write(f"v {v[0]:.4f} {v[1]:.4f} {v[2]:.4f}\n")
            for a, b, c in F:
                f.write(f"f {a + off} {b + off} {c + off}\n")
            off += len(V)
