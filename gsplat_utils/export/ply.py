"""Minimal, dependency-free PLY IO (binary little endian + ascii) plus the
standard 3D Gaussian Splatting PLY layout used by gsplat / inria / most viewers."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np

_PLY_TYPES = {
    "char": "i1", "int8": "i1", "uchar": "u1", "uint8": "u1",
    "short": "i2", "int16": "i2", "ushort": "u2", "uint16": "u2",
    "int": "i4", "int32": "i4", "uint": "u4", "uint32": "u4",
    "float": "f4", "float32": "f4", "double": "f8", "float64": "f8",
}


def read_ply(path: os.PathLike | str) -> dict[str, dict[str, np.ndarray]]:
    """Read a PLY into {element_name: {property_name: array}}.

    List properties (e.g. face vertex_indices) are only supported for ascii files
    and the 'face' element in binary files with a single list property.
    """
    path = Path(path)
    with open(path, "rb") as f:
        header = []
        while True:
            line = f.readline()
            if not line:
                raise ValueError("truncated PLY header")
            header.append(line.decode("ascii", errors="replace").strip())
            if header[-1] == "end_header":
                break
        fmt = "ascii"
        elements: list[tuple[str, int, list[tuple[str, str, str | None, str | None]]]] = []
        for h in header:
            tok = h.split()
            if not tok:
                continue
            if tok[0] == "format":
                fmt = tok[1]
            elif tok[0] == "element":
                elements.append((tok[1], int(tok[2]), []))
            elif tok[0] == "property":
                if tok[1] == "list":
                    elements[-1][2].append((tok[4], "list", tok[2], tok[3]))
                else:
                    elements[-1][2].append((tok[2], tok[1], None, None))
        out: dict[str, dict[str, np.ndarray]] = {}
        if fmt == "ascii":
            text = f.read().decode("ascii", errors="replace").split("\n")
            li = 0
            for name, count, props in elements:
                cols: dict[str, list] = {p[0]: [] for p in props}
                for _ in range(count):
                    while li < len(text) and not text[li].strip():
                        li += 1
                    vals = text[li].split(); li += 1
                    vi = 0
                    for pname, ptype, ctype, itype in props:
                        if ptype == "list":
                            n = int(vals[vi]); vi += 1
                            cols[pname].append([float(v) for v in vals[vi:vi + n]]); vi += n
                        else:
                            cols[pname].append(float(vals[vi])); vi += 1
                out[name] = {k: (np.array(v) if props[[p[0] for p in props].index(k)][1] != "list"
                                 else np.array(v, dtype=object)) for k, v in cols.items()}
            return out
        endian = "<" if fmt == "binary_little_endian" else ">"
        for name, count, props in elements:
            if any(p[1] == "list" for p in props):
                # Only the common "one list property" face layout.
                if len(props) != 1:
                    raise ValueError("binary PLY with mixed list/scalar properties unsupported")
                pname, _, ctype, itype = props[0]
                cdt = np.dtype(endian + _PLY_TYPES[ctype]); idt = np.dtype(endian + _PLY_TYPES[itype])
                rows = []
                for _ in range(count):
                    n = int(np.frombuffer(f.read(cdt.itemsize), dtype=cdt)[0])
                    rows.append(np.frombuffer(f.read(idt.itemsize * n), dtype=idt).astype(np.int64))
                out[name] = {pname: np.array(rows, dtype=object) if rows and any(len(r) != len(rows[0]) for r in rows)
                             else np.array(rows)}
                continue
            dt = np.dtype([(p[0], endian + _PLY_TYPES[p[1]]) for p in props])
            buf = f.read(dt.itemsize * count)
            arr = np.frombuffer(buf, dtype=dt, count=count)
            out[name] = {p[0]: np.ascontiguousarray(arr[p[0]]) for p in props}
        return out


def write_ply(
    path: os.PathLike | str,
    vertex: dict[str, np.ndarray],
    faces: np.ndarray | None = None,
    comments: list[str] | None = None,
) -> None:
    """Write vertex properties (equal-length 1D arrays) and optional int faces (F,k)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    names = list(vertex.keys())
    n = len(vertex[names[0]])
    fields = []
    for k in names:
        a = np.asarray(vertex[k])
        if a.dtype == np.float64:
            a = a.astype(np.float32)
        vertex[k] = a
        fields.append((k, a.dtype.newbyteorder("<")))
    dt = np.dtype(fields)
    arr = np.empty(n, dtype=dt)
    for k in names:
        arr[k] = vertex[k]
    inv = {v: k for k, v in _PLY_TYPES.items()}
    lines = ["ply", "format binary_little_endian 1.0"]
    for c in comments or []:
        lines.append(f"comment {c}")
    lines.append(f"element vertex {n}")
    for k in names:
        lines.append(f"property {inv[vertex[k].dtype.str[1:]]} {k}")
    if faces is not None:
        faces = np.asarray(faces, dtype=np.int32)
        lines.append(f"element face {len(faces)}")
        lines.append("property list uchar int vertex_indices")
    lines.append("end_header")
    with open(path, "wb") as f:
        f.write(("\n".join(lines) + "\n").encode("ascii"))
        f.write(arr.tobytes())
        if faces is not None and len(faces):
            k = faces.shape[1]
            fdt = np.dtype([("n", "u1"), ("idx", "<i4", (k,))])
            fa = np.empty(len(faces), dtype=fdt)
            fa["n"] = k
            fa["idx"] = faces
            f.write(fa.tobytes())


def write_point_ply(path, xyz: np.ndarray, rgb: np.ndarray | None = None, normals: np.ndarray | None = None) -> None:
    xyz = np.asarray(xyz, dtype=np.float32)
    v = {"x": xyz[:, 0], "y": xyz[:, 1], "z": xyz[:, 2]}
    if normals is not None:
        nm = np.asarray(normals, dtype=np.float32)
        v.update(nx=nm[:, 0], ny=nm[:, 1], nz=nm[:, 2])
    if rgb is not None:
        c = np.asarray(rgb)
        if c.dtype != np.uint8:
            c = np.clip(c * 255 if c.max() <= 1.0 else c, 0, 255).astype(np.uint8)
        v.update(red=c[:, 0], green=c[:, 1], blue=c[:, 2])
    write_ply(path, v)


def read_point_ply(path) -> tuple[np.ndarray, np.ndarray | None]:
    d = read_ply(path)["vertex"]
    xyz = np.stack([d["x"], d["y"], d["z"]], axis=1).astype(np.float64)
    rgb = None
    if "red" in d:
        rgb = np.stack([d["red"], d["green"], d["blue"]], axis=1)
    return xyz, rgb


# ---------------------------------------------------------------------------
# 3D Gaussian Splatting PLY
# ---------------------------------------------------------------------------
SH_C0 = 0.28209479177387814


@dataclass
class Gaussians:
    means: np.ndarray      # (N,3)
    scales: np.ndarray     # (N,3) log-scale
    quats: np.ndarray      # (N,4) wxyz (not necessarily normalised)
    opacities: np.ndarray  # (N,) logit
    sh0: np.ndarray        # (N,3) DC coefficients
    shN: np.ndarray        # (N,K,3) higher-order coefficients (K may be 0)

    def __len__(self) -> int:
        return len(self.means)

    @property
    def sh_degree(self) -> int:
        k = self.shN.shape[1] if self.shN.ndim == 3 else 0
        return {0: 0, 3: 1, 8: 2, 15: 3}.get(k, int(round((k + 1) ** 0.5)) - 1)

    @property
    def alpha(self) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-self.opacities))

    @property
    def rgb(self) -> np.ndarray:
        return np.clip(0.5 + SH_C0 * self.sh0, 0.0, 1.0)

    def select(self, mask: np.ndarray) -> "Gaussians":
        return Gaussians(self.means[mask], self.scales[mask], self.quats[mask],
                         self.opacities[mask], self.sh0[mask], self.shN[mask])

    def transformed(self, R: np.ndarray, t: np.ndarray, s: float = 1.0) -> "Gaussians":
        """Apply the similarity x' = s * R x + t to every gaussian."""
        from scipy.spatial.transform import Rotation

        means = (self.means @ R.T) * s + t
        scales = self.scales + np.log(s)
        q = self.quats / np.maximum(np.linalg.norm(self.quats, axis=1, keepdims=True), 1e-12)
        rot = Rotation.from_quat(q[:, [1, 2, 3, 0]])  # scipy wants xyzw
        rot2 = Rotation.from_matrix(R) * rot
        q2 = rot2.as_quat()[:, [3, 0, 1, 2]]
        return Gaussians(means, scales, q2, self.opacities.copy(), self.sh0.copy(), self.shN.copy())


def read_gaussian_ply(path) -> Gaussians:
    d = read_ply(path)["vertex"]
    n = len(d["x"])
    means = np.stack([d["x"], d["y"], d["z"]], 1).astype(np.float32)
    scales = np.stack([d[f"scale_{i}"] for i in range(3)], 1).astype(np.float32)
    quats = np.stack([d[f"rot_{i}"] for i in range(4)], 1).astype(np.float32)
    opac = d["opacity"].astype(np.float32)
    sh0 = np.stack([d[f"f_dc_{i}"] for i in range(3)], 1).astype(np.float32)
    rest = sorted((k for k in d if k.startswith("f_rest_")), key=lambda k: int(k.split("_")[-1]))
    if rest:
        r = np.stack([d[k] for k in rest], 1).astype(np.float32)  # (N, 3K) channel-major
        K = r.shape[1] // 3
        shN = r.reshape(n, 3, K).transpose(0, 2, 1)  # (N,K,3)
    else:
        shN = np.zeros((n, 0, 3), np.float32)
    return Gaussians(means, scales, quats, opac, sh0, shN)


def write_gaussian_ply(path, g: Gaussians) -> None:
    n = len(g)
    v: dict[str, np.ndarray] = {
        "x": g.means[:, 0], "y": g.means[:, 1], "z": g.means[:, 2],
        "nx": np.zeros(n, np.float32), "ny": np.zeros(n, np.float32), "nz": np.zeros(n, np.float32),
    }
    for i in range(3):
        v[f"f_dc_{i}"] = g.sh0[:, i]
    K = g.shN.shape[1] if g.shN.ndim == 3 else 0
    if K:
        r = g.shN.transpose(0, 2, 1).reshape(n, 3 * K)  # channel-major like inria
        for i in range(3 * K):
            v[f"f_rest_{i}"] = r[:, i]
    v["opacity"] = g.opacities
    for i in range(3):
        v[f"scale_{i}"] = g.scales[:, i]
    for i in range(4):
        v[f"rot_{i}"] = g.quats[:, i]
    write_ply(path, {k: np.asarray(a, dtype=np.float32) for k, a in v.items()})
