// splat-loader.js — read .splat (antimatter15 layout), 3DGS .ply and plain point .ply
// into flat typed arrays ready for the GPU. No dependencies.
//
// Output: { count, kind, positions: Float32Array(n*3), scales: Float32Array(n*3),
//           quats: Float32Array(n*4) [x,y,z,w], colors: Uint8Array(n*4) rgba,
//           bounds: {min, max, center, radius} }

const SH_C0 = 0.28209479177387814;

export async function fetchWithProgress(url, onProgress) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`failed to load ${url}: ${res.status} ${res.statusText}`);
  const total = Number(res.headers.get('content-length')) || 0;
  if (!res.body || !res.body.getReader) {
    const buf = await res.arrayBuffer();
    onProgress && onProgress(buf.byteLength, buf.byteLength);
    return buf;
  }
  const reader = res.body.getReader();
  const chunks = [];
  let received = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    received += value.byteLength;
    onProgress && onProgress(received, total);
  }
  const out = new Uint8Array(received);
  let off = 0;
  for (const c of chunks) { out.set(c, off); off += c.byteLength; }
  return out.buffer;
}

export function detectFormat(buffer, url = '') {
  const head = new Uint8Array(buffer, 0, Math.min(4, buffer.byteLength));
  const isPly = head[0] === 0x70 && head[1] === 0x6c && head[2] === 0x79; // "ply"
  if (isPly) return 'ply';
  if (/\.ply(\?|$)/i.test(url)) return 'ply';
  return 'splat';
}

export function parseSplat(buffer) {
  const stride = 32;
  const n = Math.floor(buffer.byteLength / stride);
  const f32 = new Float32Array(buffer, 0, n * 8);
  const u8 = new Uint8Array(buffer, 0, n * stride);
  const positions = new Float32Array(n * 3), scales = new Float32Array(n * 3);
  const quats = new Float32Array(n * 4), colors = new Uint8Array(n * 4);
  for (let i = 0; i < n; i++) {
    positions[i * 3] = f32[i * 8]; positions[i * 3 + 1] = f32[i * 8 + 1]; positions[i * 3 + 2] = f32[i * 8 + 2];
    scales[i * 3] = f32[i * 8 + 3]; scales[i * 3 + 1] = f32[i * 8 + 4]; scales[i * 3 + 2] = f32[i * 8 + 5];
    const b = i * stride + 24;
    colors[i * 4] = u8[b]; colors[i * 4 + 1] = u8[b + 1]; colors[i * 4 + 2] = u8[b + 2]; colors[i * 4 + 3] = u8[b + 3];
    const w = (u8[b + 4] - 128) / 128, x = (u8[b + 5] - 128) / 128, y = (u8[b + 6] - 128) / 128, z = (u8[b + 7] - 128) / 128;
    const l = Math.hypot(x, y, z, w) || 1;
    quats[i * 4] = x / l; quats[i * 4 + 1] = y / l; quats[i * 4 + 2] = z / l; quats[i * 4 + 3] = w / l;
  }
  return { count: n, kind: 'splat', positions, scales, quats, colors };
}

const PLY_TYPES = {
  char: ['Int8', 1], int8: ['Int8', 1], uchar: ['Uint8', 1], uint8: ['Uint8', 1],
  short: ['Int16', 2], int16: ['Int16', 2], ushort: ['Uint16', 2], uint16: ['Uint16', 2],
  int: ['Int32', 4], int32: ['Int32', 4], uint: ['Uint32', 4], uint32: ['Uint32', 4],
  float: ['Float32', 4], float32: ['Float32', 4], double: ['Float64', 8], float64: ['Float64', 8],
};

export function parsePlyHeader(buffer) {
  const u8 = new Uint8Array(buffer);
  const limit = Math.min(u8.length, 1 << 20);
  let text = '';
  for (let i = 0; i < limit; i++) text += String.fromCharCode(u8[i]);
  const end = text.indexOf('end_header');
  if (end < 0) throw new Error('PLY: end_header not found');
  let headerLen = end + 'end_header'.length;
  while (text[headerLen] === ' ' || text[headerLen] === '\t') headerLen++; // tolerate trailing blanks
  if (text[headerLen] === '\r') headerLen++;
  if (text[headerLen] === '\n') headerLen++;
  const lines = text.slice(0, end).split(/\r?\n/);
  let format = 'ascii';
  const elements = [];
  for (const line of lines) {
    const tok = line.trim().split(/\s+/);
    if (tok[0] === 'format') format = tok[1];
    else if (tok[0] === 'element') elements.push({ name: tok[1], count: parseInt(tok[2], 10), props: [] });
    else if (tok[0] === 'property') {
      if (tok[1] === 'list') elements[elements.length - 1].props.push({ name: tok[4], list: true, countType: tok[2], itemType: tok[3] });
      else elements[elements.length - 1].props.push({ name: tok[2], type: tok[1] });
    }
  }
  return { format, elements, headerLen };
}

/** Read a binary PLY vertex element into {name: Float32Array}. */
function readVertexBinary(buffer, header) {
  const vert = header.elements.find((e) => e.name === 'vertex');
  if (!vert) throw new Error('PLY: no vertex element');
  const little = header.format === 'binary_little_endian';
  const n = vert.count;
  const stride = vert.props.reduce((s, p) => s + (p.list ? 0 : PLY_TYPES[p.type][1]), 0);
  if (vert.props.some((p) => p.list)) throw new Error('PLY: list properties in vertex element unsupported');
  const dv = new DataView(buffer, header.headerLen);
  const out = {};
  const allF32 = vert.props.every((p) => PLY_TYPES[p.type][0] === 'Float32') && little && ((header.headerLen % 4) === 0);
  if (allF32) {
    const f = new Float32Array(buffer, header.headerLen, n * vert.props.length);
    const k = vert.props.length;
    vert.props.forEach((p, j) => {
      const a = new Float32Array(n);
      for (let i = 0; i < n; i++) a[i] = f[i * k + j];
      out[p.name] = a;
    });
    return { n, out };
  }
  let offsets = [], off = 0;
  for (const p of vert.props) { offsets.push(off); off += PLY_TYPES[p.type][1]; }
  vert.props.forEach((p, j) => {
    const a = new Float32Array(n);
    const getter = 'get' + PLY_TYPES[p.type][0];
    const o = offsets[j];
    for (let i = 0; i < n; i++) a[i] = dv[getter](i * stride + o, little);
    out[p.name] = a;
  });
  return { n, out };
}

function readVertexAscii(buffer, header) {
  const vert = header.elements.find((e) => e.name === 'vertex');
  const text = new TextDecoder().decode(new Uint8Array(buffer, header.headerLen));
  const n = vert.count;
  const out = {};
  for (const p of vert.props) out[p.name] = new Float32Array(n);
  let i = 0;
  for (const line of text.split('\n')) {
    if (i >= n) break;
    const t = line.trim();
    if (!t) continue;
    const vals = t.split(/\s+/);
    vert.props.forEach((p, j) => { out[p.name][i] = parseFloat(vals[j]); });
    i++;
  }
  return { n, out };
}

export function parsePly(buffer, opts = {}) {
  const header = parsePlyHeader(buffer);
  const { n, out } = header.format === 'ascii' ? readVertexAscii(buffer, header) : readVertexBinary(buffer, header);
  const positions = new Float32Array(n * 3), scales = new Float32Array(n * 3);
  const quats = new Float32Array(n * 4), colors = new Uint8Array(n * 4);
  const gaussian = 'f_dc_0' in out && 'scale_0' in out && 'rot_0' in out;
  const sig = (x) => 1 / (1 + Math.exp(-x));
  const toByte = (v) => Math.max(0, Math.min(255, Math.round(v * 255)));
  const pointSize = opts.pointSize ?? 0.02;
  for (let i = 0; i < n; i++) {
    positions[i * 3] = out.x[i]; positions[i * 3 + 1] = out.y[i]; positions[i * 3 + 2] = out.z[i];
    if (gaussian) {
      scales[i * 3] = Math.exp(out.scale_0[i]); scales[i * 3 + 1] = Math.exp(out.scale_1[i]); scales[i * 3 + 2] = Math.exp(out.scale_2[i]);
      const w = out.rot_0[i], x = out.rot_1[i], y = out.rot_2[i], z = out.rot_3[i];
      const l = Math.hypot(x, y, z, w) || 1;
      quats[i * 4] = x / l; quats[i * 4 + 1] = y / l; quats[i * 4 + 2] = z / l; quats[i * 4 + 3] = w / l;
      colors[i * 4] = toByte(0.5 + SH_C0 * out.f_dc_0[i]);
      colors[i * 4 + 1] = toByte(0.5 + SH_C0 * out.f_dc_1[i]);
      colors[i * 4 + 2] = toByte(0.5 + SH_C0 * out.f_dc_2[i]);
      colors[i * 4 + 3] = toByte(sig(out.opacity[i]));
    } else {
      scales[i * 3] = scales[i * 3 + 1] = scales[i * 3 + 2] = pointSize;
      quats[i * 4 + 3] = 1;
      const has = 'red' in out;
      const r = has ? out.red[i] : 200, g = has ? out.green[i] : 200, b = has ? out.blue[i] : 200;
      const scale = (has && (r > 1 || g > 1 || b > 1)) || !has ? 1 : 255; // float 0..1 colours
      colors[i * 4] = Math.round(r * scale); colors[i * 4 + 1] = Math.round(g * scale); colors[i * 4 + 2] = Math.round(b * scale);
      colors[i * 4 + 3] = 255;
    }
  }
  return { count: n, kind: gaussian ? 'gaussian-ply' : 'point-ply', positions, scales, quats, colors };
}

/** Robust scene bounds: 2nd..98th percentile box on a sample, plus a centre/radius. */
export function computeBounds(positions) {
  const n = positions.length / 3;
  const step = Math.max(1, Math.floor(n / 50000));
  const xs = [], ys = [], zs = [];
  for (let i = 0; i < n; i += step) { xs.push(positions[i * 3]); ys.push(positions[i * 3 + 1]); zs.push(positions[i * 3 + 2]); }
  const q = (arr, p) => { const a = Float32Array.from(arr).sort(); return a[Math.min(a.length - 1, Math.floor(p * (a.length - 1)))]; };
  const min = [q(xs, 0.02), q(ys, 0.02), q(zs, 0.02)];
  const max = [q(xs, 0.98), q(ys, 0.98), q(zs, 0.98)];
  const center = [(min[0] + max[0]) / 2, (min[1] + max[1]) / 2, (min[2] + max[2]) / 2];
  const radius = Math.max(1e-3, 0.5 * Math.hypot(max[0] - min[0], max[1] - min[1], max[2] - min[2]));
  return { min, max, center, radius };
}

export async function loadModel(url, { onProgress, pointSize = 0.02 } = {}) {
  const buffer = await fetchWithProgress(url, onProgress);
  const fmt = detectFormat(buffer, url);
  const model = fmt === 'ply' ? parsePly(buffer, { pointSize }) : parseSplat(buffer);
  model.bounds = computeBounds(model.positions);
  return model;
}
