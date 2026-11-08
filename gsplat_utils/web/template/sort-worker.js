// sort-worker.js — depth sort of splat indices (back to front) with a 16-bit counting sort.
// Messages in:  {type:'init', positions: Float32Array(n*3)}
//               {type:'sort', view: Float32Array(16) column-major world->camera, buffer?: ArrayBuffer}
// Messages out: {type:'ready', count}
//               {type:'sorted', indices: Uint32Array(n), view}
let positions = null;
let n = 0;
let depth = null;     // Float32Array
let keys = null;      // Uint32Array
let counts = new Uint32Array(65536);

self.onmessage = (e) => {
  const m = e.data;
  if (m.type === 'init') {
    positions = m.positions;
    n = positions.length / 3;
    depth = new Float32Array(n);
    keys = new Uint32Array(n);
    self.postMessage({ type: 'ready', count: n });
    return;
  }
  if (m.type === 'sort') {
    if (!positions) return;
    const v = m.view;
    const a = v[2], b = v[6], c = v[10], d = v[14]; // camera-space z = row 2 of the view matrix
    let min = Infinity, max = -Infinity;
    for (let i = 0; i < n; i++) {
      const z = a * positions[i * 3] + b * positions[i * 3 + 1] + c * positions[i * 3 + 2] + d;
      depth[i] = z;
      if (z < min) min = z;
      if (z > max) max = z;
    }
    const indices = m.buffer && m.buffer.byteLength === n * 4 ? new Uint32Array(m.buffer) : new Uint32Array(n);
    const range = max - min;
    if (!(range > 0)) {
      for (let i = 0; i < n; i++) indices[i] = i;
    } else {
      counts.fill(0);
      const scale = 65535 / range;
      for (let i = 0; i < n; i++) {
        const k = Math.min(65535, Math.max(0, ((depth[i] - min) * scale) | 0));
        keys[i] = k;
        counts[k]++;
      }
      // exclusive prefix sums -> start offsets (ascending depth)
      let sum = 0;
      for (let k = 0; k < 65536; k++) { const cnt = counts[k]; counts[k] = sum; sum += cnt; }
      // place far splats first: ascending rank j -> slot n-1-j
      for (let i = 0; i < n; i++) {
        const j = counts[keys[i]]++;
        indices[n - 1 - j] = i;
      }
    }
    self.postMessage({ type: 'sorted', indices, view: v }, [indices.buffer]);
  }
};
