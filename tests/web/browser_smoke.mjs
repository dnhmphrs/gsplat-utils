// Browser smoke test (manual): builds a synthetic site, serves it, opens it in headless Chromium
// via Playwright and checks the page boots, the loader + sort worker run in a real browser, and
// (when WebGPU is available) that a transition renders. Run: node tests/web/browser_smoke.mjs
import { spawn, execFileSync } from 'node:child_process';
import { mkdtempSync, writeFileSync, mkdirSync, existsSync } from 'node:fs';
import { tmpdir } from 'node:os';
import path from 'node:path';
import { createRequire } from 'node:module';
import { fileURLToPath } from 'node:url';

const require = createRequire(import.meta.url);
const { chromium } = require('playwright');
const here = path.dirname(fileURLToPath(import.meta.url));
const repo = path.resolve(here, '../..');

// --- synthetic work dir: axis triad + random cloud, 3 cameras ---------------------------------
const work = mkdtempSync(path.join(tmpdir(), 'gsu-smoke-'));
mkdirSync(path.join(work, 'splat'), { recursive: true });
const n = 6000;
const buf = new ArrayBuffer(n * 32);
const f32 = new Float32Array(buf), u8 = new Uint8Array(buf);
let i = 0;
const put = (p, s, rgb, a = 255) => {
  f32.set([...p, ...s], i * 8);
  u8.set([...rgb, a, 255, 128, 128, 128], i * 32 + 24); // identity quat (w=1)
  i++;
};
for (let k = 0; k < 400; k++) put([k * 0.01, 0, 0], [0.02, 0.02, 0.02], [255, 40, 40]);   // +x red
for (let k = 0; k < 400; k++) put([0, k * 0.01, 0], [0.02, 0.02, 0.02], [40, 255, 40]);   // +y green
for (let k = 0; k < 400; k++) put([0, 0, k * 0.01], [0.02, 0.02, 0.02], [60, 120, 255]);  // +z blue
while (i < n) put([Math.random() * 6 - 3, Math.random() * 6 - 3, Math.random() * 2], [0.04, 0.04, 0.04], [200, 200, 200], 180);
writeFileSync(path.join(work, 'splat', 'model.splat'), Buffer.from(buf));
const { lookRotation, matrixFromQuat } = await import('../../gsplat_utils/web/template/transitions.js');
// three cameras looking at the triad at the origin (OpenCV axes: x right, y down, z forward)
const camDefs = [
  { pos: [0, -4, 1.5], fwd: [0, 1, -0.2] },   // from -y: expect red (+x) to the right, blue (+z) up, green receding
  { pos: [4, 0, 1.5], fwd: [-1, 0, -0.2] },   // from +x: expect green (+y) to the right, blue up
  { pos: [-3, -3, 3], fwd: [1, 1, -0.8] },    // oblique
];
const cams = camDefs.map((d, j) => {
  const R = matrixFromQuat(lookRotation(d.fwd, [0, 0, 1]));
  const c2w = [[...R[0], d.pos[0]], [...R[1], d.pos[1]], [...R[2], d.pos[2]], [0, 0, 0, 1]];
  return { name: `frame_${String(j * 10).padStart(6, '0')}.jpg`, frame_index: j * 10, time_s: j, width: 848, height: 480,
    fx: 600, fy: 600, cx: 424, cy: 240, model: 'PINHOLE', params: [], camera_to_world: c2w };
});
writeFileSync(path.join(work, 'cameras.json'), JSON.stringify({ frame: 'colmap', images: cams }));
execFileSync('gsu', ['poses', '--work', work, '--set', 'studio', 'frame_000000.jpg', '--set', 'about', '10', '--set', 'shop', '2.0'], { stdio: 'inherit' });
execFileSync('gsu', ['web', '--work', work, '--title', 'smoke'], { stdio: 'inherit' });
const site = path.join(work, 'web');

// --- serve + open -----------------------------------------------------------------------------
const port = 8765 + Math.floor(Math.random() * 200);
const server = spawn('python3', ['-m', 'http.server', String(port), '--directory', site], { stdio: 'ignore' });
await new Promise((r) => setTimeout(r, 800));
const browser = await chromium.launch({ args: ['--enable-unsafe-webgpu', '--ignore-gpu-blocklist', '--use-angle=swiftshader', '--enable-features=Vulkan', '--use-vulkan=swiftshader'] });
const page = await browser.newPage({ viewport: { width: 900, height: 600 } });
const errors = [];
page.on('console', (m) => { if (m.type() === 'error') errors.push(m.text()); });
page.on('pageerror', (e) => errors.push(`pageerror: ${e.message}`));
await page.goto(`http://localhost:${port}/#about`);
await page.waitForTimeout(1500);
const state = await page.evaluate(() => ({
  gpu: !!navigator.gpu,
  navButtons: [...document.querySelectorAll('.nav-btn')].map((b) => b.textContent.trim()),
  message: document.getElementById('message').hidden ? null : document.getElementById('message').textContent,
  title: document.title,
}));
console.log('page state', JSON.stringify(state));
// loader + worker in the real browser, independent of WebGPU
const loaderResult = await page.evaluate(async () => {
  const { loadModel } = await import('./splat-loader.js');
  const { Camera } = await import('./camera.js');
  const { lookRotation } = await import('./transitions.js');
  const m = await loadModel('model.splat');
  const w = new Worker('sort-worker.js');
  const msgs = [];
  const done = new Promise((res) => { w.onmessage = (e) => { msgs.push(e.data.type); if (e.data.type === 'sorted') res(e.data.indices.length); }; });
  w.postMessage({ type: 'init', positions: m.positions });
  const cam = new Camera([0, 0, 1]);
  cam.setPose({ position: [0, -5, 1], quaternion: lookRotation([0, 1, 0], [0, 0, 1]), fov: 60 });
  w.postMessage({ type: 'sort', view: cam.viewMatrix() });
  const sortedLen = await done;
  return { count: m.count, kind: m.kind, bounds: m.bounds.center.map((v) => +v.toFixed(2)), msgs, sortedLen };
});
console.log('loader+worker', JSON.stringify(loaderResult));
let rendered = null;
if (state.gpu && !state.message) {
  await page.waitForFunction(() => window.viewer && window.viewer.renderer && window.viewer.renderer.count > 0, null, { timeout: 20000 });
  const shots = path.join(work, 'shots'); mkdirSync(shots);
  await page.evaluate(() => window.viewer.jumpTo('studio'));
  await page.waitForTimeout(600);
  await page.screenshot({ path: path.join(shots, 'pose1-studio.png') });
  await page.evaluate(() => window.viewer.jumpTo('about'));
  await page.waitForTimeout(600);
  await page.screenshot({ path: path.join(shots, 'pose2-about.png') });
  await page.evaluate(() => window.viewer.goTo('shop', { duration_ms: 4000, effect: 'scatter', effect_strength: 1, mode: 'arc', easing: 'linear' }));
  await page.waitForTimeout(2000);
  await page.screenshot({ path: path.join(shots, 'mid-arc-scatter.png') });
  await page.waitForTimeout(2600);
  await page.screenshot({ path: path.join(shots, 'after-shop.png') });
  await page.evaluate(() => window.viewer.goTo('studio', { duration_ms: 4000, effect: 'dissolve', effect_strength: 1, mode: 'linear', easing: 'linear' }));
  await page.waitForTimeout(2000);
  await page.screenshot({ path: path.join(shots, 'mid-dissolve.png') });
  const active = await page.evaluate(() => document.querySelector('.nav-btn.active')?.dataset.id);
  console.log('active nav during transition:', active, 'hash:', await page.evaluate(() => location.hash));
  await page.waitForTimeout(2600);
  rendered = shots;
}
await browser.close();
server.kill();
console.log(JSON.stringify({ errors, rendered, work }, null, 1));
if (errors.length) process.exit(1);
