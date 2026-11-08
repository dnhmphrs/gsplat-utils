// ui.js — navigation bar, transition options panel, status, pose sharing.
import { MODES, EFFECTS, EASINGS } from './transitions.js';

const $ = (id) => document.getElementById(id);

export class UI {
  constructor(viewer, config) {
    this.viewer = viewer;
    this.config = config;
    this.hidden = false;
    $('title').textContent = config.title || 'gsplat';
    this.buildNav();
    this.buildOptions();
    this.bindKeys();
    $('btn-options').addEventListener('click', () => this.togglePanel('options'));
    $('btn-share').addEventListener('click', () => this.sharePose());
    $('btn-help').addEventListener('click', () => this.togglePanel('help'));
    for (const b of document.querySelectorAll('[data-close]')) b.addEventListener('click', () => this.togglePanel(b.dataset.close, false));
    $('share-copy').addEventListener('click', () => this.copyShare());
    window.addEventListener('hashchange', () => {
      const id = decodeURIComponent(location.hash.slice(1));
      if (id && id !== this.viewer.currentPoseId && this.viewer.poseById(id)) this.viewer.goTo(id);
    });
  }

  // --- navigation ------------------------------------------------------------
  buildNav() {
    const nav = $('nav');
    nav.innerHTML = '';
    const poses = this.viewer.poses;
    if (!poses.length) {
      nav.innerHTML = '<span class="hint">No poses yet — press P to copy the current view, then <code>gsu poses</code>.</span>';
      return;
    }
    const prev = document.createElement('button');
    prev.className = 'nav-arrow'; prev.textContent = '‹'; prev.title = 'previous (←)';
    prev.addEventListener('click', () => this.viewer.nextPose(-1));
    nav.appendChild(prev);
    poses.forEach((p, i) => {
      const b = document.createElement('button');
      b.className = 'nav-btn';
      b.dataset.id = p.id;
      b.innerHTML = `<span class="k">${i < 9 ? i + 1 : ''}</span>${p.label || p.id}`;
      b.addEventListener('click', () => this.viewer.goTo(p.id));
      nav.appendChild(b);
    });
    const next = document.createElement('button');
    next.className = 'nav-arrow'; next.textContent = '›'; next.title = 'next (→)';
    next.addEventListener('click', () => this.viewer.nextPose(1));
    nav.appendChild(next);
    if (this.config.poses_auto) {
      const h = document.createElement('div');
      h.className = 'hint';
      h.textContent = 'automatic viewpoints — name your own with: gsu poses --set studio <time|frame|image>';
      nav.appendChild(h);
    }
  }
  setActivePose(id) {
    for (const b of document.querySelectorAll('.nav-btn')) b.classList.toggle('active', b.dataset.id === id);
  }

  // --- options panel ---------------------------------------------------------
  buildOptions() {
    const o = this.viewer.transitionOptions;
    const root = $('options-form');
    root.innerHTML = '';
    const row = (label, el, help = '') => {
      const l = document.createElement('label');
      l.innerHTML = `<span>${label}</span>`;
      l.appendChild(el);
      if (help) { const s = document.createElement('small'); s.textContent = help; l.appendChild(s); }
      root.appendChild(l);
      return el;
    };
    const select = (key, values, onchange) => {
      const s = document.createElement('select');
      for (const v of values) { const op = document.createElement('option'); op.value = v; op.textContent = v; s.appendChild(op); }
      s.value = String(o[key]);
      if (![...s.options].some((op) => op.value === s.value)) { const op = document.createElement('option'); op.value = s.value; op.textContent = s.value; s.appendChild(op); s.value = String(o[key]); }
      s.addEventListener('change', () => { o[key] = s.value; onchange && onchange(); });
      return s;
    };
    const number = (key, min, max, step, scale = 1) => {
      const wrap = document.createElement('span');
      wrap.className = 'range';
      const r = document.createElement('input');
      r.type = 'range'; r.min = min; r.max = max; r.step = step; r.value = Number(o[key]) * scale;
      const out = document.createElement('output');
      out.textContent = r.value;
      r.addEventListener('input', () => { o[key] = Number(r.value) / scale; out.textContent = r.value; });
      wrap.appendChild(r); wrap.appendChild(out);
      return wrap;
    };
    const check = (key) => {
      const c = document.createElement('input');
      c.type = 'checkbox'; c.checked = !!o[key];
      c.addEventListener('change', () => { o[key] = c.checked; });
      return c;
    };
    row('mode', select('mode', MODES, () => this.refreshModeRows()), 'how the camera travels');
    row('easing', select('easing', EASINGS.filter((e) => e !== 'cubicBezier').concat(['cubicBezier(0.65,0,0.35,1)'])));
    row('duration (ms)', number('duration_ms', 200, 8000, 50));
    row('duration / metre (ms)', number('duration_per_metre', 0, 2000, 10), '0 = fixed duration');
    row('effect', select('effect', EFFECTS), 'what the splats do while moving');
    row('effect strength', number('effect_strength', 0, 2, 0.05));
    row('keep horizon level', check('up_lock'));
    row('fov pulse (°)', number('fov_pulse', -30, 30, 1), 'dolly-zoom flavour');
    this.modeRows = {
      arc: [row('arc side', number('arc_side', -1, 1, 0.05)), row('arc height', number('arc_height', -1, 1, 0.05)), row('arc look at pivot', number('arc_look', 0, 1, 0.05))],
      dolly: [row('dolly overlap', number('dolly_overlap', 0, 1, 0.05)), row('dolly order', select('dolly_order', ['move-then-turn', 'turn-then-move']))],
      route: [row('route blend', number('route_blend', 0, 1, 0.05)), row('look ahead (m)', number('look_ahead', 0.2, 6, 0.1)),
        row('turn-in fraction', number('look_blend_in', 0.02, 0.6, 0.01)), row('settle fraction', number('look_blend_out', 0.05, 0.8, 0.01))],
    };
    const sz = document.createElement('span'); sz.className = 'range';
    const r = document.createElement('input'); r.type = 'range'; r.min = 0.2; r.max = 4; r.step = 0.05; r.value = 1;
    const out = document.createElement('output'); out.textContent = '1';
    r.addEventListener('input', () => { this.viewer.sizeScale = Number(r.value); out.textContent = r.value; this.viewer.needsRender = true; });
    sz.appendChild(r); sz.appendChild(out);
    row('splat size', sz, 'global size multiplier');
    const replay = document.createElement('button');
    replay.type = 'button'; replay.textContent = 'replay last transition';
    replay.addEventListener('click', () => this.replay());
    root.appendChild(replay);
    const exp = document.createElement('button');
    exp.type = 'button'; exp.textContent = 'copy transition defaults as JSON';
    exp.addEventListener('click', () => this.copyText(JSON.stringify({ default: this.viewer.transitionOptions }, null, 2), exp));
    root.appendChild(exp);
    if (!this.viewer.route) {
      const note = document.createElement('small');
      note.className = 'hint';
      note.textContent = 'no walk route in viewer-config.json: "route" mode falls back to spline (run gsu geometry, then gsu web)';
      root.appendChild(note);
    }
    this.refreshModeRows();
  }
  refreshModeRows() {
    const mode = this.viewer.transitionOptions.mode;
    for (const [m, els] of Object.entries(this.modeRows)) for (const el of els) el.parentElement.hidden = m !== mode;
  }
  replay() {
    const v = this.viewer;
    const ids = v.poses.map((p) => p.id);
    if (ids.length < 2) return;
    const cur = v.currentPoseId || ids[0];
    const i = ids.indexOf(cur);
    const from = ids[(i - 1 + ids.length) % ids.length];
    v.jumpTo(from);
    setTimeout(() => v.goTo(cur), 250);
  }

  // --- panels / keys -----------------------------------------------------------
  togglePanel(name, force) {
    const el = $(`panel-${name}`);
    const show = force === undefined ? el.hidden : force;
    for (const p of document.querySelectorAll('.panel')) p.hidden = true;
    el.hidden = !show;
  }
  bindKeys() {
    window.addEventListener('keydown', (e) => {
      if (e.target && (e.target.tagName === 'INPUT' || e.target.tagName === 'SELECT' || e.target.tagName === 'TEXTAREA')) return;
      if (e.metaKey || e.ctrlKey || e.altKey) return;
      const v = this.viewer;
      if (e.key >= '1' && e.key <= '9') { const p = v.poses[Number(e.key) - 1]; if (p) v.goTo(p.id); }
      else if (e.key === 'ArrowRight') v.nextPose(1);
      else if (e.key === 'ArrowLeft') v.nextPose(-1);
      else if (e.key === 'p' || e.key === 'P') this.sharePose();
      else if (e.key === 'h' || e.key === 'H') this.toggleUI();
      else if (e.key === 'o' || e.key === 'O') this.togglePanel('options');
      else if (e.key === 'r' || e.key === 'R') { if (v.currentPoseId) v.jumpTo(v.currentPoseId); else if (v.poses[0]) v.jumpTo(v.poses[0].id); else v.overview(); }
      else if (e.key === 'Escape') for (const p of document.querySelectorAll('.panel')) p.hidden = true;
      else return;
      e.preventDefault();
    });
  }
  toggleUI() {
    this.hidden = !this.hidden;
    $('ui').classList.toggle('hidden', this.hidden);
  }

  // --- share pose -----------------------------------------------------------------
  sharePose() {
    const json = JSON.stringify(this.viewer.currentPoseJSON(), null, 2);
    $('share-json').value = json;
    this.togglePanel('share', true);
    this.copyText(json, $('share-copy'));
  }
  copyShare() { this.copyText($('share-json').value, $('share-copy')); }
  async copyText(text, btn) {
    try {
      await navigator.clipboard.writeText(text);
      if (btn) { const old = btn.textContent; btn.textContent = 'copied'; setTimeout(() => (btn.textContent = old), 1200); }
    } catch (_) { /* clipboard may be blocked; the textarea is selectable */ }
  }

  // --- status / loading ----------------------------------------------------------
  setProgress(received, total, label) {
    const el = $('loading');
    el.hidden = false;
    const pct = total ? Math.round((received / total) * 100) : null;
    $('loading-bar').style.width = pct === null ? '30%' : `${pct}%`;
    $('loading-text').textContent = pct === null ? `${label} (${(received / 1048576).toFixed(1)} MB)` : `${label} ${pct}%`;
  }
  hideLoading() { $('loading').hidden = true; }
  showMessage(html) {
    const m = $('message');
    m.hidden = false;
    m.innerHTML = html;
    this.hideLoading();
  }
  setModelInfo(model) {
    this.modelInfo = `${model.count.toLocaleString()} ${model.kind === 'point-ply' ? 'points' : 'splats'}`;
  }
  updateStatus({ fps, count, pose, transition }) {
    const t = transition ? ` · ${transition.opts.mode} ${(transition.t * 100) | 0}%` : '';
    $('status').textContent = `${this.modelInfo || count} · ${fps} fps${t}`;
  }
}
