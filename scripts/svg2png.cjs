// Rasterise SVG files to PNG with headless Chromium (needs `npm i playwright`).
// usage: node scripts/svg2png.cjs work/geometry/iso.svg work/geometry/plan_all.svg
const { chromium } = require('playwright');
const fs = require('fs');
(async () => {
  const files = process.argv.slice(2);
  const browser = await chromium.launch();
  const page = await browser.newPage({ deviceScaleFactor: 1 });
  for (const f of files) {
    const svg = fs.readFileSync(f, 'utf8');
    const m = svg.match(/width="(\d+)" height="(\d+)"/);
    const w = Math.min(1800, parseInt(m[1])), h = Math.min(1800, parseInt(m[2]));
    await page.setViewportSize({ width: w, height: h });
    await page.setContent(`<html><body style="margin:0">${svg.replace(/width="\d+" height="\d+"/, `width="${w}" height="${h}"`)}</body></html>`);
    await page.screenshot({ path: f.replace(/\.svg$/, '.png') });
    console.log('rendered', f, w, h);
  }
  await browser.close();
})();
