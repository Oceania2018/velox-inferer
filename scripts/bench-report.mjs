#!/usr/bin/env node
// Renders a benchmark run's speed-test results (scripts/speed-test.mjs --json, plus `title`,
// `date` and `setup` added by hand) as a self-contained HTML report: grouped bar charts drawn
// inline (one per metric and output length; groups are the concurrency levels, bars the input
// sizes) with a metric/output switcher, the setup and the full table. No external files.
//
// usage: node scripts/bench-report.mjs <run-dir>
//   reads <run-dir>/results.json, writes <run-dir>/index.html

import { readFileSync, writeFileSync } from "node:fs";
import { join } from "node:path";

const [runDir] = process.argv.slice(2);
if (!runDir) {
  console.error("usage: node scripts/bench-report.mjs <run-dir>");
  process.exit(2);
}
const doc = JSON.parse(readFileSync(join(runDir, "results.json"), "utf8"));
const rows = doc.results;

const METRICS = {
  throughput: { key: "throughput", title: "Throughput, all requests", unit: "tok/s", digits: 0 },
  latency: { key: "latencyS", title: "Latency per request", unit: "s", digits: 2 },
  decode: { key: "decode", title: "Decode rate per request", unit: "tok/s", digits: 0 },
};
// Colour-blind-safe (Okabe-Ito), distinguishable in grey too by lightness.
const COLORS = ["#0072B2", "#E69F00", "#009E73", "#CC79A7"];
const W = 720, H = 360, M = { top: 56, right: 24, bottom: 56, left: 64 };

const esc = (s) => String(s).replace(/&/g, "&amp;").replace(/</g, "&lt;");
const kTok = (n) => (n >= 1024 ? `${n / 1024}K` : `${n}`);

function niceMax(v) {
  const p = 10 ** Math.floor(Math.log10(v || 1));
  return [1, 2, 2.5, 5, 10].map((m) => m * p).find((m) => m >= v);
}

function chart(metric, output) {
  const m = METRICS[metric];
  const data = rows.filter((r) => r.output === output);
  const levels = [...new Set(data.map((r) => r.concurrency))].sort((a, b) => a - b);
  const inputs = [...new Set(data.map((r) => r.input))].sort((a, b) => a - b);
  const value = (c, i) => data.find((r) => r.concurrency === c && r.input === i)?.[m.key];
  const max = niceMax(Math.max(...data.map((r) => r[m.key]).filter(Number.isFinite)));
  const pw = W - M.left - M.right, ph = H - M.top - M.bottom;
  const gw = pw / levels.length, bw = Math.min(40, (gw * 0.8) / inputs.length);
  const y = (v) => M.top + ph - (v / max) * ph;
  const out = [];
  out.push(
    `<svg xmlns="http://www.w3.org/2000/svg" width="${W}" height="${H}" viewBox="0 0 ${W} ${H}" ` +
      `font-family="system-ui, -apple-system, Segoe UI, sans-serif" role="img" ` +
      `aria-label="${esc(`${m.title} (${m.unit}) by concurrency, ${output} output tokens`)}">`,
    // Opaque background: readable on GitHub's dark theme too.
    `<rect width="${W}" height="${H}" fill="#ffffff"/>`,
    `<text x="${M.left}" y="24" font-size="16" font-weight="600" fill="#1f2328">${esc(m.title)} (${m.unit}), ${output} output tokens</text>`,
  );
  // Legend.
  inputs.forEach((inp, k) => {
    const lx = M.left + k * 110;
    out.push(
      `<rect x="${lx}" y="34" width="12" height="12" fill="${COLORS[k % COLORS.length]}"/>`,
      `<text x="${lx + 18}" y="44" font-size="12" fill="#1f2328">${kTok(inp)} input</text>`,
    );
  });
  // Grid and y axis.
  for (let t = 0; t <= 4; t++) {
    const v = (max * t) / 4, yy = y(v);
    out.push(
      `<line x1="${M.left}" x2="${W - M.right}" y1="${yy}" y2="${yy}" stroke="#d0d7de" stroke-width="1"/>`,
      `<text x="${M.left - 8}" y="${yy + 4}" font-size="11" text-anchor="end" fill="#57606a">${+v.toFixed(m.digits)}</text>`,
    );
  }
  // Bars.
  levels.forEach((c, g) => {
    const gx = M.left + g * gw + (gw - bw * inputs.length) / 2;
    inputs.forEach((inp, k) => {
      const v = value(c, inp);
      if (!Number.isFinite(v)) return;
      const x = gx + k * bw, yy = y(v);
      out.push(
        `<rect x="${x + 1}" y="${yy}" width="${bw - 2}" height="${M.top + ph - yy}" fill="${COLORS[k % COLORS.length]}">` +
          `<title>${c} at once, ${kTok(inp)} input: ${v.toFixed(m.digits)} ${m.unit}</title></rect>`,
        `<text x="${x + bw / 2}" y="${yy - 4}" font-size="10" text-anchor="middle" fill="#1f2328">${v.toFixed(m.digits)}</text>`,
      );
    });
    out.push(
      `<text x="${M.left + g * gw + gw / 2}" y="${M.top + ph + 18}" font-size="12" text-anchor="middle" fill="#1f2328">${c}</text>`,
    );
  });
  out.push(
    `<text x="${M.left + pw / 2}" y="${H - 12}" font-size="12" text-anchor="middle" fill="#57606a">requests at once</text>`,
    `</svg>`,
  );
  return out.join("\n");
}

const outputs = [...new Set(rows.map((r) => r.output))].sort((a, b) => a - b);
const svgs = {};
for (const metric of Object.keys(METRICS)) {
  for (const o of outputs) svgs[`${metric}-${o}`] = chart(metric, o);
}
const out = join(runDir, "index.html");
writeFileSync(out, report(doc, svgs, outputs));
console.log(out);

function report(doc, svgs, outputs) {
  const num = (v, d = 0) => (Number.isFinite(v) ? v.toLocaleString("en-US", { maximumFractionDigits: d, minimumFractionDigits: d }) : "-");
  const sorted = [...rows].sort((a, b) => a.output - b.output || a.input - b.input || a.concurrency - b.concurrency);
  const setup = Object.entries(doc.setup ?? {})
    .map(([k, v]) => `<tr><th scope="row">${esc(k)}</th><td>${esc(v)}</td></tr>`)
    .join("\n");
  const table = sorted
    .map(
      (r) =>
        `<tr><td>${r.concurrency}</td><td>${kTok(r.input)}</td><td>${r.output}</td><td>${num(r.decode)}</td>` +
        `<td>${num(r.prefillS, 2)}</td><td>${num(r.prefillRate)}</td><td>${num(r.latencyS, 2)}</td>` +
        `<td>${num(r.e2e, 1)}</td><td>${num(r.throughput)}</td></tr>`,
    )
    .join("\n");
  const figures = Object.entries(svgs)
    // Without JavaScript the default chart stays visible.
    .map(([k, svg]) => `<figure data-chart="${k}"${k === `throughput-${outputs[outputs.length - 1]}` ? "" : " hidden"}>${svg}</figure>`)
    .join("\n");
  const radios = (name, items, first) =>
    items
      .map(
        ([v, label]) =>
          `<label><input type="radio" name="${name}" value="${v}"${v === first ? " checked" : ""}> ${esc(label)}</label>`,
      )
      .join("\n");
  const last = String(outputs[outputs.length - 1]);
  return `<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>velox benchmark ${esc(doc.date ?? "")}</title>
<style>
:root { --bg: #ffffff; --fg: #1f2328; --muted: #57606a; --line: #d0d7de; --accent: #0072B2; --row: #f6f8fa; }
@media (prefers-color-scheme: dark) {
  :root { --bg: #0d1117; --fg: #e6edf3; --muted: #9198a1; --line: #30363d; --accent: #4493f8; --row: #161b22; }
}
* { box-sizing: border-box; }
body { margin: 0; background: var(--bg); color: var(--fg); font: 15px/1.5 system-ui, -apple-system, "Segoe UI", sans-serif; }
main { max-width: 980px; margin: 0 auto; padding: 24px 16px 48px; }
h1 { font-size: 24px; margin: 0 0 4px; }
h2 { font-size: 18px; margin: 32px 0 8px; }
.sub { color: var(--muted); margin: 0 0 16px; }
fieldset { border: 1px solid var(--line); border-radius: 6px; padding: 8px 12px; margin: 0 12px 12px 0; display: inline-block; }
legend { color: var(--muted); font-size: 13px; padding: 0 4px; }
label { margin-right: 12px; cursor: pointer; }
input:focus-visible { outline: 2px solid var(--accent); outline-offset: 2px; }
figure { margin: 0; overflow-x: auto; }
figure svg { max-width: 100%; height: auto; border-radius: 6px; }
.scroll { overflow-x: auto; }
table { border-collapse: collapse; width: 100%; font-variant-numeric: tabular-nums; }
th, td { border-bottom: 1px solid var(--line); padding: 6px 8px; text-align: right; white-space: nowrap; }
thead th { position: sticky; top: 0; background: var(--bg); }
tbody tr:nth-child(even) { background: var(--row); }
.setup th, .setup td { text-align: left; white-space: normal; vertical-align: top; }
.setup th { width: 120px; color: var(--muted); font-weight: 600; }
</style>
</head>
<body>
<main>
<h1>${esc(doc.title)}</h1>
<p class="sub">${esc(doc.date ?? "")} · ${esc(doc.model ?? "")} · ${rows.length} configurations, ${esc(doc.rounds)} rounds each</p>

<h2>Charts</h2>
<form id="pick" aria-label="Chart selection">
<fieldset><legend>Metric</legend>
${radios("metric", Object.entries(METRICS).map(([k, m]) => [k, `${m.title} (${m.unit})`]), "throughput")}
</fieldset>
<fieldset><legend>Output tokens</legend>
${radios("output", outputs.map((o) => [String(o), String(o)]), last)}
</fieldset>
</form>
<div aria-live="polite">
${figures}
</div>
<noscript><p>Switching charts needs JavaScript; the table below has every number.</p></noscript>

<h2>Setup</h2>
<div class="scroll"><table class="setup">${setup}</table></div>

<h2>Results</h2>
<p class="sub">Medians per request. Throughput: completion tokens of all requests per second of round wall time.</p>
<div class="scroll"><table>
<thead><tr><th scope="col">At once</th><th scope="col">Input</th><th scope="col">Output</th><th scope="col">Decode tok/s</th><th scope="col">Prefill s</th><th scope="col">Prefill tok/s</th><th scope="col">Latency s</th><th scope="col">End-to-end tok/s</th><th scope="col">Throughput tok/s</th></tr></thead>
<tbody>
${table}
</tbody>
</table></div>
<p class="sub">Method and notes: README.md in this folder.</p>
</main>
<script>
const form = document.getElementById("pick");
function show() {
  const key = form.metric.value + "-" + form.output.value;
  for (const f of document.querySelectorAll("figure[data-chart]")) f.hidden = f.dataset.chart !== key;
}
form.addEventListener("change", show);
show();
</script>
</body>
</html>
`;
}
