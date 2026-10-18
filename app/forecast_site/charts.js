"use strict";
// Shared by index.html and guide.html. Expects the page to define `lang` and `t(key)`.
const FA_DIGITS = "۰۱۲۳۴۵۶۷۸۹";
// Site numerals (2026-10-16, the author's choice for the site; the thesis text keeps its
// own convention): in fa EVERY number uses Persian digits, «٫» decimal mark, «٪», true minus.
// Numbers are LTR-isolated (.num) so a sign never jumps sides in RTL text.
// The CSV export stays Latin: it is for machines.
const faNum = s => lang === "fa" ? String(s).replace(/\d/g, d => FA_DIGITS[d]).replace(/\./g, "٫").replace(/-/g, "−").replace(/%/g, "٪") : String(s);
const int = n => lang === "fa" ? String(n).replace(/\d/g, d => FA_DIGITS[d]) : String(n);
const dec = (n, d = 2) => (n == null || isNaN(n)) ? "–" : `<span class="num">${faNum(Number(n).toFixed(d))}</span>`;
// axis labels: same digits, true minus sign
const axisInt = v => { const r = Math.round(v); return (r < 0 ? "−" : "") + int(Math.abs(r)); };
const pct = (n, d = 1) => `<span class="num">${faNum(Number(n).toFixed(d) + "%")}</span>`;
const hh = h => lang === "fa" ? int(String(h).padStart(2, "0")) + ":۰۰" : String(h).padStart(2, "0") + ":00";
const esc = s => String(s).replace(/[&<>"]/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
function dayLabel(iso) {
  const d = new Date(iso + "T12:00:00");
  return d.toLocaleDateString(lang === "fa" ? "fa-IR-u-ca-gregory" : "en-GB", { weekday: "long", day: "numeric", month: "long", year: "numeric" });
}

// ---- SVG line / step chart with crosshair tooltip ------------------------------------
// step: each value occupies [i, i+1) -- an hourly price is constant within its hour.
// divider: {at: index, label} draws a vertical boundary (e.g. where tomorrow starts).
function lineChart(el, { x, series, band, yFmt = axisInt, xFmt = v => v, height = 300, xTicks, yDomain, width = 760, step = false, divider }) {
  // draw at the container's real pixel width so text stays 11px on a phone
  // (a fixed viewBox scaled to 390px shrank the labels to ~5px); hidden -> fallback
  const W = el.clientWidth > 0 ? el.clientWidth : width, H = height, m = { l: 48, r: 14, t: divider ? 22 : 10, b: 28 };
  const n = x.length, span = step ? n : Math.max(1, n - 1);
  const vals = series.flatMap(s => s.values.filter(v => v != null));
  if (band) vals.push(...band.lo.filter(v => v != null), ...band.hi.filter(v => v != null));
  let lo = yDomain ? yDomain[0] : Math.min(...vals), hi = yDomain ? yDomain[1] : Math.max(...vals); const pad = (hi - lo) * .08 || 1; lo -= pad; hi += pad;
  const X = i => m.l + (i / span) * (W - m.l - m.r);
  const Y = v => m.t + (1 - (v - lo) / (hi - lo)) * (H - m.t - m.b);
  const cx = i => step ? X(i + .5) : X(i);  // where a point / hover dot sits
  const runs = vs => { const out = []; let cur = null; vs.forEach((v, i) => { if (v == null) { cur = null; return; } if (!cur) out.push(cur = []); cur.push(i); }); return out; };
  const path = vs => runs(vs).map(r => step
    ? r.map((i, k) => `${k ? "L" : "M"}${X(i).toFixed(1)},${Y(vs[i]).toFixed(1)}L${X(i + 1).toFixed(1)},${Y(vs[i]).toFixed(1)}`).join("")
    : r.map((i, k) => `${k ? "L" : "M"}${X(i).toFixed(1)},${Y(vs[i]).toFixed(1)}`).join("")).join("");
  let svg = `<svg viewBox="0 0 ${W} ${H}" role="img">`;
  for (const v of niceTicks(lo, hi, 5)) svg += `<line x1="${m.l}" x2="${W - m.r}" y1="${Y(v)}" y2="${Y(v)}" stroke="var(--grid)"/>` +
    `<text x="${m.l - 8}" y="${Y(v) + 4}" text-anchor="end" direction="ltr" font-size="11" fill="var(--text-muted)">${esc(yFmt(v))}</text>`;
  if (lo < 0 && hi > 0) svg += `<line x1="${m.l}" x2="${W - m.r}" y1="${Y(0)}" y2="${Y(0)}" stroke="var(--axis)"/>`;
  const xt0 = xTicks || x.map((_, i) => i).filter(i => i % Math.ceil(n / 8) === 0);
  // keep a tick only if the previous label (est. 6.5px/char at 11px) has room; no collisions
  const xt = []; for (const i of xt0) if (!xt.length || X(i) - X(xt[xt.length - 1]) >= String(xFmt(x[xt[xt.length - 1]])).length * 6.5 + 12) xt.push(i);
  for (const i of xt) svg += `<text x="${X(i)}" y="${H - 8}" text-anchor="${step ? "start" : "middle"}" font-size="11" fill="var(--text-muted)">${esc(xFmt(x[i]))}</text>`;
  if (band) for (const r of runs(band.lo)) {
    const pts = (arr, rev) => (rev ? [...r].reverse() : r).flatMap(i => step ? (rev ? [[X(i + 1), arr[i]], [X(i), arr[i]]] : [[X(i), arr[i]], [X(i + 1), arr[i]]]) : [[X(i), arr[i]]]);
    const ring = [...pts(band.hi, false), ...pts(band.lo, true)].map(([px, v]) => `${px.toFixed(1)},${Y(v).toFixed(1)}`).join("L");
    svg += `<path d="M${ring}Z" fill="var(--series-1-band)"/>`;
  }
  if (divider) svg += `<line x1="${X(divider.at)}" x2="${X(divider.at)}" y1="${m.t - 6}" y2="${H - m.b}" stroke="var(--text-muted)" stroke-dasharray="2 3"/>` +
    `<text x="${X(divider.at) + 6}" y="${m.t - 9}" font-size="11" font-weight="600" fill="var(--text-secondary)">${esc(divider.label)}</text>`;
  for (const s of series) svg += `<path d="${path(s.values)}" fill="none" stroke="${s.color}" stroke-width="2" ${s.dash ? 'stroke-dasharray="5 4"' : ""} stroke-linejoin="round" stroke-linecap="round"/>`;
  svg += `<line class="xh" y1="${m.t}" y2="${H - m.b}" stroke="var(--axis)" stroke-dasharray="3 3" visibility="hidden"/>`;
  for (const [k, s] of series.entries()) svg += `<circle class="dot d${k}" r="4.5" fill="${s.color}" stroke="var(--surface-1)" stroke-width="2" visibility="hidden"/>`;
  svg += `<rect x="${m.l}" y="0" width="${W - m.l - m.r}" height="${H}" fill="transparent"/></svg><div class="tip"></div>`;
  el.innerHTML = svg;
  const sv = el.querySelector("svg"), tip = el.querySelector(".tip"), xh = el.querySelector(".xh");
  const show = evt => {
    const r = sv.getBoundingClientRect(), px = (evt.clientX - r.left) * W / r.width, f = (px - m.l) / (W - m.l - m.r) * span;
    const i = Math.max(0, Math.min(n - 1, step ? Math.floor(f) : Math.round(f)));
    xh.setAttribute("x1", cx(i)); xh.setAttribute("x2", cx(i)); xh.setAttribute("visibility", "visible");
    let rows = "";
    series.forEach((s, k) => {
      const dot = el.querySelector(`.d${k}`), v = s.values[i];
      if (v == null) { dot.setAttribute("visibility", "hidden"); return; }
      dot.setAttribute("cx", cx(i)); dot.setAttribute("cy", Y(v)); dot.setAttribute("visibility", "visible");
      rows += `<div class="row"><span class="k">${esc(s.name)}</span><span>${dec(v)}</span></div>`;
    });
    if (band && band.lo[i] != null) rows += `<div class="row"><span class="k">${esc(t("band"))}</span><span>${dec(band.lo[i], 0)}–${dec(band.hi[i], 0)}</span></div>`;
    tip.innerHTML = `<b>${esc(xFmt(x[i]))}</b>${rows}`; tip.style.display = "block";
    const left = cx(i) / W * r.width; tip.style.left = Math.min(r.width - tip.offsetWidth - 4, Math.max(0, left + 12)) + "px"; tip.style.top = "8px";
  };
  sv.addEventListener("mousemove", show);
  sv.addEventListener("mouseleave", () => { tip.style.display = "none"; xh.setAttribute("visibility", "hidden"); el.querySelectorAll(".dot").forEach(d => d.setAttribute("visibility", "hidden")); });
}
function niceTicks(lo, hi, n) {
  const step0 = (hi - lo) / n, mag = 10 ** Math.floor(Math.log10(step0)), f = step0 / mag;
  const step = (f < 1.5 ? 1 : f < 3 ? 2 : f < 7 ? 5 : 10) * mag, out = [];
  for (let v = Math.ceil(lo / step) * step; v <= hi; v += step) out.push(+v.toFixed(6));
  return out;
}
const legend = items => `<div class="legend">${items.map(([cls, color, label]) => `<span><i class="${cls}" style="border-color:${color}"></i>${esc(label)}</span>`).join("")}</div>`;

