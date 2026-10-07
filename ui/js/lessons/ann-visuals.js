// Vẽ SVG cho thuật toán ANN trên dữ liệu 2D: IVF (cụm + nprobe) và HNSW (đồ thị nhiều tầng + đường tìm).
import { esc } from "../util.js";

const CELL_COLORS = ["#2563eb", "#dc2626", "#059669", "#7c3aed", "#d97706", "#0891b2", "#db2777", "#65a30d"];
const S = 360;  // kích thước khung vẽ
const px = (v) => 14 + v * (S - 28);

const star = (x, y, r = 11) => `<polygon points="${Array.from({ length: 10 }, (_, i) => {
  const rad = i % 2 ? r * 0.45 : r, a = (Math.PI / 5) * i - Math.PI / 2;
  return `${x + rad * Math.cos(a)},${y + rad * Math.sin(a)}`;
}).join(" ")}" fill="#111" stroke="#fff" stroke-width="1.5"/>`;

const frame = (inner, label) => `<svg viewBox="0 0 ${S} ${S}" class="toysvg" role="img" aria-label="${esc(label)}"><rect x="1" y="1" width="${S - 2}" height="${S - 2}" rx="8" fill="#fbfcfd" stroke="#dbe1e7"/>${inner}</svg>`;

/** stage: raw | cells | probe | scan | result */
export function ivfSvg(d, stage) {
  const [qx, qy] = d.query.map(px);
  const probed = new Set(d.probed), found = new Set(d.found), exact = new Set(d.exact);
  const dots = d.points.map(([x, y], i) => {
    const c = d.cell[i];
    const inProbe = probed.has(c);
    const color = stage === "raw" ? "#94a3b8" : CELL_COLORS[c % 8];
    const faded = (stage === "scan" || stage === "result") && !inProbe;
    return `<circle cx="${px(x)}" cy="${px(y)}" r="${faded ? 3 : 4}" fill="${color}" opacity="${faded ? 0.18 : 0.85}"/>`;
  }).join("");
  const cents = stage === "raw" ? "" : d.centroids.map(([x, y], c) => `<g><circle cx="${px(x)}" cy="${px(y)}" r="9" fill="#fff" stroke="${CELL_COLORS[c % 8]}" stroke-width="3"/>
    <text x="${px(x)}" y="${px(y) + 4}" font-size="10" text-anchor="middle" font-weight="700">${c}</text></g>`).join("");
  const rings = ["probe", "scan", "result"].includes(stage) ? d.probed.map((c) => `<circle cx="${px(d.centroids[c][0])}" cy="${px(d.centroids[c][1])}" r="46" fill="${CELL_COLORS[c % 8]}" fill-opacity=".08" stroke="${CELL_COLORS[c % 8]}" stroke-dasharray="5 4" stroke-width="2"/>`).join("") : "";
  const marks = stage === "result" ? d.points.map(([x, y], i) => {
    const f = found.has(i), e = exact.has(i);
    if (f && e) return `<circle cx="${px(x)}" cy="${px(y)}" r="9" fill="none" stroke="#15803d" stroke-width="3"/>`;
    if (e) return `<circle cx="${px(x)}" cy="${px(y)}" r="9" fill="none" stroke="#dc2626" stroke-width="3" stroke-dasharray="3 2"/>`;
    return f ? `<circle cx="${px(x)}" cy="${px(y)}" r="9" fill="none" stroke="#d97706" stroke-width="2"/>` : "";
  }).join("") : "";
  const q = stage === "raw" || stage === "cells" ? "" : star(qx, qy);
  return frame(`${rings}${dots}${cents}${marks}${q}`, "IVF 2D");
}

/** Một panel cho mỗi tầng. stage: layers | descent | layer0 | result. */
export function hnswSvg(d, stage) {
  const [qx, qy] = d.query.map(px);
  const exact = new Set(d.exact), found = new Set(d.found), visited = new Set(d.visited);
  const order = new Map(d.layer0_order.map((n, i) => [n, i + 1]));
  const panels = [];
  for (let l = d.top_layer; l >= 0; l--) {
    const L = d.layers[l], nodes = new Set(L.nodes);
    const edges = L.edges.map(([a, b]) => `<line x1="${px(d.points[a][0])}" y1="${px(d.points[a][1])}" x2="${px(d.points[b][0])}" y2="${px(d.points[b][1])}" stroke="#cbd5e1" stroke-width="1"/>`).join("");
    const hop = d.descent.filter((p) => p.layer === l);
    const pathLines = ["descent", "layer0", "result"].includes(stage) ? hop.slice(1).map((p, i) => {
      const a = d.points[hop[i].node], b = d.points[p.node];
      return `<line x1="${px(a[0])}" y1="${px(a[1])}" x2="${px(b[0])}" y2="${px(b[1])}" stroke="#dc2626" stroke-width="3.5"/>`;
    }).join("") : "";
    const dots = L.nodes.map((n) => {
      const [x, y] = d.points[n];
      let fill = "#64748b", r = 3.8;
      if (l === 0 && stage === "layer0" && visited.has(n)) { fill = "#d97706"; r = 5.5; }
      if (l === 0 && stage === "result") { fill = found.has(n) ? (exact.has(n) ? "#15803d" : "#d97706") : exact.has(n) ? "#dc2626" : visited.has(n) ? "#f3c98b" : "#64748b"; r = found.has(n) || exact.has(n) ? 6 : visited.has(n) ? 4.5 : 3.8; }
      const lbl = l === 0 && stage === "layer0" && order.has(n) ? `<text x="${px(x) + 6}" y="${px(y) - 5}" font-size="10" font-weight="700" fill="#92400e">${order.get(n)}</text>` : "";
      return `<circle cx="${px(x)}" cy="${px(y)}" r="${r}" fill="${fill}"/>${lbl}`;
    }).join("");
    const entry = l === d.top_layer && stage !== "layers" ? `<circle cx="${px(d.points[d.entry][0])}" cy="${px(d.points[d.entry][1])}" r="9" fill="none" stroke="#dc2626" stroke-width="2.5"/>` : "";
    const qq = stage === "layers" ? "" : star(qx, qy, 9);
    panels.push(`<div class="layerpanel"><div class="layerlabel">Tầng ${l}${l === d.top_layer ? " (trên cùng, thưa)" : l === 0 ? " (dưới cùng, đủ mọi điểm)" : ""} · ${nodes.size} điểm</div>${frame(`${edges}${pathLines}${dots}${entry}${qq}`, "tầng " + l)}</div>`);
  }
  return `<div class="layers">${panels.join("")}</div>`;
}

export const ivfLegend = `<div class="legend"><span><i class="dot" style="background:#2563eb"></i>màu = cụm (cell)</span><span>◯ có số = tâm cụm (centroid)</span><span>★ = query</span><span>vòng nét đứt = cụm được quét</span></div>`;
export const resultLegend = `<div class="legend"><span><b style="color:#15803d">◯ xanh</b> = tìm đúng láng giềng thật</span><span><b style="color:#dc2626">◌ đỏ nét đứt</b> = láng giềng thật bị BỎ SÓT</span><span><b style="color:#d97706">◯ cam</b> = trả về nhưng không thuộc top thật</span></div>`;
