// Bản đồ 2D (PCA) của tài liệu + query + (tùy chọn) văn bản tự nhập và kết quả của một cấu hình.
import { $, esc, topicColor } from "./util.js";

const W = 640, H = 430, PAD = 22;

function project([x, y], bounds) {
  const [x0, x1, y0, y1] = bounds;
  const cx = Math.min(Math.max(x, x0), x1), cy = Math.min(Math.max(y, y0), y1);  // điểm ngoài khung -> kẹp vào mép
  return [PAD + ((cx - x0) / (x1 - x0)) * (W - 2 * PAD), H - PAD - ((cy - y0) / (y1 - y0)) * (H - 2 * PAD)];
}

function star(cx, cy, r) {
  const pts = Array.from({ length: 10 }, (_, i) => {
    const radius = i % 2 ? r * 0.45 : r, angle = (Math.PI / 5) * i - Math.PI / 2;
    return `${cx + radius * Math.cos(angle)},${cy + radius * Math.sin(angle)}`;
  });
  return `<polygon points="${pts.join(" ")}" fill="#111" stroke="#fff" stroke-width="1.5"/>`;
}

/**
 * Gắn bản đồ vào container. update({queryPoint, extra, tenant, configs}):
 *  - queryPoint: {raw, centered} của query (sao đen)
 *  - extra: [{label, raw, centered}] điểm tự nhập, vẽ số thứ tự
 *  - configs: [{key,label,results}] để nối query tới top-K của cấu hình được chọn
 */
export function mountMap(container, space) {
  container.innerHTML = `<div class="row opts mapctl"></div>
    <svg viewBox="0 0 ${W} ${H}" role="img" aria-label="Bản đồ embedding 2D"></svg><div class="legend maplegend"></div>`;
  const state = { mode: "raw", config: "hybrid_rerank", payload: null };

  function controls() {
    const configs = state.payload.configs;
    $(".mapctl", container).innerHTML = `
      <label>Chế độ <select class="mode"><option value="raw">Gốc (trục lớn nhất = ngôn ngữ)</option>
        <option value="centered">Đã trừ trung bình ngôn ngữ (lộ chủ đề)</option></select></label>
      ${configs ? `<label>Nối query tới top-K của <select class="config">${configs.map((c) =>
        `<option value="${c.key}">${esc(c.label)}</option>`).join("")}</select></label>` : ""}`;
    $(".mode", container).value = state.mode;
    $(".mode", container).onchange = (e) => { state.mode = e.target.value; draw(); };
    const cfg = $(".config", container);
    if (cfg) { cfg.value = state.config; cfg.onchange = (e) => { state.config = e.target.value; draw(); }; }
  }

  function draw() {
    const { mode, payload } = state;
    const bounds = space.bounds[mode];
    const picked = new Set(payload.configs
      ? (payload.configs.find((c) => c.key === state.config) ?? payload.configs[0]).results.map((d) => d.id) : []);
    const [qx, qy] = project(payload.queryPoint[mode], bounds);

    const links = space.docs.filter((d) => picked.has(d.id)).map((d) => {
      const [x, y] = project(d[mode], bounds);
      return `<line x1="${qx}" y1="${qy}" x2="${x}" y2="${y}" stroke="#111" stroke-opacity=".35" stroke-dasharray="4 3"/>`;
    }).join("");

    const dots = space.docs.map((d) => {
      const [x, y] = project(d[mode], bounds);
      const color = topicColor(d.topic), hot = picked.has(d.id);
      const shape = d.language === "vi"
        ? `<circle cx="${x}" cy="${y}" r="${hot ? 8 : 5.5}" fill="${color}"/>`
        : `<rect x="${x - 5}" y="${y - 5}" width="10" height="10" transform="rotate(45 ${x} ${y})" fill="${color}"/>`;
      const ring = hot ? `<circle cx="${x}" cy="${y}" r="12" fill="none" stroke="#111" stroke-width="2"/>` : "";
      const muted = payload.tenant && (d.tenant_id !== payload.tenant || d.status !== "current") ? ' opacity=".45"' : "";
      return `<g${muted}>${ring}${shape}<title>${esc(d.id)} [${esc(d.language)}/${esc(d.tenant_id)}/${esc(d.status)}] ${esc(d.text)}</title></g>`;
    }).join("");

    const extra = (payload.extra ?? []).map((p) => {
      const [x, y] = project(p[mode], bounds);
      return `<g><circle cx="${x}" cy="${y}" r="11" fill="#fff" stroke="#111" stroke-width="2"/>
        <text x="${x}" y="${y + 4}" font-size="12" font-weight="700" text-anchor="middle">${esc(p.label)}</text></g>`;
    }).join("");

    $("svg", container).innerHTML = `${links}${dots}${extra}${star(qx, qy, 14)}
      <text x="${qx + 16}" y="${qy - 10}" font-size="12" font-weight="600">query</text>`;
    const topics = [...new Set(space.docs.map((d) => d.topic))].sort();
    $(".maplegend", container).innerHTML =
      `<span>● tiếng Việt</span><span>◆ tiếng Anh</span>${payload.tenant ? "<span>mờ = ngoài tenant/không còn hiệu lực</span>" : ""}` +
      topics.map((t) => `<span><i class="dot" style="background:${topicColor(t)}"></i>${esc(t)}</span>`).join("") +
      `<span>PCA 2D chỉ giữ ~20% phương sai: khoảng cách trên hình chỉ mang tính minh họa.</span>`;
  }

  return { update(payload) { state.payload = payload; controls(); draw(); } };
}
