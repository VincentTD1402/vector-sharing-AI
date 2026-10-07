// Các mảnh HTML nhỏ dùng chung trong nội dung bài học (luôn escape dữ liệu động bằng esc()).
import { esc, fmt } from "../util.js";

/** Hộp nhấn mạnh. level: good | bad | warn | info. */
export const callout = (level, html) => `<div class="callout ${level}">${html}</div>`;

export const sqlBlock = (text) => `<pre class="sql">${esc(text)}</pre>`;
export const code = (text) => `<code>${esc(text)}</code>`;

export const pill = (text, level = "") => `<span class="badge ${level}">${esc(text)}</span>`;

/** Hạng của doc đúng: #1 xanh, #2-3 vàng, ngoài đó/không có -> đỏ. */
export function rankPill(rank) {
  if (rank == null) return `<span class="rankpill bad">không có</span>`;
  return `<span class="rankpill ${rank === 1 ? "good" : rank <= 3 ? "warn" : "bad"}">#${rank}</span>`;
}

export const bar = (value, max, cls = "") =>
  `<div class="bar ${cls}" style="width:${Math.max(2, (Math.abs(value) / (max || 1)) * 100)}%"></div>`;

/** Bảng nhỏ: headers = [{t, num?}], rows = mảng các mảng ô HTML (đã escape), rowClass(i) tuỳ chọn. */
export function table(headers, rows, rowClass = () => "") {
  return `<div class="tablewrap"><table><thead><tr>${headers.map((h) =>
    `<th class="${h.num ? "num" : ""}">${h.t}</th>`).join("")}</tr></thead><tbody>${rows.map((r, i) =>
    `<tr class="${rowClass(i)}">${r.map((c, j) => `<td class="${headers[j]?.num ? "num" : ""}">${c}</td>`).join("")}</tr>`).join("")}</tbody></table></div>`;
}

/** Danh sách "nhãn: giá trị" gọn. */
export const kv = (pairs) => `<dl class="kv">${pairs.map(([k, v]) => `<dt>${esc(k)}</dt><dd>${v}</dd>`).join("")}</dl>`;

/** Một tài liệu rút gọn (id + ngôn ngữ + nội dung), có thể kèm điểm và dấu đúng/sai. */
export function docLine(d, { score, mark } = {}) {
  const m = mark === true ? `<b class="ok">✓</b> ` : mark === false ? `<b class="no">✗</b> ` : "";
  return `<div class="docline">${m}<span class="badge">${esc(d.id)}</span><span class="badge">${esc(d.language ?? "")}</span>
    <span class="dtext">${esc((d.content ?? d.text ?? "").slice(0, 120))}</span>${score != null ? `<span class="score">${esc(fmt(score, 3))}</span>` : ""}</div>`;
}

export const ms = (v) => `${fmt(v, v < 10 ? 1 : 0)} ms`;
