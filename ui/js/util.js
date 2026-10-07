// Tiện ích dùng chung: chọn phần tử, escape HTML, màu theo chủ đề.
export const $ = (selector, root = document) => root.querySelector(selector);

const ESCAPES = { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" };
// Mọi nội dung đến từ dữ liệu đều đi qua esc() trước khi chèn vào innerHTML.
export const esc = (value) => String(value ?? "").replace(/[&<>"']/g, (c) => ESCAPES[c]);

const PALETTE = ["#2563eb", "#dc2626", "#059669", "#7c3aed", "#d97706", "#0891b2", "#db2777", "#65a30d", "#475569", "#9333ea", "#ea580c"];
const topicColors = new Map();
export function topicColor(topic) {
  if (!topicColors.has(topic)) topicColors.set(topic, PALETTE[topicColors.size % PALETTE.length]);
  return topicColors.get(topic);
}

export const fmt = (n, digits = 2) => (n == null ? "–" : Number(n).toFixed(digits));
