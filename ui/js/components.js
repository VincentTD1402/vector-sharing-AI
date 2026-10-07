// Thành phần giao diện dùng chung giữa các bài học.
import { esc } from "./util.js";

export function stepHeader(step) {
  return `<header class="stephead"><span class="step">${step.num}</span>
    <div><h1>${esc(step.title)}</h1><p>${esc(step.tagline)}</p>
    ${step.slides ? `<span class="slides">Slide ${esc(step.slides)}</span>` : ""}</div></header>`;
}

export function hint(html) { return `<p class="hint">${html}</p>`; }

/** Chạy một thao tác bất đồng bộ trên nút (khóa nút + đổi nhãn); hiện lỗi vào errorBox nếu có, không thì alert. */
export async function withBusy(button, label, task, errorBox) {
  const original = button.textContent;
  button.disabled = true;
  button.textContent = label;
  if (errorBox) errorBox.hidden = true;
  try { return await task(); }
  catch (err) {
    if (errorBox) { errorBox.textContent = `Không chạy được: ${err.message}`; errorBox.hidden = false; }
    else alert(`Không chạy được: ${err.message}`);
  } finally { button.disabled = false; button.textContent = original; }
}

/** Biểu đồ cột cho 48 chiều đầu của vector (xanh = dương, cam = âm). */
export function dimsChart(dims, label = "48 chiều đầu của vector") {
  const max = Math.max(...dims.map(Math.abs), 0.001);
  const w = 800, h = 90, mid = h / 2, bw = w / dims.length;
  const bars = dims.map((v, i) => {
    const height = (Math.abs(v) / max) * (mid - 4);
    return `<rect x="${i * bw + 1}" y="${v >= 0 ? mid - height : mid}" width="${bw - 2}" height="${height}" fill="${v >= 0 ? "#0f766e" : "#b45309"}"><title>chiều ${i}: ${v}</title></rect>`;
  }).join("");
  return `<svg class="dims" viewBox="0 0 ${w} ${h}" preserveAspectRatio="none" role="img" aria-label="${esc(label)}">
    <line x1="0" x2="${w}" y1="${mid}" y2="${mid}" stroke="#c5ced6"/>${bars}</svg>`;
}

export const stat = (label, value, sub = "") =>
  `<div class="stat"><span class="v">${esc(value)}</span><span class="l">${esc(label)}</span>${sub ? `<span class="s">${esc(sub)}</span>` : ""}</div>`;
