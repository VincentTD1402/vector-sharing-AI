// Thí nghiệm tùy biến của bài 7: chạy TOÀN BỘ bộ eval (streaming) với tiến độ, bảng từng query và tổng hợp.
import { streamNdjson } from "../api.js";
import { hint, withBusy } from "../components.js";
import { callout } from "../lesson/render.js";
import { $, esc, fmt } from "../util.js";

const KINDS = ["exact", "natural", "paraphrase"];
const hitCell = (hit) => `<span class="hit ${hit === 1 ? "h1" : hit ? "h2" : "h0"}">${hit ?? "–"}</span>`;

function summaryTable(configs, summary, k) {
  const bar = (value) => `<div class="bar" style="width:${value * 100}%"></div><span class="val">${fmt(value, 3)}</span>`;
  const rows = configs.map((c) => {
    const s = summary[c.key];
    return `<tr><td><b>${esc(c.label)}</b></td><td class="metric-cell">${bar(s.recall)}</td><td class="metric-cell">${bar(s.mrr)}</td>
      <td class="num">${fmt(s.precision, 3)}</td><td class="num">${fmt(s.ms, 0)} ms</td>
      ${KINDS.map((kind) => `<td class="num">${s.recall_by_kind[kind] == null ? "–" : fmt(s.recall_by_kind[kind])}</td>`).join("")}</tr>`;
  }).join("");
  return `<div class="tablewrap"><table><thead><tr><th>Cấu hình</th><th>Recall@${k}</th><th>MRR</th><th class="num">Precision@${k}</th>
    <th class="num">Latency TB</th>${KINDS.map((x) => `<th class="num">Recall@${k} · ${x}</th>`).join("")}</tr></thead><tbody>${rows}</tbody></table></div>`;
}

function liveSummary(configs, rows) {
  const mean = (key, metric, subset = rows) => subset.reduce((s, r) => s + r.configs[key][metric], 0) / subset.length;
  return Object.fromEntries(configs.map((c) => [c.key, {
    recall: mean(c.key, "recall"), mrr: mean(c.key, "mrr"), precision: mean(c.key, "precision"), ms: mean(c.key, "ms"),
    recall_by_kind: Object.fromEntries(KINDS.map((kind) => {
      const subset = rows.filter((r) => r.kind === kind);
      return [kind, subset.length ? mean(c.key, "recall", subset) : null];
    })),
  }]));
}

export function mountEvalRunner(el, ctx) {
  const { configs, metric_k: k } = ctx.meta;
  el.innerHTML = `<div class="row"><button id="run" class="primary">Chạy toàn bộ bộ eval (36 query × 5 cấu hình)</button>
      <span id="st" class="hint">Mất khoảng 30–60 giây trên CPU. Số liệu cập nhật trực tiếp khi mỗi query chạy xong.</span></div>
    <div class="progress" id="prog" hidden><div id="bar"></div></div>
    <div id="sum"></div><div id="rows" class="tablewrap"></div><div id="note"></div>`;
  $("#run", el).onclick = () => withBusy($("#run", el), "Đang chạy…", async () => {
    const rows = [];
    $("#prog", el).hidden = false; $("#bar", el).style.width = "0"; $("#note", el).innerHTML = "";
    $("#rows", el).innerHTML = `<table><thead><tr><th>#</th><th>Query</th><th>Loại</th>${configs.map((c) => `<th>${esc(c.label)}</th>`).join("")}</tr></thead><tbody id="body"></tbody></table>`;
    await streamNdjson("/api/eval/stream", (row) => {
      if (row.done) { $("#sum", el).innerHTML = summaryTable(configs, row.summary, k); return; }
      rows.push(row);
      $("#bar", el).style.width = `${(row.i / row.n) * 100}%`;
      $("#st", el).textContent = `Đang chạy… ${row.i}/${row.n} query`;
      $("#body", el).insertAdjacentHTML("beforeend", `<tr><td class="num">${row.i}</td><td>${esc(row.query)}</td><td>${esc(row.kind)}</td>${configs.map((c) => `<td>${hitCell(row.configs[c.key].first_hit)}</td>`).join("")}</tr>`);
      $("#sum", el).innerHTML = summaryTable(configs, liveSummary(configs, rows), k);
      $("#rows", el).scrollTop = $("#rows", el).scrollHeight;
    });
    $("#st", el).textContent = `Xong ${rows.length} query. Số trong bảng từng query = hạng của doc đúng đầu tiên (– = không có).`;
    $("#note", el).innerHTML = callout("info", `<b>Đọc kết quả:</b> rerank là bước cải thiện lớn nhất; hybrid chỉ nhỉnh hơn vector (trong ngưỡng nhiễu của 36 query) nhưng sửa được các mã chính xác. Bộ eval nhỏ nên chênh lệch cỡ 0.03 là nhiễu: chỉ đọc xu hướng. Bấm tình huống "Debug" để xem vì sao một query cụ thể thất bại.`);
  }, null);
}
