// Thí nghiệm TÙY BIẾN của bài 3 trên pgvector thật: dữ liệu đang lưu, ANN 50.000 vector (streaming), filter chọn lọc.
import { get, post, streamNdjson } from "../api.js";
import { stat, withBusy } from "../components.js";
import { callout, table } from "../lesson/render.js";
import { $, esc, fmt } from "../util.js";

const COLORS = { exact: "#475569", ivfflat: "#d97706", hnsw: "#0f766e" };
const LABELS = { exact: "Exact (seq scan)", ivfflat: "IVFFlat (lists=100)", hnsw: "HNSW (m=16, ef_c=64)" };

/** Scatter recall (%) theo latency (ms, thang log). */
function scatter(series) {
  const W = 640, H = 300, L = 46, B = 34, T = 14, R = 14;
  const pts = Object.entries(series).flatMap(([kind, rows]) => rows.map((r) => ({ ...r, kind })));
  if (!pts.length) return "";
  const lo = Math.log10(Math.min(...pts.map((p) => p.ms), 1)), hi = Math.log10(Math.max(...pts.map((p) => p.ms), 10) * 1.2);
  const x = (v) => L + ((Math.log10(v) - lo) / (hi - lo)) * (W - L - R), y = (rc) => T + (1 - rc / 100) * (H - T - B);
  const grid = [0, 25, 50, 75, 100].map((v) => `<line x1="${L}" x2="${W - R}" y1="${y(v)}" y2="${y(v)}" stroke="#e4e9ee"/><text x="${L - 6}" y="${y(v) + 4}" font-size="11" text-anchor="end">${v}%</text>`).join("");
  const ticks = [1, 10, 100].filter((t) => Math.log10(t) >= lo && Math.log10(t) <= hi).map((t) => `<text x="${x(t)}" y="${H - 12}" font-size="11" text-anchor="middle">${t} ms</text>`).join("");
  const lines = Object.entries(series).map(([k, rows]) => rows.length > 1 ? `<polyline fill="none" stroke="${COLORS[k]}" stroke-width="1.5" points="${rows.map((r) => `${x(r.ms)},${y(r.recall)}`).join(" ")}"/>` : "").join("");
  const dots = pts.map((p) => `<circle cx="${x(p.ms)}" cy="${y(p.recall)}" r="5" fill="${COLORS[p.kind]}"><title>${esc(LABELS[p.kind])} ${esc(p.param)}: recall ${p.recall}%, ${p.ms} ms</title></circle>`).join("");
  return `<svg viewBox="0 0 ${W} ${H}" class="chart" role="img" aria-label="Recall theo latency">${grid}${ticks}${lines}${dots}
    <text x="${W / 2}" y="${H - 1}" font-size="11" text-anchor="middle" fill="#5d6b79">latency / query (log) →</text></svg>
    <div class="legend">${Object.keys(COLORS).map((k) => `<span><i class="dot" style="background:${COLORS[k]}"></i>${LABELS[k]}</span>`).join("")}</div>`;
}

/** Thí nghiệm A: bảng/chỉ mục/dòng mẫu đang lưu trong pgvector. */
export function mountStoredData(el) {
  el.innerHTML = `<div class="row"><button class="primary" id="load">Đọc dữ liệu từ pgvector</button></div><div id="out" class="emptystate">Chưa tải. Bấm nút để xem bảng <code>document_chunks</code>, các chỉ mục và vài dòng mẫu.</div>`;
  $("#load", el).onclick = () => withBusy($("#load", el), "Đang đọc…", async () => {
    const info = await get("/api/index/info");
    $("#out", el).className = "";
    $("#out", el).innerHTML = `<div class="stats">${stat("chunk", info.rows)}${stat("kích thước bảng + chỉ mục", `${info.total_mb} MB`)}${stat("số chiều vector", info.dim)}</div>
      <div class="chips">${info.groups.map((g) => `<span class="badge">${esc(g.tenant_id)} · ${esc(g.status)} · ${esc(g.language)}: <b>${g.count}</b></span>`).join(" ")}</div>
      <h3>Chỉ mục</h3>${table([{ t: "Tên" }, { t: "Loại" }, { t: "Kích thước", num: true }, { t: "Định nghĩa" }],
        info.indexes.map((i) => [esc(i.name), `<span class="badge ${i.type === "hnsw" ? "good" : ""}">${esc(i.type)}</span>`, `${i.mb} MB`, `<code>${esc(i.ddl)}</code>`]))}
      <h3>Vài dòng mẫu</h3>${table([{ t: "id" }, { t: "tenant" }, { t: "lang" }, { t: "status" }, { t: "content" }, { t: "embedding (6 chiều đầu)" }],
        info.sample.map((s) => [esc(s.id), esc(s.tenant_id), esc(s.language), esc(s.status), esc(s.content) + "…", `<code>[${s.embedding.join(", ")}, …]</code>`]))}
      <p class="hint">Mỗi chunk = nội dung + vector + <b>metadata</b> (tenant, language, status, department). Metadata phải lưu ngay lúc index vì dùng để filter/phân quyền lúc query (bài 5).</p>`;
  }, null);
}

/** Thí nghiệm B: exact vs IVFFlat vs HNSW trên 50.000 vector (streaming). */
export function mountExplorer(el) {
  el.innerHTML = `<p class="hint">Bảng <code>ann_bench</code>: <b>50.000 vector</b> tổng hợp có cấu trúc (30 query, k=10); recall tính so với exact search. Lần đầu nạp ~20 giây, build HNSW ~30 giây.</p>
    <div class="row"><button data-kind="exact">Exact (quét toàn bảng)</button><button data-kind="ivfflat">IVFFlat</button><button data-kind="hnsw">HNSW</button>
      <button id="all" class="primary">Chạy cả ba</button><span id="st" class="hint"></span></div>
    <div class="two"><div id="chart"><div class="emptystate">Chưa chạy. Bấm một nút để đo recall và latency.</div></div><div id="rows" class="tablewrap"></div></div><div id="builds"></div>`;
  const series = { exact: [], ivfflat: [], hnsw: [] }, builds = {};
  const draw = () => {
    $("#chart", el).innerHTML = scatter(series);
    const rows = Object.entries(series).flatMap(([k, rs]) => rs.map((r) => [`<i class="dot" style="background:${COLORS[k]}"></i>${esc(LABELS[k])}`, esc(r.param), `${r.recall}%`, `${r.ms} ms`]));
    $("#rows", el).innerHTML = rows.length ? table([{ t: "Chỉ mục" }, { t: "Tham số" }, { t: "Recall@10", num: true }, { t: "Latency", num: true }], rows) : "";
    $("#builds", el).innerHTML = Object.entries(builds).map(([k, b]) => `<p class="hint"><b>${esc(LABELS[k])}</b>: build ${b.seconds}s, chỉ mục ${b.mb} MB - <code>${esc(b.ddl)}</code></p>`).join("");
  };
  const run = async (kind) => {
    series[kind] = [];
    await streamNdjson(`/api/ann/stream?kind=${kind}`, (e) => {
      if (e.error) throw new Error(e.error);
      if (e.status) $("#st", el).textContent = e.status;
      if (e.build) builds[e.build.kind] = e.build;
      if (e.row) series[kind].push(e.row);
      if (e.done) $("#st", el).textContent = `Xong ${LABELS[kind]}.`;
      draw();
    });
  };
  el.querySelectorAll("button[data-kind]").forEach((b) => { b.onclick = () => withBusy(b, "Đang chạy…", () => run(b.dataset.kind), null); });
  $("#all", el).onclick = () => withBusy($("#all", el), "Đang chạy cả ba…", async () => { for (const k of ["exact", "ivfflat", "hnsw"]) await run(k); }, null);
}

/** Thí nghiệm C: HNSW + WHERE chọn lọc trả thiếu kết quả. */
export function mountFilterLab(el) {
  el.innerHTML = `<p class="hint">Tenant chỉ chiếm 1% dữ liệu. HNSW chỉ xét ~<code>ef_search</code> ứng viên TOÀN BẢNG rồi mới lọc theo tenant nên có thể trả thiếu mà không báo lỗi.</p>
    <div class="row"><button class="primary" id="go2">Chạy thí nghiệm (build HNSW ~30s)</button></div><div id="out2"></div>`;
  $("#go2", el).onclick = () => withBusy($("#go2", el), "Đang chạy…", async () => {
    const r = await post("/api/ann/filter");
    if (r.error) throw new Error(r.error);
    $("#out2", el).innerHTML = table([{ t: "Cách làm" }, { t: `Hàng trả về (k=${r.k})`, num: true }, { t: "Recall", num: true }],
      r.rows.map((x) => [esc(x.label), `${fmt(x.returned, 1)}/${r.k}`, `${x.recall}%`]), (i) => (r.rows[i].returned < r.k ? "bad-row" : "rel")) +
      callout("warn", `Tenant ${r.tenant} có ${r.matching_rows} / 50.000 hàng. HNSW mặc định trả <b>${fmt(r.rows[0].returned, 1)}/${r.k}</b> hàng mà <b>không báo lỗi</b>. Cách xử lý: iterative scan (pgvector ≥ 0.8), exact search trên tập đã lọc nhỏ, hoặc tách partition / chỉ mục riêng theo tenant.`);
  }, null);
}
