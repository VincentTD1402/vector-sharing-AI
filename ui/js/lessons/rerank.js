// Bài 6 - Reranking: cross-encoder chấm lại ứng viên của retriever; khi nào cứu được, khi nào không.
import { post } from "../api.js";
import { bar, callout, kv, ms, rankPill, table } from "../lesson/render.js";
import { esc, fmt } from "../util.js";

const move = (c) => {
  const d = c.before_rank - c.after_rank;
  return d > 0 ? `<span class="move up">▲${d}</span>` : d < 0 ? `<span class="move down">▼${-d}</span>` : "<span class='hint'>=</span>";
};

function steps(d, v) {
  const rel = new Set(d.relevant ?? []);
  const maxAfter = Math.max(...d.candidates.map((c) => Math.abs(c.after_score)), 1);
  const byBefore = [...d.candidates].sort((a, b) => a.before_rank - b.before_rank);
  const relBefore = byBefore.find((c) => c.relevant)?.before_rank ?? null;
  return [
    { title: "Retriever lấy ứng viên", ms: ms(d.ms.retrieve),
      explain: `Retriever (${d.source === "hybrid" ? "BM25 + vector + RRF" : "vector"}) nhanh nhưng xấp xỉ: nó chọn <b>${d.pairs} ứng viên</b> để reranker xem xét. Bi-encoder encode query và doc <b>riêng rẽ</b> rồi so khoảng cách, nên bỏ qua tương tác giữa từng từ của hai bên.`,
      summary: `${d.pairs} ứng viên` + (d.relevant ? ` · doc đúng ${relBefore ? "hạng #" + relBefore : "KHÔNG có trong pool"}` : ""),
      html: table([{ t: "Hạng", num: true }, { t: "Doc" }, { t: "Điểm retriever", num: true }, { t: "Nội dung" }],
        byBefore.slice(0, 8).map((c) => [c.before_rank, `<span class="badge">${esc(c.id)}</span>`, fmt(c.before_score, 3), esc(c.content.slice(0, 80)) + "…"]),
        (i) => (byBefore[i].relevant ? "rel" : "")) + (d.pairs > 8 ? `<p class="hint">… và ${d.pairs - 8} ứng viên nữa</p>` : "") +
        (d.relevant && relBefore == null ? callout("bad", `Doc đúng <b>không nằm trong ${d.pairs} ứng viên</b>: reranker chỉ xếp lại những gì retriever đã đưa cho nó, nên không thể cứu.`) : "") },
    { title: "Cross-encoder chấm từng cặp", ms: ms(d.ms.rerank),
      explain: `Cross-encoder đọc <b>query và doc CÙNG LÚC</b> (cặp [query, doc] → 1 điểm), nên chính xác hơn nhưng phải chạy model <b>cho từng cặp</b>: ${d.pairs} cặp × ~${fmt(d.ms.per_pair, 1)} ms. Không precompute được như vector.`,
      summary: `${d.pairs} cặp × ${fmt(d.ms.per_pair, 1)} ms`,
      html: `<div class="pairs">${byBefore.map((c, i) => `<div class="pairrow" style="animation-delay:${i * 45}ms"><span class="badge">${esc(c.id)}</span>
          <div class="pairbar">${bar(c.after_score, maxAfter, c.after_score < 0 ? "neg" : "")}</div><span class="score">${fmt(c.after_score, 2)}</span></div>`).join("")}</div>
        <p class="hint">Điểm cross-encoder không cùng thang với cosine: chỉ thứ hạng có ý nghĩa. Điểm âm = "không liên quan" theo model.</p>` },
    { title: "Sắp xếp lại", ms: null,
      explain: "Sắp theo điểm cross-encoder giảm dần. ▲▼ cho biết mỗi doc nhảy lên/xuống bao nhiêu bậc so với thứ tự của retriever.",
      summary: d.candidates[0] ? `top-1: ${d.candidates[0].id}` : "—",
      html: table([{ t: "Hạng sau", num: true }, { t: "Đổi" }, { t: "Doc" }, { t: "Hạng trước", num: true }, { t: "Điểm", num: true }, { t: "Nội dung" }],
        d.candidates.slice(0, 8).map((c) => [`<b>${c.after_rank}</b>`, move(c), `<span class="badge">${esc(c.id)}</span>`, c.before_rank, fmt(c.after_score, 2), esc(c.content.slice(0, 70)) + "…"]),
        (i) => (d.candidates[i].relevant ? "rel" : "")) },
    { title: "Trước và sau, cái giá phải trả", ms: null,
      explain: "Đo hạng của doc đúng trước/sau, và chi phí thêm vào mỗi request. Nguyên tắc: <b>tối ưu retriever cho recall, tối ưu reranker cho precision</b>.",
      summary: d.metrics?.before ? `${d.metrics.before.first_hit ?? "–"} → ${d.metrics.after.first_hit ?? "–"}` : "không có nhãn",
      html: kv([["Hạng doc đúng (trước → sau)", d.relevant ? `${rankPill(d.metrics.before.first_hit)} → ${rankPill(d.metrics.after.first_hit)}` : "<i>query không có nhãn</i>"],
        ["MRR (trước → sau)", d.relevant ? `${fmt(d.metrics.before.mrr)} → <b>${fmt(d.metrics.after.mrr)}</b>` : "—"],
        ["Latency retrieve", ms(d.ms.retrieve)], ["Latency rerank", `${ms(d.ms.rerank)} (<b>${fmt(d.ms.rerank / Math.max(d.ms.retrieve, 1), 1)}×</b> retrieve)`]]) },
  ];
}

function diagnose(d) {
  if (!d.relevant) return { level: "info", title: "Query này chưa có nhãn đáp án", html: "Chọn một tình huống mẫu để thấy hạng của doc đúng trước/sau rerank." };
  const b = d.metrics.before.first_hit, a = d.metrics.after.first_hit;
  if (b == null) return { level: "bad", title: "Reranker không cứu được: doc đúng không nằm trong danh sách ứng viên",
    html: `Với pool ${d.pairs}, retriever không đưa doc đúng vào danh sách nên cross-encoder không có gì để nâng lên. <b>Reranker chỉ sắp xếp lại ứng viên có sẵn</b>; recall của retriever mới là giới hạn trên. Hãy tăng pool.` };
  if (a < b) return { level: "good", title: `Rerank đưa doc đúng từ hạng #${b} lên #${a}`,
    html: `Retriever để doc đúng ở hạng ${b}${b > 3 ? " (ngoài top-3 mà LLM thường nhận)" : ""}; cross-encoder đọc cặp query-doc nên nhận ra nó và đưa lên #${a}. Cái giá: ${ms(d.ms.rerank)} (~${fmt(d.ms.rerank / Math.max(d.ms.retrieve, 1), 0)}× thời gian retrieve).` };
  if (a > b) return { level: "bad", title: `Rerank làm tệ đi: doc đúng từ hạng #${b} xuống #${a}`,
    html: "Reranker không phải phép màu: nó là một model khác với điểm mù riêng (ở đây ưu tiên doc nói về mạng/VPN chung chung hơn doc đúng). Vì vậy luôn <b>đo trên bộ eval</b> (bài 7) trước khi bật rerank cho mọi query." };
  return { level: "info", title: `Thứ hạng doc đúng không đổi (#${a}), nhưng vẫn tốn ${ms(d.ms.rerank)}`,
    html: `Reranker đồng ý với retriever nên query này không được lợi gì. Chi phí vẫn phải trả: <b>${d.pairs} cặp × ${fmt(d.ms.per_pair, 1)} ms = ${ms(d.ms.rerank)}</b> (${fmt(d.ms.rerank / Math.max(d.ms.retrieve, 1), 1)}× thời gian retrieve) và <b>tăng tuyến tính theo pool</b>: thử giảm pool xuống 10 để thấy.` };
}

export default {
  id: "rerank", num: 6, group: "B", title: "Reranking", slides: "20",
  tagline: "Retriever nhanh nhưng xấp xỉ; reranker chậm nhưng chính xác. Khi nào nó cứu được, khi nào không.",
  theory: {
    what: `<p><b>Bi-encoder</b> (embedding): encode query và doc riêng, so khoảng cách. Nhanh, doc vector precompute được.</p>
      <p><b>Cross-encoder</b> (reranker): đọc [query + doc] cùng lúc, cho điểm từng cặp. Chính xác hơn, nhưng phải chạy lại cho mỗi cặp.</p>`,
    why: `<p>Vector search ưu tiên tốc độ và recall, không phải precision: doc đúng có thể nằm hạng 6 trong khi LLM chỉ nhận top-3.</p>
      <p>Pipeline chuẩn: retriever lấy top-50 → cross-encoder chấm lại → top-5 cho LLM.</p>`,
    points: ["Chi phí rerank <b>tăng tuyến tính theo số ứng viên</b>, nên chỉ rerank top-N nhỏ.",
      "Reranker <b>không thể cứu</b> doc mà retriever đã bỏ lỡ (ngoài pool).",
      "Reranker cũng có thể làm tệ đi: luôn đo."],
    visual: `<svg viewBox="0 0 760 120" class="theorysvg" role="img" aria-label="Sơ đồ rerank">
      <rect x="10" y="30" width="150" height="56" rx="8" fill="#dcfce7" stroke="#16a34a"/><text x="85" y="55" text-anchor="middle" font-size="13" font-weight="700">Retriever</text><text x="85" y="73" text-anchor="middle" font-size="11">top-50 · nhanh · recall</text>
      <text x="180" y="64" font-size="22">→</text>
      <rect x="205" y="30" width="170" height="56" rx="8" fill="#fef3c7" stroke="#d97706"/><text x="290" y="55" text-anchor="middle" font-size="13" font-weight="700">Cross-encoder</text><text x="290" y="73" text-anchor="middle" font-size="11">50 cặp · chậm · precision</text>
      <text x="395" y="64" font-size="22">→</text>
      <rect x="420" y="30" width="110" height="56" rx="8" fill="#ede9fe" stroke="#7c3aed"/><text x="475" y="55" text-anchor="middle" font-size="13" font-weight="700">top-5</text><text x="475" y="73" text-anchor="middle" font-size="11">gửi cho LLM</text>
      <text x="560" y="50" font-size="12" fill="#5d6b79">Tối ưu retriever cho RECALL,</text><text x="560" y="68" font-size="12" fill="#5d6b79">reranker cho PRECISION.</text></svg>`,
  },
  fields: [
    { name: "query", label: "Query", type: "text", wide: true },
    { name: "source", label: "Ứng viên lấy từ", type: "select", options: [{ value: "vector", label: "Vector" }, { value: "hybrid", label: "Hybrid (BM25 + vector)" }] },
    { name: "pool", label: "Số ứng viên (pool)", type: "range", min: 3, max: 30 },
  ],
  cases: [
    { id: "rescue", tag: "problem", title: "Doc đúng ở hạng 6, LLM chỉ nhận top-3", slide: "20",
      story: "Vector search ưu tiên tốc độ và recall, không phải precision. Reranker giải quyết vấn đề precision.",
      expect: "Retriever xếp ERR-4102 hạng #6 giữa các mã lỗi na ná; cross-encoder đưa nó lên #1.",
      values: { query: "ERR-4102", source: "vector", pool: 20 } },
    { id: "worse", tag: "error", title: "Reranker cũng có thể làm tệ đi", slide: "20, 25",
      story: "Mọi quyết định đều là đánh đổi; không có cấu hình tốt nhất tuyệt đối. Reranker thêm latency và không phải lúc nào cũng tốt hơn.",
      expect: "Retriever xếp doc VPN đúng hạng #1, nhưng cross-encoder đẩy xuống #3.",
      values: { query: "connect to the company network from home", source: "vector", pool: 20 },
      fixes: [{ label: "Thử lấy ứng viên từ Hybrid", patch: { source: "hybrid" } }] },
    { id: "small-pool", tag: "problem", title: "Pool quá nhỏ: reranker không cứu được", slide: "20",
      story: "Retriever lấy Top-50 ứng viên rồi cross-encoder mới chấm lại; recall của retriever là giới hạn trên.",
      expect: "Pool = 5 thì ERR-4102 (hạng 6) không có trong danh sách → reranker bất lực.",
      values: { query: "ERR-4102", source: "vector", pool: 5 },
      fixes: [{ label: "Tăng pool lên 20", patch: { pool: 20 } }] },
    { id: "cost", tag: "observe", title: "Chi phí tăng tuyến tính theo pool", slide: "20, 25",
      story: "Cross-encoder: chậm (per pair), chỉ dùng cho tập nhỏ. Reranker: precision tốt hơn nhưng latency cao hơn, tốn chi phí.",
      expect: "Pool 30 → rerank ~2× pool 15. So với retrieve vài chục ms, rerank chiếm phần lớn latency.",
      values: { query: "How to approve an invoice", source: "hybrid", pool: 30 },
      fixes: [{ label: "Giảm pool xuống 10", patch: { pool: 10 } }] },
    { id: "free", tag: "free", title: "Tự thử", slide: "20", story: "Tự nhập query, chọn nguồn ứng viên và pool.", expect: "Xem hạng thay đổi và chi phí.",
      values: { query: "how many vacation days do I get", source: "vector", pool: 20 } },
  ],
  async run(v) {
    const d = await post("/api/rerank", { query: v.query, tenant_id: "company_a", source: v.source, pool: v.pool });
    if (!d.candidates.length) throw new Error("Không có ứng viên nào cho query này.");
    const m = d.metrics;
    return {
      steps: steps(d, v), diagnosis: diagnose(d),
      lessons: ["Reranker nâng precision nhưng <b>thêm latency</b> (tỉ lệ thuận số cặp); chọn pool vừa đủ để recall không bị cắt.",
        "Recall của retriever là <b>giới hạn trên</b>: doc ngoài pool thì không rerank được.",
        "Điểm cross-encoder và điểm retriever khác thang; chỉ so thứ hạng."],
      summary: { label: `${v.source} · pool ${v.pool}`, facts: {
        "Hạng đúng trước": m.before ? (m.before.first_hit ?? { v: "không có", level: "bad" }) : "—",
        "Hạng đúng sau": m.after ? (m.after.first_hit === 1 ? { v: 1, level: "good" } : (m.after.first_hit ?? { v: "không có", level: "bad" })) : "—",
        "Rerank (ms)": d.ms.rerank, "Cặp chấm": d.pairs } },
    };
  },
};
