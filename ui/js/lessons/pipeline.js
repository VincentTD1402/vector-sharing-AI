// Bài 8 - Pipeline hoàn chỉnh: một request đi qua mọi bước, ngân sách latency, log có cấu trúc, và semantic cache.
import { post } from "../api.js";
import { dimsChart } from "../components.js";
import { callout, docLine, kv, ms, pill, rankPill, sqlBlock, table } from "../lesson/render.js";
import { esc, fmt } from "../util.js";

const TRACE_IDS = ["trace", "trace-short", "trace-latency", "free"];
const CACHE_IDS = ["cache", "cache-loose"];
const COLORS = { embed: "#2563eb", dense: "#059669", lexical: "#d97706", fusion: "#7c3aed", rerank: "#dc2626" };
const LABEL = { embed: "Embedding", dense: "Dense (pgvector)", lexical: "BM25", fusion: "RRF", rerank: "Rerank" };

function budget(d) {
  const m = d.stages_ms;
  const parts = [["embed", m.embed], ["dense", m.dense], ["lexical", m.lexical], ["fusion", m.fusion], ["rerank", m.hybrid_rerank]];
  const total = parts.reduce((s, p) => s + p[1], 0) || 1;
  return { parts, total, share: Object.fromEntries(parts.map(([k, v]) => [k, v / total])) };
}

function traceSteps(d, v) {
  const m = d.stages_ms, b = budget(d);
  const cfg = Object.fromEntries(d.configs.map((c) => [c.key, c]));
  const log = { request_id: "demo-0001", query: d.query, filters: { tenant_id: d.filters.tenant_id, filter_mode: d.filters.filter_mode, transform: v.transform },
    timings_ms: { embed: m.embed, dense_search: m.dense, lexical_search: m.lexical, fusion: m.fusion, rerank: m.hybrid_rerank, total: Math.round(b.total * 10) / 10 },
    counts: d.counts, final: cfg.hybrid_rerank.results.map((r) => ({ id: r.id, score: r.score })) };
  return [
    { title: "Nhận request", ms: null, summary: `tenant ${d.filters.tenant_id} · ${d.filters.filter_mode}-filter`,
      explain: "Request đến API: query và <b>ai đang hỏi</b>. Ở production tenant lấy từ <b>token xác thực</b>, không nhận từ client; top_k bị giới hạn; mọi request có request_id để truy vết.",
      html: kv([["Query", `<b>${esc(d.query)}</b>`], ["Tenant", esc(d.filters.tenant_id)], ["Cách lọc", pill(d.filters.filter_mode, { pre: "good", post: "warn", off: "bad" }[d.filters.filter_mode])], ["Top-K", v.top_k]])
        + (d.filters.filter_mode !== "pre" ? callout("bad", "Chế độ lọc này chỉ để minh họa lỗi (bài 5): production luôn dùng pre-filter.") : "") },
    { title: "Query transformation", ms: null, summary: d.variants.length > 1 ? `${d.variants.length} biến thể` : d.variants[0] !== d.query ? "đã viết lại" : "giữ nguyên",
      explain: "Tùy chọn: viết lại/mở rộng query trước khi embed (bài 4). Mỗi biến thể tốn thêm một lần embed.",
      html: `<div class="tokens">${d.variants.map((x) => `<span class="token ${x !== d.query ? "good" : ""}">${esc(x)}</span>`).join("")}</div>` },
    { title: "Embedding query", ms: ms(m.embed), summary: `${d.embedding.dim} chiều`,
      explain: `Mỗi biến thể → vector ${d.embedding.dim} chiều (bài 2). Có thể cache embedding của query lặp lại để bỏ bước này.`,
      html: `<div class="tokens">${d.embedding.tokens.map((t) => `<span class="token">${esc(t)}</span>`).join("")}</div>${dimsChart(d.embedding.dims)}` },
    { title: "Retrieval song song: vector ∥ BM25", ms: ms(m.dense + m.lexical), summary: `${d.counts.dense} dense · ${d.counts.lexical} BM25`,
      explain: `Hai nhánh chạy với <b>cùng filter</b> (bài 5). Vector (pgvector): ${ms(m.dense)}, ${d.counts.dense} ứng viên. BM25: ${ms(m.lexical)}, ${d.counts.lexical} doc khớp từ.`,
      html: `<div class="two-col"><div><h3>Vector</h3>${cfg.vector.results.map((r) => docLine(r, { score: r.score, mark: r.relevant ?? undefined })).join("")}</div>
        <div><h3>BM25</h3>${cfg.keyword.results.map((r) => docLine(r, { score: r.score, mark: r.relevant ?? undefined })).join("") || "<p class='hint'>Không doc nào khớp từ.</p>"}</div></div>` },
    { title: "Hợp nhất RRF", ms: ms(m.fusion), summary: `${d.counts.fused} ứng viên`,
      explain: "Gộp hai danh sách theo thứ hạng: Σ 1/(60 + hạng). Rẻ (dưới 1 ms) nhưng quyết định pool đưa cho reranker.",
      html: table([{ t: "Doc" }, { t: "Hạng BM25", num: true }, { t: "Hạng vector", num: true }, { t: "RRF", num: true }],
        d.rrf.slice(0, 5).map((r) => [esc(r.id), r.rank_lexical ?? "–", r.rank_dense ?? "–", fmt(r.score, 4)]), (i) => ((d.relevant || []).includes(d.rrf[i].id) ? "rel" : "")) },
    { title: "Rerank bằng cross-encoder", ms: ms(m.hybrid_rerank), summary: `${d.counts.fused} cặp`,
      explain: `Chấm lại ${d.counts.fused} cặp (query gốc, doc) → đây thường là <b>bước đắt nhất</b> (${fmt(b.share.rerank * 100, 0)}% tổng thời gian). Cột ▲▼ cho biết doc nhảy hạng so với trước rerank.`,
      html: cfg.hybrid_rerank.results.map((r) => docLine(r, { score: r.score, mark: r.relevant ?? undefined }).replace("</div>", `${r.base_rank != null && r.base_rank !== r.rank ? `<span class="move ${r.base_rank > r.rank ? "up" : "down"}">${r.base_rank > r.rank ? "▲" : "▼"}${Math.abs(r.base_rank - r.rank)}</span>` : ""}</div>`)).join("") },
    { title: "Kết quả cuối và so sánh", ms: null, summary: d.relevant ? `hybrid+rerank: doc đúng #${cfg.hybrid_rerank.metrics?.first_hit ?? "–"}` : "top-K cho LLM",
      explain: "Top-K sau cùng được đưa cho LLM làm ngữ cảnh. Bảng so sánh cho thấy từng giai đoạn đã đóng góp gì (khi query có nhãn).",
      html: table([{ t: "Cấu hình" }, { t: "Hạng doc đúng" }, { t: "Latency", num: true }],
        d.configs.map((c) => [esc(c.label), d.relevant ? rankPill(c.metrics?.first_hit) : "<i>không có nhãn</i>", ms(c.latency_ms)])) },
    { title: "Log có cấu trúc và ngân sách latency", ms: null, summary: `tổng ${ms(b.total)}`,
      explain: "Mỗi request ghi một dòng log JSON (query, filter, thời gian từng stage, id kết quả) để <b>debug được production</b> (slide 31). Thanh dưới là ngân sách latency: không đo từng stage thì không biết tối ưu chỗ nào.",
      html: `<div class="latbar">${b.parts.map(([k, val]) => `<span style="width:${(val / b.total) * 100}%;background:${COLORS[k]}">${val / b.total > 0.08 ? Math.round((val / b.total) * 100) + "%" : ""}</span>`).join("")}</div>
        <div class="latlegend">${b.parts.map(([k, val]) => `<span><i class="dot" style="background:${COLORS[k]}"></i>${LABEL[k]} ${ms(val)}</span>`).join("")}</div>
        <p class="hint">Dòng log của request này:</p>${sqlBlock(JSON.stringify(log, null, 2))}` },
  ];
}

function traceDiag(d, v) {
  const b = budget(d);
  const top = Object.entries(b.share).sort((x, y) => y[1] - x[1])[0];
  const fin = d.configs.find((c) => c.key === "hybrid_rerank");
  const q = d.relevant ? ` Doc đúng cuối cùng đứng hạng <b>#${fin.metrics?.first_hit ?? "–"}</b>.` : "";
  if (top[0] === "rerank" && top[1] > 0.6) return { level: "warn", title: `Rerank chiếm ${fmt(top[1] * 100, 0)}% latency (${ms(d.stages_ms.hybrid_rerank)} / ${ms(b.total)})`,
    html: `Đây là điểm nghẽn điển hình. Cách giảm: <b>giảm pool</b> (bài 6), model rerank nhỏ hơn, batch/GPU, hoặc <b>cache</b> kết quả cho query lặp lại (tình huống tiếp theo).${q}` };
  return { level: "good", title: `Pipeline chạy xong trong ${ms(b.total)}`, html: `Bước tốn nhất: ${LABEL[top[0]]} (${fmt(top[1] * 100, 0)}%).${q}` };
}

// ---------------------------------------------------------------- cache
function cacheSteps(d) {
  const hits = d.rows.filter((r) => r.outcome === "hit"), miss = d.rows.filter((r) => r.outcome === "miss");
  const avg = (rs) => (rs.length ? rs.reduce((s, r) => s + r.ms, 0) / rs.length : null);
  const wrong = hits.filter((r) => !r.correct);
  return [
    { title: "Cache rỗng, đặt ngưỡng", ms: null, summary: `ngưỡng cosine ≥ ${fmt(d.threshold, 2)}`,
      explain: `Semantic cache lưu (vector query, kết quả). Query mới được so cosine với các query đã lưu; nếu <b>≥ ${fmt(d.threshold, 2)}</b> thì trả lại kết quả đã lưu, không chạy pipeline. e5 nén điểm vào dải hẹp (~0.7–1.0) nên ngưỡng phải <b>tự hiệu chỉnh</b>.`,
      html: kv([["Ngưỡng", `<b>${fmt(d.threshold, 2)}</b>`], ["Khóa cache", "(vector query, tenant): luôn gồm tenant để không rò rỉ"], ["Số query sẽ chạy", d.rows.length]]) },
    { title: "Chạy tuần tự các query", ms: null, summary: `${hits.length} hit · ${miss.length} miss`,
      explain: "MISS = chạy pipeline đầy đủ rồi lưu vào cache. HIT = trả ngay kết quả đã lưu. Cột \"đúng?\" so kết quả cache với kết quả tìm thật (ngoài phép đo thời gian).",
      html: table([{ t: "Query" }, { t: "Kết quả" }, { t: "Thời gian", num: true }, { t: "Độ giống với cache", num: true }, { t: "Trả lời bằng cache của" }, { t: "Đúng?" }],
        d.rows.map((r) => [esc(r.query), r.outcome === "hit" ? pill("HIT", "good") : pill("MISS", "warn"), ms(r.ms), r.similarity != null ? fmt(r.similarity, 3) : "–",
          r.matched_query ? esc(r.matched_query) : "–", r.outcome === "hit" ? (r.correct ? "<b class='ok'>✓</b>" : "<b class='no'>✗ SAI</b>") : "–"]),
        (i) => (d.rows[i].outcome === "hit" ? (d.rows[i].correct ? "rel" : "bad-row") : "")) },
    { title: "Phân tích", ms: null, summary: `hit rate ${fmt((hits.length / d.rows.length) * 100, 0)}%`,
      explain: "Cache tiết kiệm bao nhiêu thời gian, và có trả lời sai không?",
      html: kv([["Hit rate", `<b>${hits.length}/${d.rows.length}</b>`], ["Thời gian TB khi MISS", avg(miss) != null ? ms(avg(miss)) : "–"], ["Thời gian TB khi HIT", avg(hits) != null ? ms(avg(hits)) : "–"],
        ["Hit nhưng trả lời SAI", wrong.length ? `<b class="no">${wrong.length}</b>` : "0"]]) },
  ];
}

function cacheDiag(d) {
  const wrong = d.rows.filter((r) => r.outcome === "hit" && !r.correct);
  const hits = d.rows.filter((r) => r.outcome === "hit");
  if (wrong.length) return { level: "bad", title: `Ngưỡng ${fmt(d.threshold, 2)} quá thấp: cache trả lời SAI cho ${wrong.length} query`,
    html: `Ví dụ «${esc(wrong[0].query)}» nhận kết quả đã cache của «${esc(wrong[0].matched_query)}» (độ giống ${fmt(wrong[0].similarity, 3)}) dù ý khác hẳn. Ngưỡng thấp = hit nhiều nhưng sai ý. Hãy tăng ngưỡng và đo trên dữ liệu thật.` };
  if (!hits.length) return { level: "warn", title: `Ngưỡng ${fmt(d.threshold, 2)} quá cao: không có HIT nào`, html: "Mỗi query đều chạy lại toàn bộ pipeline nên cache không giúp gì. Hạ ngưỡng từ từ và kiểm tra các hit có đúng ý không." };
  return { level: "good", title: `${hits.length}/${d.rows.length} query được phục vụ từ cache, đều đúng ý`, html: "Query lặp lại và gần như đồng nghĩa dùng lại kết quả; hit nhanh hơn miss cả bậc độ lớn. Nhớ đặt TTL ngắn khi dữ liệu thay đổi." };
}

const DEFAULT_CACHE_Q = ["how do I reset my password", "how do I reset my password", "How do I reset my password?", "how can I reset my password", "create production order", "how do I create a production order"].join("\n");
const LOOSE_Q = ["how do I reset my password", "create production order", "how do I approve an invoice", "how do I reset my password?", "I want to take a day off"].join("\n");

export default {
  id: "pipeline", num: 8, group: "C", title: "Pipeline hoàn chỉnh", slides: "28, 29, 30, 31, 32",
  tagline: "Mọi bước nối với nhau trên một request: ngân sách latency, log để debug, và cache để giảm chi phí.",
  theory: {
    what: `<p>Pipeline production: <b>Ingestion</b> (offline) → <b>Query transform</b> → <b>Hybrid retrieval + filter</b> → <b>RRF</b> → <b>Rerank</b> → <b>LLM</b>. Mỗi thành phần có thể thay độc lập để A/B test.</p>`,
    why: `<p>Chất lượng phụ thuộc <b>toàn stack</b>, không chỉ vector DB. Và production cần thêm <b>observability</b> (log từng bước) và <b>tối ưu</b> (cache, batch, async).</p>`,
    points: ["Log mỗi request: query, filter, điểm, thời gian từng stage: không có thì không debug được.",
      "Latency budget: rerank thường chiếm phần lớn; đo từng stage trước khi tối ưu.",
      "Semantic cache: tiết kiệm lớn nhưng ngưỡng sai sẽ trả lời sai; cache theo tenant."],
    visual: `<svg viewBox="0 0 760 90" class="theorysvg" role="img" aria-label="Pipeline production">
      ${["Query", "Transform", "Embed", "Dense ∥ BM25", "RRF", "Rerank", "Top-K → LLM"].map((t, i) => `<rect x="${6 + i * 108}" y="26" width="98" height="38" rx="7" fill="${i === 5 ? "#fde8e8" : "#e0ecff"}" stroke="${i === 5 ? "#b91c1c" : "#3b82f6"}"/><text x="${55 + i * 108}" y="50" text-anchor="middle" font-size="12">${t}</text>${i < 6 ? `<text x="${106 + i * 108}" y="50" font-size="14">→</text>` : ""}`).join("")}
      <text x="10" y="84" font-size="11" fill="#5d6b79">Mỗi mũi tên là một điểm có thể đo thời gian, ghi log và cache.</text></svg>`,
  },
  fields: [
    { name: "query", label: "Query", type: "text", showFor: TRACE_IDS, wide: true },
    { name: "tenant", label: "Tenant", type: "select", showFor: TRACE_IDS, options: ["company_a", "company_b", "company_c"].map((x) => ({ value: x, label: x })) },
    { name: "filter_mode", label: "Cách lọc", type: "select", showFor: TRACE_IDS, options: [{ value: "pre", label: "Pre-filter (đúng)" }, { value: "post", label: "Post-filter" }, { value: "off", label: "Tắt filter" }] },
    { name: "transform", label: "Query transformation", type: "select", showFor: TRACE_IDS, options: [{ value: "none", label: "Không biến đổi" }, { value: "rewrite", label: "Rewrite" }, { value: "expand", label: "Rewrite + Expansion" }, { value: "multi", label: "Multi-query" }] },
    { name: "top_k", label: "Top-K", type: "range", min: 1, max: 8, showFor: TRACE_IDS },
    { name: "queries", label: "Các query chạy tuần tự (mỗi dòng một query, tối đa 12)", type: "textarea", rows: 6, showFor: CACHE_IDS, wide: true },
    { name: "threshold", label: "Ngưỡng cosine của cache", type: "range", min: 0.7, max: 0.99, step: 0.01, showFor: CACHE_IDS },
  ],
  cases: [
    { id: "trace", tag: "observe", title: "Một request đi từ đầu đến cuối", slide: "28",
      story: "Pipeline production: ingestion → query processing → hybrid search + metadata pre-filter → RRF → reranker → LLM.",
      expect: "8 bước, mỗi bước có thời gian thật; ERR-4102: vector hạng 6 → hybrid 3 → rerank 1.",
      values: { query: "ERR-4102", tenant: "company_a", filter_mode: "pre", transform: "none", top_k: 3 } },
    { id: "trace-short", tag: "problem", title: "Query ngắn đi qua cả pipeline với multi-query", slide: "27, 28",
      story: "Mỗi component có thể được swap độc lập; query transformation là bước tùy chọn đầu pipeline.",
      expect: "\"pto request\" có 3 biến thể, 3 lần embed; doc đúng lên hạng 1. Tắt transform để thấy khác biệt.",
      values: { query: "pto request", tenant: "company_a", filter_mode: "pre", transform: "multi", top_k: 3 },
      fixes: [{ label: "Tắt query transformation", patch: { transform: "none" } }] },
    { id: "trace-latency", tag: "problem", title: "Điểm nghẽn latency: rerank", slide: "32",
      story: "Latency budget: embedding 20-50ms, vector 5-20ms, rerank 50-200ms; profiling từng bước là bắt buộc.",
      expect: "Thanh latency cho thấy rerank chiếm khoảng 80–90% tổng thời gian request.",
      values: { query: "How to approve an invoice", tenant: "company_a", filter_mode: "pre", transform: "none", top_k: 5 } },
    { id: "cache", tag: "fix", title: "Semantic cache: query lặp lại và tương tự", slide: "32",
      story: "Cache tránh gọi lại pipeline cho query phổ biến; semantic cache dùng similarity threshold.",
      expect: "Query giống hệt hoặc gần đồng nghĩa nhận HIT nhanh hơn MISS cả bậc độ lớn.",
      values: { queries: DEFAULT_CACHE_Q, threshold: 0.95 }, fixes: [{ label: "Hạ ngưỡng xuống 0.85", patch: { threshold: 0.85 } }, { label: "Nâng ngưỡng lên 0.99", patch: { threshold: 0.99 } }] },
    { id: "cache-loose", tag: "error", title: "Ngưỡng cache quá thấp → trả lời sai", slide: "32",
      story: "Ngưỡng quá thấp thì trả lời sai cho query khác ý; quá cao thì không bao giờ hit.",
      expect: "Với ngưỡng thấp, một query về chủ đề khác nhận lại kết quả đã cache của query trước đó.",
      values: { queries: LOOSE_Q, threshold: 0.76 }, fixes: [{ label: "Nâng ngưỡng lên 0.95", patch: { threshold: 0.95 } }] },
    { id: "free", tag: "free", title: "Tự thử", slide: "28", story: "Tự nhập query, tenant, cách lọc và biến đổi query.", expect: "Đọc từng bước và log của request.",
      values: { query: "I forgot my password", tenant: "company_a", filter_mode: "pre", transform: "none", top_k: 3 } },
  ],
  async run(v, ctx, current) {
    if (CACHE_IDS.includes(current.id)) {
      const d = await post("/api/lesson/cache", { queries: v.queries.split("\n").map((x) => x.trim()).filter(Boolean).slice(0, 12), threshold: v.threshold });
      const hits = d.rows.filter((r) => r.outcome === "hit"), wrong = hits.filter((r) => !r.correct);
      return { steps: cacheSteps(d), diagnosis: cacheDiag(d),
        lessons: ["Ngưỡng semantic cache là <b>đánh đổi</b> giữa tiết kiệm và đúng ý: hiệu chỉnh trên dữ liệu thật.", "Khóa cache luôn gồm <b>tenant</b> (và top_k, filter) để không rò rỉ hay trả sai ngữ cảnh.", "Đặt TTL ngắn khi dữ liệu thay đổi; ghi log hit/miss để theo dõi hit rate."],
        summary: { label: `ngưỡng ${fmt(v.threshold, 2)}`, facts: { "Hit": `${hits.length}/${d.rows.length}`, "Hit sai": { v: wrong.length, level: wrong.length ? "bad" : "good" } } } };
    }
    const d = await post("/api/compare", { query: v.query, tenant_id: v.tenant, top_k: v.top_k, filter_mode: v.filter_mode, transform: v.transform });
    const b = budget(d), fin = d.configs.find((c) => c.key === "hybrid_rerank");
    return { steps: traceSteps(d, v), diagnosis: traceDiag(d, v),
      lessons: ["Mỗi stage <b>đo được và thay được độc lập</b>: nhờ vậy A/B test từng phần (embedding, hybrid, reranker).", "Log có cấu trúc của mỗi request là thứ cứu bạn khi production trả kết quả lạ.", "Tối ưu theo số đo: thường là giảm chi phí rerank trước (pool, model, cache)."],
      summary: { label: `${v.transform} · ${v.filter_mode} · top-${v.top_k}`, facts: { "Tổng latency": ms(b.total), "Rerank chiếm": `${fmt(b.share.rerank * 100, 0)}%`, "Hạng đúng cuối": d.relevant ? (fin.metrics?.first_hit ?? { v: "không có", level: "bad" }) : "—" } } };
  },
};
