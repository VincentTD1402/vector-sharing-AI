// Bài 7 - Đánh giá: đo Recall/Precision/MRR, chạy bộ eval, và debug MỘT query thất bại theo 8 bước (slide 33).
import { get } from "../api.js";
import { dimsChart } from "../components.js";
import { callout, docLine, kv, ms, pill, rankPill, sqlBlock, table } from "../lesson/render.js";
import { esc, fmt } from "../util.js";
import { mountEvalRunner } from "./eval-runner.js";

const DEBUG_IDS = ["debug-hard", "debug-pool", "debug-vi", "debug-regress"];
const parse = (s) => s.split(/[,\s]+/).map((x) => x.trim()).filter(Boolean);

// ---------------------------------------------------------------- metric bằng tay
function metricSteps(rel, got, k) {
  const topK = got.slice(0, k), hits = topK.filter((x) => rel.includes(x));
  const missing = rel.filter((x) => !topK.includes(x));
  const first = got.findIndex((x) => rel.includes(x)) + 1;
  const recall = rel.length ? hits.length / rel.length : 0, precision = k ? hits.length / k : 0, mrr = first ? 1 / first : 0;
  const chip = (x, i) => `<span class="token ${rel.includes(x) ? "good" : ""} ${i >= k ? "dim" : ""}">${i + 1}. ${esc(x)}${rel.includes(x) ? " ✓" : ""}</span>`;
  return { recall, precision, mrr, hits, missing, steps: [
    { title: "Dữ liệu: đáp án đúng và kết quả", ms: null, summary: `${rel.length} doc đúng · lấy ${got.length}`,
      explain: "Cần hai thứ: <b>tập doc liên quan</b> (nhãn do người gán) và <b>danh sách retriever trả về</b>. Chỉ top-K đầu được tính (phần mờ nằm ngoài K).",
      html: `<p><b>Doc đúng:</b> ${rel.map((x) => `<span class="token good">${esc(x)}</span>`).join(" ")}</p><p><b>Retriever trả về:</b></p><div class="tokens">${got.map(chip).join("")}</div>` },
    { title: `Recall@${k}`, ms: null, summary: `${hits.length}/${rel.length} = ${fmt(recall, 2)}`,
      explain: `<b>Trong tất cả doc liên quan, bao nhiêu % được lấy về trong top-${k}?</b> Quan trọng nhất cho RAG: doc không được lấy về thì LLM không bao giờ thấy.`,
      html: `<div class="formula">Recall@${k} = (liên quan trong top-${k}) / (tổng liên quan) = <b>${hits.length} / ${rel.length} = ${fmt(recall, 2)}</b></div>${missing.length ? callout("warn", `Bỏ sót: ${missing.map((x) => `<b>${esc(x)}</b>`).join(", ")}`) : callout("good", "Không bỏ sót doc nào.")}` },
    { title: `Precision@${k}`, ms: null, summary: `${hits.length}/${k} = ${fmt(precision, 2)}`,
      explain: `<b>Trong top-${k} lấy về, bao nhiêu % thực sự liên quan?</b> Đo mức độ "nhiễu" mà LLM phải đọc.`,
      html: `<div class="formula">Precision@${k} = (liên quan trong top-${k}) / ${k} = <b>${hits.length} / ${k} = ${fmt(precision, 2)}</b></div>` },
    { title: "MRR (Mean Reciprocal Rank)", ms: null, summary: first ? `1/${first} = ${fmt(mrr, 2)}` : "không có",
      explain: "<b>Doc liên quan đầu tiên đứng hạng mấy?</b> Điểm = 1/hạng. Quan trọng khi chỉ cần một câu trả lời đúng.",
      html: `<div class="formula">MRR = 1 / (hạng của doc đúng đầu tiên) = ${first ? `<b>1 / ${first} = ${fmt(mrr, 2)}</b>` : "<b>0</b> (không có doc đúng)"}</div>` },
    { title: "Đọc kết quả", ms: null, summary: `R=${fmt(recall, 2)} P=${fmt(precision, 2)} MRR=${fmt(mrr, 2)}`,
      explain: "Ba số đo ba khía cạnh khác nhau; không có số nào đủ một mình.",
      html: kv([["Recall", `${fmt(recall, 2)} ${recall < 1 ? "→ còn doc bị bỏ sót" : "→ đủ"}`], ["Precision", `${fmt(precision, 2)} ${precision < 0.5 ? "→ nhiều nhiễu" : ""}`], ["MRR", `${fmt(mrr, 2)} ${mrr === 1 ? "→ doc đúng đứng đầu" : ""}`]]) },
  ] };
}

// ---------------------------------------------------------------- debug 8 bước
function debugSteps(d, ctx) {
  const firstRank = (name) => Math.min(...Object.values(d.ranks[name]).map((x) => x ?? 99));
  const show = (r) => (r === 99 ? null : r);
  const top5 = d.dense.slice(0, 5);
  const relInTop5 = top5.some((x) => x.relevant);
  const gap = d.gap;
  return [
    { title: "1 · Kiểm tra query", ms: null, summary: d.kind,
      explain: "Đầu tiên đọc lại <b>query và đáp án đúng</b>: query có rõ không, loại gì (mã chính xác / tự nhiên / paraphrase), doc nào là đáp án.",
      html: `<p>Query: <b>${esc(d.query)}</b> ${pill(d.kind)}</p><p class="hint">Từ khóa BM25 sau khi bỏ stopword:</p><div class="tokens">${d.tokens.map((t) => `<span class="token">${esc(t)}</span>`).join("") || "<i>(không còn từ nào)</i>"}</div>
        <p class="hint">Doc đúng (nhãn):</p>${d.relevant_docs.map((x) => docLine(x)).join("")}` },
    { title: "2 · Kiểm tra query embedding", ms: ms(d.embedding.ms), summary: `${d.embedding.dim} chiều · ‖v‖=${d.embedding.norm}`,
      explain: "Vector query có bất thường không (toàn 0, độ dài ≠ 1, số chiều sai)? Ở đây ổn: vector hợp lệ, đã normalize.",
      html: `${dimsChart(d.embedding.dims)}${kv([["Số chiều", d.embedding.dim], ["Độ dài ‖v‖", d.embedding.norm]])}` },
    { title: "3 · Kiểm tra các chunk được retrieve", ms: null, summary: relInTop5 ? "doc đúng có trong top-5" : "doc đúng KHÔNG trong top-5",
      explain: "Nhìn trực tiếp top-5 vector search: chúng có liên quan không? Doc đúng (✓) đứng ở đâu?",
      html: top5.map((x) => docLine(x, { score: x.score, mark: x.relevant ? true : undefined })).join("") + (relInTop5 ? "" : callout("warn", "Không doc nào trong top-5 là doc đúng: các mã/ý tưởng na ná nhau chiếm chỗ.")) },
    { title: "4 · Kiểm tra similarity scores", ms: null, summary: gap.relevant != null ? `doc đúng ${fmt(gap.relevant, 3)} vs top-1 ${fmt(gap.top1, 3)}` : "doc đúng ngoài pool",
      explain: "Khoảng cách điểm giữa doc đúng và các doc khác: nếu sát nhau, vector search <b>không đủ khả năng phân biệt</b> (e5 nén điểm vào dải hẹp ~0.7–1.0).",
      html: gap.relevant != null ? kv([["Điểm top-1", fmt(gap.top1, 4)], ["Điểm doc đúng", `<b>${fmt(gap.relevant, 4)}</b>`], ["Điểm hạng 10", gap.tenth != null ? fmt(gap.tenth, 4) : "–"], ["Chênh top-1 − doc đúng", fmt(gap.top1 - gap.relevant, 4)]])
        + (gap.top1 - gap.relevant < 0.02 ? callout("warn", "Chênh lệch dưới 0.02: các doc gần như đồng điểm nên thứ hạng dễ sai. Cần tín hiệu khác (keyword, reranker).") : "")
        : callout("bad", `Doc đúng không nằm trong ${d.filter.pool_size} ứng viên của vector search nên không có điểm để so.`) },
    { title: "5 · Kiểm tra metadata filter", ms: null, summary: Object.values(d.filter.relevant_passes).every(Boolean) ? "không loại nhầm" : "doc đúng bị lọc!",
      explain: "Filter có quá chặt, loại mất doc đúng không? Kiểm tra doc đúng có thỏa điều kiện tenant/status không.",
      html: `${sqlBlock(d.filter.sql)}${kv([["Kích thước pool", d.filter.pool_size], ...Object.entries(d.filter.relevant_passes).map(([id, ok]) => [`Doc ${id} qua filter?`, ok ? "<b class='ok'>✓ có</b>" : "<b class='no'>✗ bị loại</b>"])])}` },
    { title: "6 · So sánh vector với keyword", ms: null, summary: `vector #${show(firstRank("vector")) ?? "–"} · keyword #${show(firstRank("keyword")) ?? "–"} · hybrid #${show(firstRank("hybrid")) ?? "–"}`,
      explain: "BM25 có tìm thấy doc mà vector bỏ sót (hoặc ngược lại) không? Đây là lúc quyết định hybrid có đáng dùng.",
      html: table([{ t: "Doc đúng" }, { t: "Keyword" }, { t: "Vector" }, { t: "Hybrid (RRF)" }],
        Object.keys(d.ranks.vector).map((id) => [esc(id), rankPill(d.ranks.keyword[id]), rankPill(d.ranks.vector[id]), rankPill(d.ranks.hybrid[id])])) },
    { title: "7 · Kiểm tra reranker scores", ms: ms(d.ms.hybrid_rerank), summary: `sau rerank #${show(firstRank("hybrid_rerank")) ?? "–"}`,
      explain: "Reranker có đồng ý với retriever không? Điểm cross-encoder của doc đúng so với các doc đứng trên.",
      html: d.rerank.map((x, i) => docLine({ ...x, id: x.id }, { score: x.score, mark: x.relevant ? true : undefined })).join("") +
        `<p class="hint">Hạng doc đúng trước → sau rerank: ${Object.keys(d.ranks.hybrid).map((id) => `${rankPill(d.ranks.hybrid[id])} → ${rankPill(d.ranks.hybrid_rerank[id])}`).join(" · ")}</p>` },
    { title: "8 · Đo bằng số trên query này", ms: null, summary: d.stage,
      explain: "Chốt bằng số liệu: hạng doc đúng của từng cấu hình. Từ đó biết lỗi nằm ở retrieval, ranking, hay do reranker.",
      html: table([{ t: "Cấu hình" }, { t: "Hạng doc đúng" }, { t: "Recall@3", num: true }, { t: "MRR", num: true }],
        Object.entries({ keyword: "Keyword", vector: "Vector", hybrid: "Hybrid", vector_rerank: "Vector + Rerank", hybrid_rerank: "Hybrid + Rerank" }).map(([k, label]) =>
          [label, rankPill(d.metrics[k].first_hit), fmt(d.metrics[k].recall), fmt(d.metrics[k].mrr)])) },
  ];
}

const STAGE = {
  retrieval: { level: "bad", title: "Lỗi ở RETRIEVAL: doc đúng không vào được pool ứng viên", hint: "Reranker không thể cứu doc mà retriever đã bỏ lỡ. Hãy cải thiện chunking/embedding/hybrid hoặc tăng pool." },
  ranking: { level: "bad", title: "Lỗi ở RANKING: doc đúng có trong pool nhưng bị xếp thấp", hint: "Xem reranker, ngưỡng top-K, hoặc bổ sung tín hiệu xếp hạng." },
  regressed: { level: "bad", title: "Reranker làm tệ đi", hint: "Retrieval làm đúng nhưng bước rerank đẩy doc đúng xuống. Đo reranker trên bộ eval trước khi bật." },
  recovered: { level: "good", title: "Retrieval có nhánh sai nhưng pipeline đã cứu được", hint: "Các bước sau (hybrid/rerank) bù lại điểm yếu của retrieval ban đầu." },
  ok: { level: "good", title: "Mọi bước đều đưa doc đúng lên đầu", hint: "" },
};

export default {
  id: "evaluate", num: 7, group: "C", title: "Đánh giá & Debug", slides: "21, 23, 25, 33",
  tagline: "Đo thay vì cảm giác: Recall/Precision/MRR, bộ eval, và quy trình debug 8 bước khi một query thất bại.",
  theory: {
    what: `<p><b>Recall@K</b>: lấy về được bao nhiêu % doc liên quan. <b>Precision@K</b>: trong top-K có bao nhiêu % liên quan. <b>MRR</b>: doc đúng đầu tiên đứng hạng mấy. Cộng thêm <b>latency</b>.</p>
      <p>Cần một <b>bộ eval có nhãn</b>: cặp (query, doc đúng).</p>`,
    why: `<p>Chất lượng retrieval ≠ chất lượng RAG: retrieval sai thì LLM nhận ngữ cảnh sai và trả lời sai nhưng tự tin. <b>Debug retrieval trước khi đổ lỗi cho LLM.</b></p>`,
    points: ["\"Cảm giác tốt hơn\" không phải kỹ thuật: đo là bắt buộc.", "Đo Recall@K <b>độc lập</b> với chất lượng generation.", "Khi một query sai: xác định lỗi ở retrieval, filter hay ranking."],
    visual: `<svg viewBox="0 0 760 100" class="theorysvg" role="img" aria-label="8 bước debug">
      <text x="10" y="22" font-size="13" font-weight="700">Debug retrieval (slide 33):</text>
      ${["query", "embedding", "chunks", "scores", "filter", "vector/BM25", "reranker", "đo bằng số"].map((t, i) => `<rect x="${10 + i * 92}" y="40" width="86" height="40" rx="6" fill="${i < 4 ? "#e0ecff" : "#fef3c7"}" stroke="${i < 4 ? "#3b82f6" : "#d97706"}"/><text x="${53 + i * 92}" y="64" text-anchor="middle" font-size="11">${i + 1}. ${t}</text>`).join("")}</svg>`,
  },
  fields: [
    { name: "relevant", label: "Doc liên quan (cách nhau bằng dấu phẩy)", type: "text", showFor: ["metrics"], wide: true },
    { name: "retrieved", label: "Doc retriever trả về, theo thứ tự", type: "text", showFor: ["metrics"], wide: true },
    { name: "k", label: "K", type: "range", min: 1, max: 10, showFor: ["metrics"] },
    { name: "qid", label: "Query trong bộ eval", type: "select", showFor: DEBUG_IDS, options: [
      { value: "q34", label: "q34 · ERR-4102 (mã lỗi)" }, { value: "q15", label: "q15 · how many vacation days do I get" }, { value: "q18", label: "q18 · chuyển hàng giữa hai kho" },
      { value: "q13", label: "q13 · my account is locked after typing the wrong password" }, { value: "q26", label: "q26 · connect to the company network from home" }] },
    { name: "pool", label: "Pool ứng viên của retriever", type: "range", min: 5, max: 30, showFor: DEBUG_IDS },
  ],
  cases: [
    { id: "metrics", tag: "observe", title: "Tính Recall@K, Precision@K, MRR bằng tay", slide: "21",
      story: "Ví dụ: doc liên quan {A,B,C}, top-5 lấy về {A,X,B,Y,Z}: A và B được lấy về, C bị bỏ sót → Recall@5 = 2/3 ≈ 0.67.",
      expect: "Recall@5 = 2/3, Precision@5 = 2/5, MRR = 1. Đổi K hoặc danh sách để thấy đánh đổi Recall–Precision.",
      values: { relevant: "A, B, C", retrieved: "A, X, B, Y, Z", k: 5 },
      fixes: [{ label: "Tăng K lên 10 (Recall tăng, Precision giảm)", patch: { k: 10 } }, { label: "K = 2 (Precision cao, Recall thấp)", patch: { k: 2 } }] },
    { id: "runner", tag: "observe", title: "Chạy toàn bộ bộ eval", slide: "21, 23",
      story: "Tạo evaluation dataset (query-doc có nhãn) → chạy pipeline → tính Recall@K, Precision@K, MRR → so sánh các cấu hình.",
      expect: "5 cấu hình trên 36 query: rerank cải thiện lớn nhất; hybrid sửa lỗi mã chính xác của vector.",
      custom: (el, ctx) => mountEvalRunner(el, ctx) },
    { id: "debug-hard", tag: "problem", title: "Debug: doc đúng ở hạng 6", slide: "33",
      story: "Đừng debug RAG chỉ bằng cách nhìn câu trả lời cuối của LLM: kiểm tra từng bước, từ query tới reranker.",
      expect: "ERR-4102: vector hạng 6, BM25 hạng 1, hybrid hạng 3, rerank hạng 1 → pipeline cứu được.",
      values: { qid: "q34", pool: 20 } },
    { id: "debug-pool", tag: "error", title: "Debug: doc đúng suýt rơi khỏi pool", slide: "20, 33",
      story: "Retriever lấy Top-50 ứng viên rồi mới rerank; doc ngoài pool thì reranker không thể cứu.",
      expect: "\"how many vacation days\": doc đúng ở hạng 20/20. Giảm pool xuống 10 thì lỗi nằm ở RETRIEVAL.",
      values: { qid: "q15", pool: 20 }, fixes: [{ label: "Thử pool = 10 (doc rơi ra ngoài)", patch: { pool: 10 } }, { label: "Pool = 30 (an toàn)", patch: { pool: 30 } }] },
    { id: "debug-vi", tag: "problem", title: "Debug: query tiếng Việt, doc tiếng Anh", slide: "9, 33",
      story: "Multilingual có thể kém hơn monolingual; mỗi lỗi cần được truy về đúng tầng.",
      expect: "\"chuyển hàng giữa hai kho\" tìm doc tiếng Anh ở hạng 15 vì embedding tách vi/en; rerank cứu được nếu pool đủ lớn.",
      values: { qid: "q18", pool: 20 }, fixes: [{ label: "Thử pool = 10", patch: { pool: 10 } }] },
    { id: "debug-regress", tag: "error", title: "Debug: reranker làm tệ đi", slide: "20, 33",
      story: "Kiểm tra reranker scores: reranker có đồng ý với retriever không?",
      expect: "\"connect to the company network from home\": retrieval đúng hạng 1 nhưng rerank đẩy xuống hạng 3.",
      values: { qid: "q26", pool: 20 } },
  ],
  async run(v, ctx, current) {
    if (current.id === "metrics") {
      const rel = parse(v.relevant), got = parse(v.retrieved);
      if (!rel.length || !got.length) throw new Error("Nhập ít nhất một doc liên quan và một doc retriever trả về.");
      const m = metricSteps(rel, got, v.k);
      return { steps: m.steps,
        diagnosis: m.missing.length ? { level: "warn", title: `Recall@${v.k} = ${fmt(m.recall, 2)}: bỏ sót ${m.missing.join(", ")}`, html: `Precision@${v.k} = ${fmt(m.precision, 2)}. Tăng K thường tăng Recall nhưng <b>giảm Precision</b> (thêm nhiễu): đó là đánh đổi cơ bản.` }
          : { level: "good", title: `Recall@${v.k} = 1.00: không bỏ sót doc nào`, html: `Precision@${v.k} = ${fmt(m.precision, 2)}${m.precision < 1 ? ": vẫn còn nhiễu trong top-K." : "."}` },
        lessons: ["Recall quan trọng nhất cho RAG: doc không được lấy về thì LLM không thể dùng.", "Precision cho biết LLM phải đọc bao nhiêu nhiễu; MRR cho biết doc đúng có lên đầu không.", "Tối ưu retriever cho recall, reranker cho precision."],
        summary: { label: `K = ${v.k}`, facts: { "Recall": fmt(m.recall, 2), "Precision": fmt(m.precision, 2), "MRR": fmt(m.mrr, 2) } } };
    }
    const d = await get(`/api/lesson/debug/run?qid=${encodeURIComponent(v.qid)}&pool=${v.pool}`);
    const st = STAGE[d.stage];
    return { steps: debugSteps(d, ctx), diagnosis: { level: st.level, title: st.title, html: `${esc(d.detail)} ${st.hint}` },
      lessons: ["Debug theo <b>tầng</b>: query → embedding → chunks → scores → filter → vector vs keyword → rerank → số đo.", "Phân biệt <b>lỗi retrieval</b> (không vào pool) với <b>lỗi ranking</b> (có trong pool nhưng xếp thấp): cách sửa khác nhau.", "Ghi log query, chunk, điểm và filter của mỗi request thì mới debug được production (bài 8)."],
      summary: { label: `${d.id} · pool ${d.filter.pool}`, facts: {
        "Hạng sau hybrid": d.ranks.hybrid && Object.values(d.ranks.hybrid)[0] != null ? Object.values(d.ranks.hybrid)[0] : { v: "ngoài pool", level: "bad" },
        "Hạng sau rerank": Object.values(d.ranks.hybrid_rerank)[0] != null ? (Object.values(d.ranks.hybrid_rerank)[0] === 1 ? { v: 1, level: "good" } : Object.values(d.ranks.hybrid_rerank)[0]) : { v: "ngoài pool", level: "bad" },
        "Chẩn đoán": { v: d.stage, level: STAGE[d.stage].level } } } };
  },
};
