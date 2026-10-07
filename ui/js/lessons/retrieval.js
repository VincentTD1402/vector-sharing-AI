// Bài 5 - Retrieval & Filter: BM25 vs vector vs hybrid (RRF) và ràng buộc nghiệp vụ (tenant, trạng thái).
import { post } from "../api.js";
import { bar, callout, kv, ms, pill, rankPill, sqlBlock, table } from "../lesson/render.js";
import { esc, fmt } from "../util.js";

const MODE_TEXT = {
  pre: ["good", "<b>Pre-filter</b>: điều kiện nằm trong WHERE nên DB chỉ xếp hạng các dòng hợp lệ → luôn đủ k kết quả (nếu có dữ liệu)."],
  post: ["warn", "<b>Post-filter</b>: lấy top-k thuần similarity rồi mới loại dòng không hợp lệ → có thể trả THIẾU (doc đúng của tenant có thể nằm ngoài top-k)."],
  off: ["bad", "<b>Không filter</b>: similarity quyết định tất cả → dữ liệu của tenant khác và tài liệu lỗi thời lọt vào kết quả."],
};

function steps(d) {
  const f = d.filters, c = Object.fromEntries(d.configs.map((x) => [x.key, x]));
  const maxBm = Math.max(...d.bm25.map((b) => b.score), 0.001);
  const [lvl, modeHtml] = MODE_TEXT[f.filter_mode];
  const leaked = d.dense.filter((x) => x.leaked).length;
  return [
    { title: "Query và ràng buộc", ms: null,
      explain: "Trước khi tìm, hệ thống xác định <b>ai đang hỏi</b> (tenant) và <b>được xem gì</b> (tài liệu còn hiệu lực). Đây là ràng buộc nghiệp vụ, không phải chuyện similarity.",
      summary: `tenant=${f.tenant_id} · ${f.filter_mode}-filter`,
      html: `${kv([["Query", `<b>${esc(d.query)}</b>`], ["Tenant", esc(f.tenant_id)], ["Cách lọc", pill(f.filter_mode, { pre: "good", post: "warn", off: "bad" }[f.filter_mode])]])}
        <p class="hint">Câu SQL mà nhánh vector thực thi:</p>${sqlBlock(d.sql)}${callout(lvl, modeHtml)}` },
    { title: "Nhánh từ khóa (BM25)", ms: ms(d.stages_ms.lexical),
      explain: "BM25 tách query thành từ, bỏ stopword, rồi chấm theo TF-IDF: từ càng hiếm càng nặng. Chỉ doc <b>có chứa ít nhất một từ</b> mới có điểm; không khớp từ nào thì không được trả về.",
      summary: d.bm25.length ? `${d.counts.lexical} doc khớp từ` : "không doc nào khớp từ",
      html: `<div class="tokens">${d.bm25_tokens.map((t) => `<span class="token">${esc(t)}</span>`).join("") || "<i>(không còn từ nào sau khi bỏ stopword)</i>"}</div>
        ${d.bm25.length ? table([{ t: "Doc" }, { t: "BM25", num: true }, { t: "Từ khớp" }, { t: "Nội dung" }],
          d.bm25.slice(0, 5).map((b) => [esc(b.id), `<div style="min-width:90px">${bar(b.score, maxBm)}<span class="hint">${fmt(b.score, 2)}</span></div>`,
            b.matched.map((t) => `<span class="token">${esc(t)}</span>`).join(" "), esc(b.content.slice(0, 70)) + "…"]),
          (i) => (d.bm25[i].relevant ? "rel" : ""))
        : callout("bad", "BM25 <b>không tìm được gì</b>: query không có từ nào trùng với tài liệu (paraphrase, khác ngôn ngữ, hoặc toàn stopword).")}`,
      warn: d.bm25.length ? null : "Đây là điểm yếu của keyword search được nêu trong slide 11/19." },
    { title: "Nhánh ngữ nghĩa (vector)", ms: ms(d.stages_ms.embed + d.stages_ms.dense),
      explain: `Query → vector 384 chiều (${ms(d.stages_ms.embed)}) → pgvector tìm ${d.counts.dense} doc gần nhất theo cosine (${ms(d.stages_ms.dense)}). Luôn trả đủ k kết quả vì mọi doc đều có một khoảng cách.`,
      summary: `top-1 cosine ${d.dense[0] ? fmt(d.dense[0].score, 3) : "—"}${leaked ? ` · ${leaked} rò rỉ` : ""}`,
      html: table([{ t: "Hạng", num: true }, { t: "Doc" }, { t: "Cosine", num: true }, { t: "Tenant / status" }, { t: "Nội dung" }],
        d.dense.slice(0, 6).map((x, i) => [i + 1, esc(x.id), `<div style="min-width:90px">${bar(x.score, 1)}<span class="hint">${fmt(x.score, 3)}</span></div>`,
          x.leaked ? pill(`${x.tenant_id} · ${x.status} (RÒ RỈ)`, "bad") : pill(`${x.tenant_id} · ${x.status}`), esc(x.content.slice(0, 70)) + "…"]),
        (i) => (d.dense[i].leaked ? "bad-row" : d.dense[i].relevant ? "rel" : "")) +
        `<p class="hint">Điểm cosine của e5 nằm trong dải hẹp ~0.7–1.0 nên các doc không liên quan vẫn có điểm cao: chỉ thứ hạng mới đáng tin.</p>`,
      warn: leaked ? `Có ${leaked} doc của tenant khác / lỗi thời trong kết quả vector.` : null },
    { title: "Hợp nhất bằng RRF", ms: ms(d.stages_ms.fusion),
      explain: "Hai danh sách dùng thang điểm khác nhau (BM25 ~0–15, cosine ~0.7–1) nên không cộng thẳng được. <b>RRF</b> chỉ dùng <b>thứ hạng</b>: mỗi danh sách đóng góp 1/(60 + hạng).",
      summary: `${d.rrf.length} ứng viên`,
      html: table([{ t: "Doc" }, { t: "Hạng BM25", num: true }, { t: "Hạng vector", num: true }, { t: "Điểm RRF", num: true }],
        d.rrf.slice(0, 6).map((r) => [esc(r.id), r.rank_lexical ?? "–", r.rank_dense ?? "–", fmt(r.score, 4)]),
        (i) => ((d.relevant || []).includes(d.rrf[i].id) ? "rel" : "")) },
    { title: "Kết quả cuối và so sánh", ms: null,
      explain: `Cùng một query, ba cách tìm cho ba kết quả khác nhau. Yêu cầu top-${d.top_k}; cột "hạng đúng" cho biết doc đúng đứng thứ mấy (khi query có nhãn).`,
      summary: Object.entries(d.ranks).map(([k, r]) => `${k}:${r ?? "–"}`).join(" · "),
      html: table([{ t: "Cách tìm" }, { t: "Trả về" }, { t: "Hạng doc đúng" }, { t: "Top-3" }],
        d.configs.map((x) => [`<b>${esc(x.label)}</b>`, `${x.returned}/${d.top_k}`, d.relevant ? rankPill(d.ranks[x.key]) : "<i>không có nhãn</i>",
          x.results.slice(0, 3).map((r) => `<span class="badge ${r.leaked ? "bad" : r.relevant ? "good" : ""}">${esc(r.id)}</span>`).join(" ")])) },
  ];
}

function diagnose(d) {
  const f = d.filters, c = Object.fromEntries(d.configs.map((x) => [x.key, x]));
  const leaked = d.configs.flatMap((x) => x.results).filter((r) => r.leaked);
  const stale = leaked.filter((r) => r.status !== "current" && r.tenant_id === f.tenant_id);
  if (f.filter_mode === "off" && stale.length) {
    return { level: "bad", title: "Tài liệu lỗi thời lọt vào kết quả",
      html: `Không lọc <code>status = 'current'</code> nên <b>${esc([...new Set(stale.map((r) => r.id))].join(", "))}</b> (đã deprecated) đứng cạnh hướng dẫn mới và có thể được chọn làm câu trả lời sai. Version control phải nằm trong filter, không phải trong similarity.` };
  }
  if (f.filter_mode === "off" && leaked.length) {
    const tenants = [...new Set(leaked.map((r) => r.tenant_id + (r.status !== "current" ? "/" + r.status : "")))];
    return { level: "bad", title: "Rò rỉ dữ liệu: kết quả chứa tài liệu không được phép xem",
      html: `Không có filter nên similarity quyết định hết: ${leaked.length} lượt doc thuộc <b>${esc(tenants.join(", "))}</b> xuất hiện. Trong hệ thống nhiều tenant đây là lỗi nghiêm trọng và <b>không thể sửa bằng cách tăng Top-K</b>.` };
  }
  if (f.filter_mode === "post" && c.vector.returned < d.top_k) {
    return { level: "bad", title: `Post-filter chỉ trả ${c.vector.returned}/${d.top_k} kết quả (vector)`,
      html: `Top-${d.top_k} thuần similarity chứa doc của tenant khác nên sau khi lọc còn thiếu. Doc đúng của tenant có thể nằm ở hạng 50 hay 200. Cách khắc phục "tăng Top-K trước khi lọc" vừa tốn vừa không đảm bảo.` };
  }
  const r = d.ranks;
  if (d.relevant) {
    if (r.keyword === 1 && (r.vector == null || r.vector > 3)) return { level: "warn", title: "Từ khóa tìm đúng nhưng vector bỏ lỡ",
      html: `BM25 xếp doc đúng hạng <b>#1</b>, vector chỉ hạng <b>${r.vector ?? "ngoài top"}</b>, hybrid kéo lên <b>#${r.hybrid ?? "–"}</b>. Mã/ID chính xác là sở trường của keyword; embedding nén thông tin nên các mã na ná nhau bị lẫn. Hybrid chưa chắc đưa doc lên đầu: cần thêm bước <a href="#/rerank?q=${encodeURIComponent(d.query)}">rerank (bài 6)</a>.` };
    if (r.keyword == null && r.vector != null && r.vector <= 3) return { level: "info", title: "Từ khóa không tìm được gì, vector tìm đúng",
      html: `Query là diễn đạt khác (paraphrase) nên không trùng từ với doc: BM25 hạng <b>${r.keyword ?? "không có"}</b>, vector hạng <b>#${r.vector}</b>. Đây là lý do cần semantic search; và hybrid giữ được kết quả tốt (#${r.hybrid}) nhờ nhánh vector.` };
    if (r.hybrid === 1) return { level: "good", title: "Hybrid đưa doc đúng lên hạng 1",
      html: `BM25 #${r.keyword ?? "–"}, vector #${r.vector ?? "–"} → RRF #${r.hybrid}. Hai nhánh bổ sung cho nhau.` };
    return { level: "warn", title: `Doc đúng đứng hạng #${r.hybrid ?? "ngoài top"} sau hybrid`, html: "Chưa lên đầu: cân nhắc rerank (bài 6) hoặc điều chỉnh retrieval." };
  }
  return { level: "info", title: "Query này chưa có nhãn đáp án", html: "Hãy chọn một tình huống mẫu (có nhãn) để thấy hạng của doc đúng; ở đây bạn vẫn quan sát được từng nhánh và cách lọc." };
}

export default {
  id: "retrieval", num: 5, group: "B", title: "Retrieval & Filter", slides: "11, 16, 17, 18, 19",
  tagline: "Tìm ứng viên: BM25 (từ khóa) + vector (ngữ nghĩa) + RRF, cùng ràng buộc nghiệp vụ (tenant, trạng thái).",
  theory: {
    what: `<p><b>Keyword search (BM25)</b> khớp từ chính xác. <b>Semantic search</b> so khoảng cách vector. <b>Hybrid</b> chạy cả hai rồi gộp bằng <b>RRF</b>.</p>
      <p><b>Metadata filter</b> áp ràng buộc nghiệp vụ (tenant, status, language) lên tập tìm kiếm.</p>`,
    why: `<p>Không phương pháp nào thắng mọi query: keyword bỏ sót paraphrase, semantic có thể lẫn các mã chính xác.</p>
      <p>Similarity tìm "liên quan về nghĩa" nhưng <b>không biết ai được xem gì</b>: phân quyền phải do filter đảm bảo.</p>`,
    points: ["<b>Pre-filter</b> (WHERE trước khi tìm) đúng và đủ k; <b>post-filter</b> (lọc sau top-k) có thể thiếu.",
      "Isolation tenant là ràng buộc <b>bắt buộc</b>, ép ở tầng API/DB; không để similarity quyết định.",
      "RRF chỉ dùng thứ hạng nên không cần chuẩn hóa điểm của hai thang khác nhau."],
    visual: `<svg viewBox="0 0 760 120" class="theorysvg" role="img" aria-label="Sơ đồ hybrid">
      <rect x="10" y="10" width="130" height="36" rx="6" fill="#e0ecff" stroke="#3b82f6"/><text x="75" y="33" text-anchor="middle" font-size="13">BM25 → list A</text>
      <rect x="10" y="66" width="130" height="36" rx="6" fill="#dcfce7" stroke="#16a34a"/><text x="75" y="89" text-anchor="middle" font-size="13">Vector → list B</text>
      <text x="165" y="60" font-size="22">→</text><rect x="190" y="34" width="150" height="50" rx="6" fill="#fef3c7" stroke="#d97706"/><text x="265" y="64" text-anchor="middle" font-size="13">RRF: Σ 1/(60+hạng)</text>
      <text x="360" y="60" font-size="22">→</text><rect x="385" y="34" width="110" height="50" rx="6" fill="#ede9fe" stroke="#7c3aed"/><text x="440" y="64" text-anchor="middle" font-size="13">Top-K</text>
      <rect x="530" y="10" width="220" height="92" rx="8" fill="#fde8e8" stroke="#b91c1c"/><text x="640" y="36" text-anchor="middle" font-size="13" font-weight="700" fill="#b91c1c">Filter tenant / status</text>
      <text x="640" y="58" text-anchor="middle" font-size="12" fill="#7f1d1d">phải áp ở CẢ hai nhánh,</text><text x="640" y="76" text-anchor="middle" font-size="12" fill="#7f1d1d">TRƯỚC khi lấy top-k</text></svg>`,
  },
  fields: [
    { name: "query", label: "Query", type: "text", wide: true },
    { name: "tenant", label: "Tenant của người hỏi", type: "select", options: ["company_a", "company_b", "company_c"].map((x) => ({ value: x, label: x })) },
    { name: "filter_mode", label: "Cách lọc", type: "select", options: [{ value: "pre", label: "Pre-filter (đúng)" }, { value: "post", label: "Post-filter (lọc sau top-k)" }, { value: "off", label: "Tắt filter (rò rỉ)" }] },
    { name: "top_k", label: "Top-K", type: "range", min: 1, max: 10 },
    { name: "language", label: "Ngôn ngữ tài liệu", type: "select", options: [{ value: "", label: "Tất cả" }, { value: "vi", label: "Tiếng Việt" }, { value: "en", label: "Tiếng Anh" }] },
  ],
  cases: [
    { id: "exact-code", tag: "problem", title: "Mã lỗi chính xác: vector tìm sai", slide: "19",
      story: "Bảng \"khi nào semantic thất bại\": mã lỗi ERR_… thì BM25 tốt nhất, vector kém. Embedding nén thông tin nên mất các mẫu từ vựng chính xác.",
      expect: "BM25 xếp ERR-4102 hạng #1; vector chỉ #6 (ngoài top-3) vì lẫn với ERR-4210, ERR-4120…; hybrid kéo lên #3 nhưng chưa phải #1.",
      values: { query: "ERR-4102", tenant: "company_a", filter_mode: "pre", top_k: 3, language: "" },
      fixes: [{ label: "Qua bài 6: thêm bước Rerank", href: "#/rerank?q=ERR-4102" }] },
    { id: "paraphrase", tag: "problem", title: "Paraphrase: từ khóa không trùng", slide: "11, 19",
      story: "Keyword search bỏ sót paraphrase; semantic search tìm theo nghĩa nên vẫn thấy doc đúng.",
      expect: "BM25 không tìm được doc nào đúng (không từ nào trùng); vector xếp doc đúng ở hạng đầu.",
      values: { query: "how do I get my business expenses refunded", tenant: "company_a", filter_mode: "pre", top_k: 3, language: "" } },
    { id: "hybrid-ok", tag: "fix", title: "Hybrid: hai nhánh bổ sung cho nhau", slide: "19",
      story: "Hybrid = BM25 + vector + RRF; thường bền vững hơn trong enterprise search với query hỗn hợp.",
      expect: "Query tự nhiên: cả hai nhánh cùng tìm đúng, RRF cộng dồn nên doc đúng lên hạng 1.",
      values: { query: "How to approve an invoice", tenant: "company_a", filter_mode: "pre", top_k: 5, language: "" } },
    { id: "leak", tag: "error", title: "Rò rỉ dữ liệu multi-tenant", slide: "16",
      story: "Top-K ngữ nghĩa có thể gồm tài liệu của công ty khác. Rò rỉ dữ liệu multi-tenant là lỗi nghiêm trọng, không sửa được bằng cách tăng Top-K.",
      expect: "Tắt filter: doc của company_b và company_c (viền đỏ) lọt vào kết quả của company_a.",
      values: { query: "How to approve an invoice", tenant: "company_a", filter_mode: "off", top_k: 5, language: "" },
      fixes: [{ label: "Bật pre-filter (WHERE tenant_id)", patch: { filter_mode: "pre" } }] },
    { id: "post", tag: "error", title: "Post-filter trả thiếu kết quả", slide: "17",
      story: "Top-10 semantic chỉ có 1 kết quả thuộc company_id đúng; doc đúng có thể nằm ở rank 50, 100, 200.",
      expect: "Lấy top-3 thuần similarity rồi lọc: vector chỉ còn 1/3 kết quả.",
      values: { query: "How to approve an invoice", tenant: "company_a", filter_mode: "post", top_k: 3, language: "" },
      fixes: [{ label: "Dùng pre-filter", patch: { filter_mode: "pre" } }] },
    { id: "outdated", tag: "error", title: "Tài liệu lỗi thời lọt vào kết quả", slide: "16",
      story: "Version control: chỉ dùng tài liệu hiện hành, không dùng tài liệu lỗi thời.",
      expect: "Tắt filter: d06 (\"gọi helpdesk ext 100\", đã deprecated) xuất hiện cạnh hướng dẫn mới.",
      values: { query: "I forgot my password", tenant: "company_a", filter_mode: "off", top_k: 4, language: "" },
      fixes: [{ label: "Lọc status = current", patch: { filter_mode: "pre" } }] },
    { id: "free", tag: "free", title: "Tự thử", slide: "11-19", story: "Tự nhập query, chọn tenant và cách lọc.", expect: "Thử query của bạn ở cả ba chế độ lọc.",
      values: { query: "ERR-4021", tenant: "company_a", filter_mode: "pre", top_k: 5, language: "" } },
  ],
  async run(v) {
    const d = await post("/api/retrieve", { query: v.query, tenant_id: v.tenant, top_k: v.top_k, language: v.language || null, filter_mode: v.filter_mode });
    const c = Object.fromEntries(d.configs.map((x) => [x.key, x]));
    const res = diagnose(d);
    return {
      steps: steps(d), diagnosis: res,
      lessons: ["Không có cách tìm nào thắng mọi query: <b>mỗi cách có kiểu lỗi riêng</b>, nên cần đo trên bộ query thật (bài 7).",
        "Filter phân quyền không phải việc của similarity: hãy ép <b>tenant_id ở tầng DB/API</b> và lọc TRƯỚC khi lấy top-k.",
        "RRF gộp theo thứ hạng nên đơn giản và bền, nhưng không sửa được mọi lỗi (cần rerank)."],
      summary: { label: `${v.filter_mode}-filter · top-${v.top_k} · ${v.tenant}`, facts: {
        "Keyword: hạng đúng": d.relevant ? (d.ranks.keyword ?? { v: "không có", level: "bad" }) : "—",
        "Vector: hạng đúng": d.relevant ? (d.ranks.vector ?? { v: "không có", level: "bad" }) : "—",
        "Hybrid: hạng đúng": d.relevant ? (d.ranks.hybrid ?? { v: "không có", level: "bad" }) : "—",
        "Vector trả về": { v: `${c.vector.returned}/${v.top_k}`, level: c.vector.returned < v.top_k ? "bad" : "good" },
        "Doc rò rỉ": { v: d.configs.flatMap((x) => x.results).filter((r) => r.leaked).length, level: d.configs.flatMap((x) => x.results).some((r) => r.leaked) ? "bad" : "good" } } },
    };
  },
};
