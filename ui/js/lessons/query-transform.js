// Bài 4 - Query transformation: query ngắn/viết tắt → biến đổi trước khi embed (rewrite, expansion, multi-query).
import { post } from "../api.js";
import { callout, docLine, ms, rankPill, table } from "../lesson/render.js";
import { esc, fmt } from "../util.js";

const top = (res, n = 3) => res.slice(0, n).map((r) => docLine(r, { score: r.score, mark: r.relevant })).join("");
const run = (d, mode) => d.runs.find((r) => r.mode === mode);

function steps(d) {
  const o = run(d, "none"), rw = run(d, "rewrite"), ex = run(d, "expand"), mu = run(d, "multi");
  const abbr = Object.entries(d.rules.abbreviations), syn = Object.entries(d.rules.synonyms);
  const rank = (r) => (d.relevant ? rankPill(r.metrics?.first_hit) : "");
  return [
    { title: "Query gốc", ms: ms(o.ms),
      explain: "Query người dùng thường <b>ngắn, viết tắt, hoặc khác từ trong tài liệu</b>. Embedding của chuỗi quá ngắn kém đại diện, nên kết quả vector search không ổn định.",
      summary: d.relevant ? `doc đúng hạng #${o.metrics?.first_hit ?? "–"}` : "không có nhãn",
      html: `<div class="tokens"><span class="token">${esc(d.query)}</span></div><p class="hint">Top-3 vector search với query gốc:</p>${top(o.results)}
        ${d.relevant ? `<p>Doc đúng đứng ${rank(o)}</p>` : ""}` },
    { title: "Rewrite: mở viết tắt", ms: null,
      explain: "Viết lại query rõ ràng hơn. Ở đây dùng từ điển viết tắt (offline, minh bạch); production dùng LLM với cùng cấu trúc: <code>rewrite(query) → query'</code>.",
      summary: abbr.length ? abbr.map(([a, b]) => `${a} → ${b}`).join(", ") : "không có viết tắt",
      html: `<div class="rewrite"><span class="token">${esc(d.query)}</span> <b>→</b> <span class="token good">${esc(rw.variants[0])}</span></div>
        ${abbr.length ? `<p class="hint">Luật đã áp dụng: ${abbr.map(([a, b]) => `<code>${esc(a)}</code> → <code>${esc(b)}</code>`).join(", ")}</p>` : callout("info", "Không có từ viết tắt nào khớp từ điển nên rewrite không thay đổi gì.")}
        <p class="hint">Top-3 sau rewrite:</p>${top(rw.results)}<p>Doc đúng đứng ${rank(rw)}</p>` },
    { title: "Expansion: thêm từ đồng nghĩa", ms: null,
      explain: "Thêm từ đồng nghĩa / từ liên quan (kể cả tiếng Việt) để tăng cơ hội khớp với cách tài liệu diễn đạt. Nhưng <b>thêm từ cũng có thể thêm nhiễu</b>.",
      summary: syn.length ? `+${syn.reduce((n, [, b]) => n + b.length, 0)} từ` : "không thêm được từ nào",
      html: `<div class="rewrite"><span class="token good">${esc(rw.variants[0])}</span> <b>+</b> ${syn.flatMap(([, b]) => b).map((x) => `<span class="token warn">${esc(x)}</span>`).join(" ") || "<i>(không có)</i>"}</div>
        <p class="hint">Top-3 sau expansion:</p>${top(ex.results)}<p>Doc đúng đứng ${rank(ex)}</p>` },
    { title: "Multi-query: tìm bằng nhiều biến thể", ms: ms(mu.ms),
      explain: "Tạo nhiều cách diễn đạt (gốc + rewrite + expansion), <b>tìm từng biến thể</b> rồi gộp kết quả. Mỗi biến thể tốn thêm một lần embed (hoặc một lần gọi LLM) → cộng vào latency.",
      summary: `${mu.variants.length} biến thể`,
      html: (mu.per_variant ?? []).map((p, i) => `<div class="variantbox"><b>Biến thể ${i + 1}:</b> <span class="token">${esc(p.variant)}</span>${top(p.results, 2)}</div>`).join("") },
    { title: "Gộp bằng RRF", ms: null,
      explain: "Gộp danh sách của các biến thể bằng RRF (chỉ dùng thứ hạng): doc xuất hiện cao ở nhiều biến thể được cộng dồn → ổn định hơn từng biến thể riêng lẻ.",
      summary: d.relevant ? `doc đúng hạng #${mu.metrics?.first_hit ?? "–"}` : "",
      html: `<p class="hint">Top-3 sau khi gộp:</p>${top(mu.results)}<p>Doc đúng đứng ${rank(mu)}</p>` },
    { title: "So sánh 4 cách", ms: null,
      explain: "Cùng một query, vector search, chỉ khác bước biến đổi. Hạng doc đúng càng nhỏ càng tốt.",
      summary: d.runs.map((r) => r.metrics?.first_hit ?? "–").join(" · "),
      html: d.relevant ? table([{ t: "Cách biến đổi" }, { t: "Hạng doc đúng" }, { t: "MRR", num: true }, { t: "Latency", num: true }],
        d.runs.map((r) => [esc(r.label), rankPill(r.metrics?.first_hit), fmt(r.metrics?.mrr), ms(r.ms)])) : callout("info", "Query này không có nhãn đáp án; chọn một tình huống mẫu để so sánh hạng.") },
  ];
}

function diagnose(d) {
  if (!d.relevant) return { level: "info", title: "Query này chưa có nhãn đáp án", html: "Chọn một tình huống mẫu để thấy hạng doc đúng của từng cách biến đổi." };
  const r = Object.fromEntries(d.runs.map((x) => [x.mode, x.metrics?.first_hit ?? 99]));
  const best = Math.min(...Object.values(r));
  const expNote = r.expand > r.rewrite ? ` Lưu ý: expansion (#${r.expand}) <b>kém hơn rewrite</b> (#${r.rewrite}) vì từ đồng nghĩa thêm vào kéo embedding sang doc khác chủ đề.` : "";
  if (r.none > best) return { level: "good", title: `Biến đổi query cải thiện: doc đúng từ #${r.none >= 99 ? "–" : r.none} lên #${best}`,
    html: `Query gốc ngắn/viết tắt nên embedding kém đại diện. Rewrite hạng <b>#${r.rewrite}</b>, expansion <b>#${r.expand}</b>, multi-query <b>#${r.multi}</b>: ${r.multi === best ? "multi-query ổn định vì gộp nhiều góc nhìn" : "cách tốt nhất không phải lúc nào cũng là multi-query"}.${expNote}` };
  if (r.expand > r.rewrite) return { level: "warn", title: `Expansion làm tệ đi: #${r.rewrite} (rewrite) → #${r.expand}`,
    html: "Từ đồng nghĩa thêm vào kéo embedding sang các doc không liên quan (ví dụ \"view\", \"inventory\" khớp nhiều chủ đề). Expansion không phải lúc nào cũng có lợi: <b>luôn đo trước khi bật</b>." };
  if (r.none === best && r.none === 1) return { level: "info", title: "Query gốc đã đủ tốt (hạng #1)",
    html: "Đừng biến đổi query bừa bãi: mỗi lần biến đổi tốn thêm latency/chi phí LLM và có thể làm tệ đi." };
  return { level: "info", title: "Các cách cho kết quả tương đương", html: "Với query này biến đổi không tạo khác biệt." };
}

export default {
  id: "transform", num: 4, group: "B", title: "Query transformation", slides: "27",
  tagline: "Query người dùng ngắn, viết tắt, khác từ tài liệu. Biến đổi trước khi tìm kiếm để thu hẹp khoảng cách đó.",
  theory: {
    what: `<p>Biến đổi query <b>trước</b> khi embed/search: <b>Rewrite</b> (viết lại rõ ràng), <b>Expansion</b> (thêm từ đồng nghĩa), <b>Multi-query</b> (nhiều cách diễn đạt rồi gộp), <b>HyDE</b> (sinh câu trả lời giả định rồi embed).</p>`,
    why: `<p>Query ngắn → embedding kém đại diện. Người dùng dùng từ khác với tài liệu (semantic gap). Một query có thể ẩn nhiều intent.</p>`,
    points: ["Mỗi kỹ thuật <b>thêm latency</b> (và chi phí LLM nếu dùng LLM) → cân nhắc theo SLA.",
      "Expansion có thể thêm nhiễu; multi-query thường ổn định hơn.",
      "HyDE cần LLM nên <b>chưa có</b> trong demo offline này (cấu trúc giống hệt: thêm một biến thể)."],
    visual: `<svg viewBox="0 0 760 110" class="theorysvg" role="img" aria-label="Sơ đồ query transformation">
      <rect x="10" y="38" width="110" height="36" rx="6" fill="#fef3c7" stroke="#d97706"/><text x="65" y="61" text-anchor="middle" font-size="13">reset pw</text>
      <text x="135" y="62" font-size="20">→</text>
      <rect x="165" y="8" width="190" height="30" rx="6" fill="#e0ecff" stroke="#3b82f6"/><text x="260" y="28" text-anchor="middle" font-size="12">rewrite: reset password</text>
      <rect x="165" y="42" width="190" height="30" rx="6" fill="#e0ecff" stroke="#3b82f6"/><text x="260" y="62" text-anchor="middle" font-size="12">expand: + credentials, đặt lại…</text>
      <rect x="165" y="76" width="190" height="30" rx="6" fill="#e0ecff" stroke="#3b82f6"/><text x="260" y="96" text-anchor="middle" font-size="12">gốc: reset pw</text>
      <text x="372" y="62" font-size="20">→</text><rect x="400" y="34" width="150" height="44" rx="6" fill="#dcfce7" stroke="#16a34a"/><text x="475" y="60" text-anchor="middle" font-size="13">search mỗi biến thể</text>
      <text x="565" y="62" font-size="20">→</text><rect x="595" y="34" width="150" height="44" rx="6" fill="#ede9fe" stroke="#7c3aed"/><text x="670" y="60" text-anchor="middle" font-size="13">RRF → kết quả</text></svg>`,
  },
  fields: [{ name: "query", label: "Query ngắn/viết tắt", type: "text", wide: true }],
  cases: [
    { id: "pw", tag: "problem", title: "Query viết tắt: \"reset pw\"", slide: "27",
      story: "Query ngắn → embedding kém đại diện; Query Rewriting: \"reset pw\" → \"How to reset account password?\".",
      expect: "Query gốc: doc đúng đứng hạng #2 (sau một mã lỗi); rewrite mở \"pw\" thành \"password\" → hạng #1.",
      values: { query: "reset pw" } },
    { id: "pto", tag: "problem", title: "Viết tắt: \"pto request\" (hạng 7 → 1)", slide: "27",
      story: "Người dùng dùng từ khác với tài liệu → semantic gap.",
      expect: "Query gốc xếp doc đúng hạng #7; sau rewrite (pto → annual leave) lên #1.",
      values: { query: "pto request" } },
    { id: "noise", tag: "error", title: "Expansion thêm nhiễu làm tệ đi", slide: "27",
      story: "Query Expansion thêm từ đồng nghĩa/liên quan; các kỹ thuật thêm latency nên cân nhắc khi SLA nghiêm ngặt.",
      expect: "\"stock chk\": gốc và rewrite đều #1, nhưng expansion kéo doc đúng xuống #4.",
      values: { query: "stock chk" } },
    { id: "multi", tag: "fix", title: "Multi-query ổn định nhất", slide: "27",
      story: "Multi-query: tạo nhiều query rồi gộp kết quả (union, dedupe, rerank).",
      expect: "\"create pr\": gốc #4, rewrite #1, expansion #2; multi-query gộp lại ra #1.",
      values: { query: "create pr" } },
    { id: "free", tag: "free", title: "Tự thử", slide: "27", story: "Tự nhập một query ngắn hoặc viết tắt (pw, inv, pr, pto, sx, 2fa, chk).", expect: "Xem rewrite và expansion áp dụng luật nào.",
      values: { query: "inv approve" } },
  ],
  async run(v) {
    const d = await post("/api/transform", { query: v.query, tenant_id: "company_a", top_k: 5 });
    const r = Object.fromEntries(d.runs.map((x) => [x.mode, x.metrics?.first_hit]));
    const cell = (x) => (d.relevant ? (x === 1 ? { v: 1, level: "good" } : x == null ? { v: "không có", level: "bad" } : x) : "—");
    return {
      steps: steps(d), diagnosis: diagnose(d),
      lessons: ["Biến đổi query là <b>đòn bẩy rẻ</b> khi query người dùng ngắn/viết tắt, nhưng không phải lúc nào cũng có lợi.",
        "Mỗi biến thể thêm một lần embed (hoặc gọi LLM): chi phí và latency tăng theo số biến thể.",
        "Từ điển ở demo này được viết sau khi biết query nên kết quả lạc quan; production phải đo trên query thật."],
      summary: { label: v.query, facts: { "Gốc": cell(r.none), "Rewrite": cell(r.rewrite), "Expansion": cell(r.expand), "Multi-query": cell(r.multi) } },
    };
  },
};
