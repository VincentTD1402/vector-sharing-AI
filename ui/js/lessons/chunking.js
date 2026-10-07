// Bài 1 - Chunking: vì sao đáp án có thể "biến mất" ở ranh giới chunk, và cách chữa.
import { post } from "../api.js";
import { callout, kv, ms, pill, rankPill, table } from "../lesson/render.js";
import { $, esc, fmt } from "../util.js";
import { bar } from "../lesson/render.js";

const MANUALS = [
  { value: "manual:production_planning_handbook_en", label: "production_planning_handbook_en (EN)" },
  { value: "manual:procurement_policy_vi", label: "procurement_policy_vi (VI)" },
  { value: "manual:warehouse_operations_guide_en", label: "warehouse_operations_guide_en (EN)" },
];

/** Tô màu văn bản theo chunk; vùng sọc là overlap; nền vàng là đáp án; đường đỏ là ranh giới cắt qua đáp án. */
function paint(text, chunks, answer) {
  const cuts = new Set([0, text.length, ...chunks.flatMap((c) => [c.start, c.end])]);
  if (answer.start != null) { cuts.add(answer.start); cuts.add(answer.end); }
  const pts = [...cuts].sort((a, b) => a - b);
  let html = "";
  for (let k = 0; k < pts.length - 1; k++) {
    const [a, b] = [pts[k], pts[k + 1]];
    const cover = chunks.filter((c) => c.start <= a && c.end >= b);
    const starts = chunks.filter((c) => c.start === a).map((c) => `<sup class="mark">#${c.i}</sup>`).join("");
    const inAns = answer.start != null && a >= answer.start && b <= answer.end;
    const cls = [cover.length > 1 ? "ov" : cover.length ? `c${cover[0].i % 6}` : "gap", inAns ? "ans" : ""].join(" ");
    const cut = answer.boundaries_inside?.includes(a) ? `<span class="cutline" title="ranh giới chunk cắt qua đáp án"></span>` : "";
    html += `${cut}${starts}<span class="${cls}" title="${cover.length ? "chunk " + cover.map((c) => c.i).join(" & ") : "không thuộc chunk nào"}">${esc(text.slice(a, b))}</span>`;
  }
  return html;
}

function steps(d) {
  const a = d.answer, c = d.config;
  const status = a.status === "intact" ? { v: "nguyên vẹn", level: "good" } : a.status === "split" ? { v: "BỊ CẮT ĐÔI", level: "bad" } : { v: "—", level: "" };
  const ranking = d.ranking.slice(0, 6);
  const maxScore = Math.max(...ranking.map((r) => r.score), 0.001);
  return [
    { title: "Nạp tài liệu", ms: null,
      explain: "Điểm xuất phát: một tài liệu dài. Embedding model chỉ nhận tối đa 512 token mỗi lần và một vector duy nhất không thể mô tả chi tiết cả tài liệu, nên phải chia nhỏ.",
      summary: `${d.doc.tokens} token · ${d.doc.chars} ký tự`,
      html: `${kv([["Tài liệu", esc(d.doc.source.replace("manual:", ""))], ["Số token", `<b>${d.doc.tokens}</b> (giới hạn model: 512)`], ["Ký tự", d.doc.chars]])}
        <div class="chunk-text small">${esc(d.doc.head)}…</div>` },
    { title: "Tokenize", ms: ms(d.ms.tokenize),
      explain: "Model không đọc chữ mà đọc <b>token</b> (mảnh từ). Chunk size được tính theo token, không phải theo ký tự hay từ.",
      summary: `${d.doc.tokens} token`,
      html: `<div class="tokens">${d.tokens_preview.map((t) => `<span class="token">${esc(t)}</span>`).join("")}<span class="token">…</span></div>
        <p class="hint">Tiếng Việt thường tốn nhiều token hơn tiếng Anh cho cùng độ dài chữ.</p>` },
    { title: "Cắt chunk", ms: ms(d.ms.chunk),
      explain: `Chiến lược <b>${esc(c.strategy)}</b>, size <b>${c.size}</b> token${c.strategy === "fixed" ? `, overlap <b>${c.overlap}</b>` : ""}. Mỗi màu là một chunk; vùng sọc là overlap; <mark class="ans">nền vàng</mark> là đáp án cần tìm.`,
      summary: `${d.chunks.length} chunk · TB ${d.avg_tokens} token`,
      html: `<div class="chunk-text">${paint(d.text, d.chunks, a)}</div>
        ${a.status === "split" ? callout("bad", `Ranh giới chunk (đường đỏ) cắt <b>xuyên qua đáp án</b> «${esc(a.text)}» → đáp án nằm ở ${a.touching.map((i) => `chunk #${i}`).join(" và ")}, không chunk nào chứa trọn.`)
          : a.status === "intact" ? callout("good", `Đáp án nằm trọn trong ${a.contained_in.length ? a.contained_in.map((i) => `chunk #${i}`).join(", ") : "một chunk"}.`)
          : a.status === "missing" ? callout("warn", "Chuỗi đáp án không có trong tài liệu này.") : ""}`,
      warn: a.status === "split" ? "Đây chính là lỗi trong slide: <i>\"Overlap giúp tránh mất thông tin ở ranh giới chunk\"</i>." : null },
    { title: "Embed từng chunk", ms: ms(d.ms.embed_chunks),
      explain: `Mỗi chunk được đưa qua model → <b>1 vector ${d.dim} chiều</b>. Đây là bước tốn thời gian ở giai đoạn index (nên làm theo batch).`,
      summary: `${d.chunks.length} vector × ${d.dim} chiều`,
      html: `<div class="vecstack">${d.chunks.slice(0, 8).map((ch) => `<div class="vecrow"><span class="vid">#${ch.i}</span><i class="vbar" style="width:${(ch.tokens / Math.max(...d.chunks.map((x) => x.tokens))) * 100}%"></i><span class="hint">${ch.tokens} token → [${d.dim} số]</span></div>`).join("")}${d.chunks.length > 8 ? `<p class="hint">… và ${d.chunks.length - 8} chunk nữa</p>` : ""}</div>` },
    { title: "Embed câu hỏi và xếp hạng", ms: ms(d.ms.embed_query),
      explain: "Câu hỏi cũng thành vector; chunk nào có cosine cao nhất được xếp trên. ✓ = chứa trọn đáp án, ◐ = chỉ chứa một phần.",
      summary: d.answer_rank ? `chunk đúng hạng #${d.answer_rank}` : "không chunk nào chứa trọn đáp án",
      html: table([{ t: "Hạng", num: true }, { t: "Chunk" }, { t: "Cosine" }, { t: "Đáp án" }],
        ranking.map((r) => [r.rank, `#${r.i} <span class="hint">(${d.chunks[r.i].tokens} token)</span>`,
          `<div style="min-width:120px">${bar(r.score, maxScore)}<span class="hint">${fmt(r.score, 3)}</span></div>`,
          r.has_full ? "<b class='ok'>✓ trọn vẹn</b>" : r.has_part ? "<b style='color:var(--warn)'>◐ một phần</b>" : ""]),
        (i) => (ranking[i].has_full ? "rel" : "")) },
    { title: "Kiểm tra đáp án", ms: null,
      explain: "Chốt: LLM sẽ nhận các chunk top-K. Nếu không chunk nào chứa trọn đáp án thì LLM không thể trả lời đúng, dù retrieval \"đúng chủ đề\".",
      summary: a.status === "split" ? "đáp án bị cắt đôi" : d.answer_rank ? `đáp án có, hạng #${d.answer_rank}` : "—",
      html: kv([["Trạng thái đáp án", pill(status.v, status.level)], ["Hạng chunk chứa đáp án", rankPill(d.answer_rank)],
        ["Token gửi LLM (top-3)", `<b>${d.ctx_tokens_top3}</b>`]]) },
  ];
}

function diagnose(d) {
  const a = d.answer;
  if (a.status === "split") return { level: "bad", title: "Đáp án bị cắt đôi ở ranh giới chunk",
    html: `Không chunk nào chứa trọn «${esc(a.text)}». Retrieval có thể vẫn trả về chunk "đúng chủ đề" nhưng LLM chỉ thấy một nửa đáp án → trả lời sai hoặc bịa. Hãy thử cách sửa bên dưới.` };
  if (a.status === "intact" && d.answer_rank === 1 && d.ctx_tokens_top3 > 400) return { level: "warn", title: `Tìm đúng nhưng phải gửi ${d.ctx_tokens_top3} token cho LLM`,
    html: `Chunk lớn (TB ${d.avg_tokens} token) kéo theo nhiều ý không liên quan vào context: tốn token/tiền và loãng ngữ cảnh dù chỉ cần một con số. Giảm size để thấy chi phí giảm mà đáp án vẫn nguyên vẹn.` };
  if (a.status === "intact" && d.answer_rank === 1) return { level: "good", title: "Đáp án nguyên vẹn và chunk đúng đứng đầu",
    html: `Cấu hình này làm đúng việc: chunk #${a.contained_in[0] ?? "?"} chứa trọn đáp án và được xếp hạng 1. Chi phí: <b>${d.ctx_tokens_top3}</b> token gửi cho LLM (top-3).` };
  if (a.status === "intact") return { level: "warn", title: `Đáp án còn nguyên nhưng chunk đúng chỉ đứng hạng #${d.answer_rank}`,
    html: "Chunk chứa đáp án chưa lên đầu: có thể do chunk quá nhỏ (mất ngữ cảnh) hoặc quá lớn (nhiều ý lẫn nhau). Đây là lúc cần reranker (bài 6) và bộ đánh giá (bài 7)." };
  return { level: "info", title: "Không tìm thấy chuỗi đáp án trong tài liệu", html: "Kiểm tra lại câu hỏi/đáp án hoặc chọn tài liệu khác." };
}

export default {
  id: "chunking", num: 1, group: "A", title: "Chunking", slides: "6, 7, 23, 25",
  tagline: "Chia tài liệu dài thành các chunk. Quyết định đầu tiên, và sai ở đây thì mọi bước sau đều hỏng.",
  theory: {
    what: `<p>Embedding model chỉ nhận tối đa ~512 token và một vector khó mô tả cả tài liệu dài. Vì vậy ta <b>chia tài liệu thành các chunk</b>, embed từng chunk, và retrieve ở mức chunk.</p>
      <p>Chiến lược: <b>Fixed-size</b> (cắt theo số token), <b>Sentence</b> (gom câu), <b>Recursive</b> (đoạn → dòng → câu → từ).</p>`,
    why: `<p>Chunk quyết định <b>retrieval tìm được gì và LLM được thấy gì</b>.</p>
      <p>Quá nhỏ: mất ngữ cảnh, nhiều chunk hơn, kém chính xác. Quá lớn: embedding kém chất lượng, lẫn nhiều ý, tốn context của LLM. Cắt giữa đáp án: đáp án "biến mất".</p>`,
    points: ["<b>Overlap</b> giữ lại phần cuối chunk trước ở đầu chunk sau để không mất thông tin ở ranh giới.",
      "Lưu <b>metadata</b> (document_id, chunk_index, page) ngay lúc chunking.",
      "Không có size tốt nhất tuyệt đối: <b>đo</b> Recall@K trên dữ liệu của bạn."],
    visual: `<svg viewBox="0 0 900 100" class="theorysvg" role="img" aria-label="Minh họa overlap">
      <rect x="10" y="10" width="300" height="30" rx="5" fill="#dbeafe" stroke="#3b82f6"/><text x="160" y="30" text-anchor="middle" font-size="13">chunk 0</text>
      <rect x="270" y="54" width="300" height="30" rx="5" fill="#dcfce7" stroke="#16a34a"/><text x="420" y="74" text-anchor="middle" font-size="13">chunk 1 (bắt đầu lùi lại = overlap)</text>
      <rect x="270" y="10" width="40" height="74" fill="#fecaca" fill-opacity=".55"/><text x="290" y="97" text-anchor="middle" font-size="11" fill="#b91c1c">overlap</text>
      <text x="600" y="30" font-size="13" fill="#b91c1c">Không overlap: đáp án ở ranh giới → mất</text>
      <text x="600" y="74" font-size="13" fill="#15803d">Có overlap: đáp án nằm trọn trong 1 chunk</text></svg>`,
  },
  fields: [
    { name: "source", label: "Tài liệu", type: "select", options: MANUALS },
    { name: "strategy", label: "Chiến lược", type: "select", options: [{ value: "fixed", label: "Fixed-size (token)" }, { value: "sentence", label: "Sentence" }, { value: "recursive", label: "Recursive" }] },
    { name: "size", label: "Chunk size", type: "range", min: 16, max: 500, step: 4, unit: "token" },
    { name: "overlap", label: "Overlap (chỉ fixed)", type: "range", min: 0, max: 60, step: 4, unit: "token" },
    { name: "query", label: "Câu hỏi", type: "text", wide: true },
    { name: "answer", label: "Đáp án (chuỗi cần nằm trọn trong một chunk)", type: "text", wide: true },
  ],
  cases: [
    { id: "split-en", tag: "problem", title: "Đáp án bị cắt đôi ở ranh giới", slide: "6",
      story: "Chunk quá nhỏ / không overlap thì mất thông tin ở ranh giới chunk.",
      expect: "Ranh giới đỏ cắt ngang «1,200 units»; không chunk nào chứa trọn đáp án (hạng: không có).",
      values: { source: "manual:production_planning_handbook_en", strategy: "fixed", size: 40, overlap: 0, query: "How many units per shift does Line 3 produce?", answer: "1,200 units" },
      fixes: [{ label: "Thêm overlap 16 token", patch: { overlap: 16 } }, { label: "Đổi sang Recursive (tách theo đoạn/câu)", patch: { strategy: "recursive", overlap: 0 } }] },
    { id: "split-vi", tag: "problem", title: "Cùng lỗi với tiếng Việt", slide: "6",
      story: "Lỗi ranh giới xảy ra với mọi ngôn ngữ; tiếng Việt tốn nhiều token hơn nên dễ bị cắt hơn.",
      expect: "«tối đa là 30%» bị cắt đôi. Sau khi thêm overlap, đáp án nguyên vẹn nhưng chunk đúng có thể chưa đứng hạng 1.",
      values: { source: "manual:procurement_policy_vi", strategy: "fixed", size: 40, overlap: 0, query: "Đặt cọc tối đa bao nhiêu phần trăm giá trị đơn hàng?", answer: "tối đa là 30%" },
      fixes: [{ label: "Thêm overlap 16 token", patch: { overlap: 16 } }, { label: "Tăng size lên 128", patch: { size: 128, overlap: 0 } }] },
    { id: "too-small", tag: "problem", title: "Chunk quá nhỏ: mất ngữ cảnh", slide: "6",
      story: "Chunk quá nhỏ → mất context, nhiều chunk hơn, retrieval kém precise.",
      expect: "Với size 24, «Z = 1.65» bị cắt đôi và có tới 21 chunk. Tăng size thì đáp án nguyên vẹn.",
      values: { source: "manual:warehouse_operations_guide_en", strategy: "fixed", size: 24, overlap: 0, query: "Which Z value is used for a 95% service level?", answer: "Z = 1.65" },
      fixes: [{ label: "Tăng size lên 128", patch: { size: 128 } }] },
    { id: "too-big", tag: "problem", title: "Chunk quá lớn: tốn context của LLM", slide: "6, 25",
      story: "Chunk quá lớn → embedding kém chính xác và tốn context; Recall tốt nhưng precision và chi phí xấu đi.",
      expect: "Đáp án vẫn tìm đúng (hạng 1) nhưng phải gửi ~555 token cho LLM chỉ để có một con số. Giảm size để thấy chi phí giảm.",
      values: { source: "manual:production_planning_handbook_en", strategy: "fixed", size: 200, overlap: 0, query: "How long does a changeover take on Line 1?", answer: "45 minutes" },
      fixes: [{ label: "Giảm size xuống 64", patch: { size: 64 } }] },
    { id: "recursive", tag: "fix", title: "Recursive/Sentence: không cắt giữa câu", slide: "6",
      story: "Recursive tách theo đoạn/câu trước nên hiếm khi cắt giữa đáp án; ví dụ code dùng RecursiveCharacterTextSplitter.",
      expect: "Cùng câu hỏi ở tình huống đầu nhưng đáp án nguyên vẹn dù size chỉ 40 token.",
      values: { source: "manual:production_planning_handbook_en", strategy: "recursive", size: 40, overlap: 0, query: "How many units per shift does Line 3 produce?", answer: "1,200 units" },
      fixes: [{ label: "Thử lại với Fixed-size, không overlap", patch: { strategy: "fixed" } }] },
    { id: "free", tag: "free", title: "Tự thử", slide: "6",
      story: "Tự chọn tài liệu, chiến lược, size, overlap, câu hỏi và đáp án.",
      expect: "Thử tìm cấu hình làm đáp án của bạn bị cắt đôi.",
      values: { source: "manual:warehouse_operations_guide_en", strategy: "fixed", size: 64, overlap: 0, query: "What temperature range is the cold room kept at?", answer: "between 2 and 8" } },
  ],
  async run(v) {
    const d = await post("/api/lesson/chunk", { source: v.source, strategy: v.strategy, size: v.size, overlap: v.strategy === "fixed" ? v.overlap : 0, query: v.query, answer: v.answer });
    const a = d.answer;
    return {
      steps: steps(d), diagnosis: diagnose(d),
      lessons: ["Chunking thất bại <b>âm thầm</b>: retrieval vẫn trả kết quả \"có vẻ đúng\", chỉ khi đọc nội dung mới thấy đáp án bị mất.",
        "Overlap và recursive/sentence là hai cách phổ biến để bảo vệ ranh giới; cái giá là nhiều chunk hơn / token lặp lại.",
        "Luôn đo Recall@K và <b>token gửi cho LLM</b> trên bộ câu hỏi thật thay vì chọn size theo cảm giác."],
      summary: { label: `${v.strategy} ${v.size} tok, overlap ${v.strategy === "fixed" ? v.overlap : 0}`, facts: {
        "Số chunk": d.chunks.length, "Token/chunk (TB)": d.avg_tokens,
        "Đáp án": a.status === "intact" ? { v: "nguyên vẹn", level: "good" } : a.status === "split" ? { v: "bị cắt đôi", level: "bad" } : "—",
        "Hạng chunk đúng": d.answer_rank ?? { v: "không có", level: "bad" }, "Token gửi LLM (top-3)": d.ctx_tokens_top3 } },
    };
  },
};
