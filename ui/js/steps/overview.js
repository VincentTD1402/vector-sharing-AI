// Trang Tổng quan: bức tranh cả pipeline (offline + online), mỗi ô dẫn tới bài học tương ứng, và cách học.
import { esc } from "../util.js";

const NODES = [
  { lane: "offline", id: "chunking", n: 1, title: "Chunking", what: "Chia tài liệu dài thành chunk", fail: "Đáp án bị cắt đôi ở ranh giới; chunk quá nhỏ/lớn" },
  { lane: "offline", id: "embedding", n: 2, title: "Embedding", what: "Biến chunk thành vector 384 chiều", fail: "Dot product bị đánh lừa; quên prefix; đa ngôn ngữ kém hơn" },
  { lane: "offline", id: "index", n: 3, title: "Index & Vector DB", what: "Lưu vector + metadata, đánh chỉ mục ANN", fail: "Exact không scale; HNSW + filter trả thiếu; vector(2048) vượt giới hạn" },
  { lane: "online", id: "transform", n: 4, title: "Query transformation", what: "Viết lại / mở rộng query", fail: "Query viết tắt; expansion thêm nhiễu" },
  { lane: "online", id: "retrieval", n: 5, title: "Retrieval & Filter", what: "BM25 + vector + RRF, kèm filter tenant", fail: "Mã lỗi vector sai; rò rỉ tenant; post-filter thiếu" },
  { lane: "online", id: "rerank", n: 6, title: "Reranking", what: "Cross-encoder chấm lại top-N", fail: "Pool nhỏ không cứu được; có khi làm tệ đi; tốn latency" },
];
const THEN = [
  { id: "evaluate", n: 7, title: "Đánh giá & Debug", what: "Recall / Precision / MRR; debug 8 bước", fail: "Chỉ nhìn câu trả lời LLM mà không đo retrieval" },
  { id: "pipeline", n: 8, title: "Pipeline hoàn chỉnh", what: "Cả luồng + log + cache + latency", fail: "Không có log; ngưỡng cache sai; rerank chiếm hết latency" },
];

const card = (x) => `<a class="flownode" href="#/${x.id}"><span class="step">${x.n}</span><div><b>${esc(x.title)}</b><span class="what">${esc(x.what)}</span>
  <span class="fail"><i>Hay hỏng:</i> ${esc(x.fail)}</span></div></a>`;

export default {
  id: "overview", num: "★", group: "0", title: "Tổng quan & cách học", slides: "1-35",
  tagline: "Từ \"đưa text vào vector database\" đến một hệ thống retrieval chạy được. Đi theo thứ tự các bài, mỗi bài đi từ lý thuyết tới lỗi thật trong slide.",
  mount(root) {
    root.innerHTML = `<header class="stephead"><span class="step">★</span><div><h1>${esc(this.title)}</h1><p>${esc(this.tagline)}</p></div></header>
      <section class="card"><h2>Một bài học gồm 4 phần</h2>
        <div class="loop"><div><span class="step">1</span><b>Lý thuyết</b><span>Khái niệm, vì sao quan trọng, điều cần nhớ</span></div><i>→</i>
          <div><span class="step">2</span><b>Tình huống trong slide</b><span>Một vấn đề / lỗi / giải pháp thật, bấm để nạp</span></div><i>→</i>
          <div><span class="step">3</span><b>Chạy từng bước</b><span>Thanh luồng + từng bước mở ra với dữ liệu thật và thời gian thật</span></div><i>→</i>
          <div><span class="step">4</span><b>Chẩn đoán &amp; sửa</b><span>Vì sao hỏng, thử cách sửa, so sánh trước/sau</span></div></div></section>
      <section class="card"><h2>Toàn bộ pipeline</h2>
        <div class="lane"><div class="lanehead offline">A · Offline: chuẩn bị dữ liệu (làm một lần)</div><div class="lanenodes">${NODES.filter((x) => x.lane === "offline").map(card).join('<i class="flowarrow">→</i>')}</div></div>
        <div class="lane"><div class="lanehead online">B · Online: mỗi query của người dùng</div><div class="lanenodes">${NODES.filter((x) => x.lane === "online").map(card).join('<i class="flowarrow">→</i>')}</div></div>
        <div class="lane"><div class="lanehead result">C · Đo lường và vận hành</div><div class="lanenodes">${THEN.map(card).join('<i class="flowarrow">→</i>')}</div></div>
        <p class="hint">Mỗi ô: bước làm gì và <b>kiểu lỗi hay gặp</b> mà bài học tương ứng tái hiện bằng dữ liệu thật. Bấm một ô để vào bài.</p></section>
      <section class="card"><h2>Thông điệp xuyên suốt</h2><ul class="prose">
        <li><b>Retrieval Engineering không phải \"đưa text vào vector DB\"</b>: mỗi giai đoạn là một quyết định kỹ thuật có đánh đổi.</li>
        <li>Lỗi thường <b>âm thầm</b>: kết quả trông có vẻ đúng nhưng đáp án đã mất (chunking), tenant khác lọt vào (filter), doc đúng ở hạng 6 (ranking).</li>
        <li><b>Đo, đừng đoán</b>: bài 7 cho cách đo và cách debug từng tầng.</li></ul></section>`;
  },
};
