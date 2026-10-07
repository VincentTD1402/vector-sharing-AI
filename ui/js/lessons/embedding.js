// Bài 2 - Embedding: từ text đến vector và cách đo "giống nhau" (cosine / dot / euclid).
import { post } from "../api.js";
import { dimsChart } from "../components.js";
import { callout, kv, ms, pill, table } from "../lesson/render.js";
import { mountMap } from "../space-map.js";
import { esc, fmt } from "../util.js";

const TEXTS = ["How to reset your password", "I remembered my password", "I never want to see my password again", "Create production order"].join("\n");

function steps(d, alt, ctx) {
  const t = d.trace;
  const rank = (r, key) => `${r[key]}${r[key] !== r.rank_cosine ? "" : ""}`;
  return [
    { title: "Tokenize", ms: ms(t.ms.tokenize),
      explain: `Model không đọc chữ mà đọc <b>token</b> (mảnh từ → số nguyên). ${d.use_prefix ? "Chú ý prefix <code>query: </code> được thêm trước câu: e5 được huấn luyện với quy ước này." : "Lần chạy này <b>không có prefix</b>, trái với quy ước huấn luyện của e5."}`,
      summary: `${t.tokens.length} token`,
      html: `<div class="tokens">${t.tokens.map((x, i) => `<span class="token" title="id ${t.token_ids[i]}">${esc(x)}</span>`).join("")}</div>
        <p class="hint">Rê chuột lên token để thấy id. Từ lạ/hiếm bị cắt thành nhiều mảnh.</p>` },
    { title: "Transformer", ms: ms(t.ms.transformer),
      explain: "Mỗi token đi qua các lớp transformer (attention) và trở thành <b>một vector ngữ cảnh</b>: cùng chữ \"password\" ở câu khác nhau sẽ có vector khác nhau.",
      summary: `${t.hidden_shape[0]} × ${t.hidden_shape[1]}`,
      html: kv([["Đầu vào", `${t.tokens.length} token`], ["Đầu ra", `ma trận <b>${t.hidden_shape[0]} × ${t.hidden_shape[1]}</b> (mỗi hàng là vector của một token)`]]) },
    { title: "Mean pooling", ms: ms(t.ms.pooling),
      explain: "Câu có nhiều token nhưng ta cần <b>một vector duy nhất</b> đại diện cho cả câu: lấy trung bình các vector token.",
      summary: `1 × ${t.dim}`,
      html: kv([["Trước", `${t.hidden_shape[0]} × ${t.hidden_shape[1]}`], ["Sau pooling", `<b>1 × ${t.dim}</b>`], ["Độ dài ‖pooled‖", t.pooled_norm]]) },
    { title: "Normalize", ms: null,
      explain: "Chuẩn hóa vector về <b>độ dài 1</b>. Khi đó cosine = dot product và độ lớn không còn ảnh hưởng. Kiểm tra: kết quả khớp với hàm <code>model.encode()</code> chính thức.",
      summary: `‖v‖ = ${t.norm} ${t.matches_encode ? "· khớp encode ✓" : "· LỆCH ✗"}`,
      html: `${dimsChart(t.dims)}<p class="hint">48 chiều đầu của ${t.dim}. Từng chiều không đọc được ý nghĩa; chỉ <b>khoảng cách giữa các vector</b> có nghĩa.</p>
        ${t.matches_encode ? callout("good", "Tự tính từng bước cho ra đúng vector của <code>model.encode()</code>.") : callout("bad", "Lệch so với model.encode().")}` },
    { title: "So sánh với các văn bản", ms: null,
      explain: `Embed ${d.rows.length} văn bản rồi đo độ giống với query bằng 3 metric. Ô <span class="diffcell">tô vàng</span> = thứ hạng khác cosine.${d.scale_last !== 1 ? ` Văn bản cuối đã bị <b>phóng to ×${d.scale_last}</b>.` : ""}`,
      summary: `cosine #1: ${esc(d.rows.find((r) => r.rank_cosine === 1)?.text.slice(0, 22) ?? "")}…`,
      html: table([{ t: "#", num: true }, { t: "Văn bản" }, { t: "‖v‖", num: true }, { t: "Cosine", num: true }, { t: "Dot", num: true }, { t: "Euclid", num: true }, ...(alt ? [{ t: `Cosine (${d.use_prefix ? "không" : "có"} prefix)`, num: true }] : [])],
        d.rows.map((r, i) => [i + 1, `${esc(r.text)} ${pill(r.language)}`, r.norm,
          `<b>${fmt(r.cosine, 3)}</b> <span class="hint">#${r.rank_cosine}</span>`,
          `<span class="${r.rank_dot !== r.rank_cosine ? "diffcell" : ""}">${fmt(r.dot, 3)} <span class="hint">#${r.rank_dot}</span></span>`,
          `<span class="${r.rank_euclid !== r.rank_cosine ? "diffcell" : ""}">${fmt(r.euclid, 3)} <span class="hint">#${r.rank_euclid}</span></span>`,
          ...(alt ? [`${fmt(alt.rows[i].cosine, 3)} <span class="hint">#${alt.rows[i].rank_cosine}</span>`] : [])]),
        (i) => (d.rows[i].rank_cosine === 1 ? "rel" : "")) +
        `<p class="hint">e5 nén điểm vào dải hẹp ~0.7–1.0: đừng bê ngưỡng cosine từ model khác sang.</p>` },
    { title: "Vị trí trong không gian vector", ms: null,
      explain: "Chiếu 384 chiều xuống 2D (PCA) để nhìn: văn bản của bạn (vòng số) và query (sao đen) trên nền toàn bộ corpus. Chỉ giữ ~20% phương sai nên khoảng cách trên hình chỉ mang tính minh họa.",
      summary: "bản đồ 2D",
      html: `<div class="mapbox"></div>`,
      after: (el) => mountMap(el.querySelector(".mapbox"), ctx.space).update({ queryPoint: d.points.query, extra: d.points.texts.map((p, i) => ({ ...p, label: String(i + 1) })) }) },
  ];
}

function diagnose(id, d) {
  const by = (needle) => d.rows.find((r) => r.text.toLowerCase().includes(needle));
  if (id === "statistics") {
    const rem = by("remembered"), reset = by("reset your password");
    if (rem && reset) return rem.rank_cosine < reset.rank_cosine
      ? { level: "warn", title: "\"I remembered my password\" gần query hơn cả \"How to reset your password\"",
          html: `cosine ${fmt(rem.cosine, 3)} (#${rem.rank_cosine}) so với ${fmt(reset.cosine, 3)} (#${reset.rank_cosine}). Câu <b>ngược nghĩa</b> vẫn rất gần vì cùng chủ đề/từ vựng. Embedding nắm bắt <b>mẫu thống kê</b> từ dữ liệu huấn luyện: nó đo "cùng ngữ cảnh", không đo đúng/sai.` }
      : { level: "good", title: "Câu đồng nghĩa gần hơn câu ngược nghĩa", html: "Lần này thứ hạng như mong đợi." };
  }
  if (id === "dot") {
    const big = d.rows[d.rows.length - 1];
    return d.scale_last > 1 && big.rank_dot === 1 && big.rank_cosine > 1
      ? { level: "bad", title: `Dot product xếp văn bản KHÔNG liên quan lên #1 chỉ vì vector dài gấp ${d.scale_last} lần`,
          html: `«${esc(big.text)}» có cosine #${big.rank_cosine} (đúng) nhưng dot #${big.rank_dot} và euclid #${big.rank_euclid}. Cosine bỏ qua độ lớn nên không bị đánh lừa. <b>Metric phải khớp cách model được huấn luyện</b> (e5/BGE: cosine).` }
      : { level: "info", title: "Khi vector đã normalize, 3 metric cho cùng thứ hạng", html: "Hãy tăng \"phóng to\" để thấy chúng tách nhau." };
  }
  if (id === "cross") {
    const vi = d.rows.find((r) => r.language === "vi" && r.text.toLowerCase().includes("mật khẩu")), en = by("reset your password");
    if (vi && en) return { level: en.cosine > vi.cosine ? "warn" : "info", title: `Cùng ý nghĩa nhưng tiếng Anh gần hơn tiếng Việt (cosine ${fmt(en.cosine, 3)} vs ${fmt(vi.cosine, 3)})`,
      html: "Model đa ngôn ngữ nhỏ vẫn có <b>thiên hướng theo ngôn ngữ</b>: truy vấn chéo ngôn ngữ kém hơn cùng ngôn ngữ (bản đồ PCA cũng tách hai vùng vi/en). Đây là lý do doc tiếng Việt hay bị xếp thấp khi hỏi bằng tiếng Anh." };
  }
  if (id === "prefix") return { level: "info", title: d.use_prefix ? "Có prefix: đúng quy ước của e5" : "Không prefix: sai quy ước huấn luyện của e5",
    html: "Cột cuối của bảng cho thấy cosine thay đổi khi bỏ/thêm prefix. Mức ảnh hưởng tới retrieval tùy dữ liệu, nhưng đây là <b>tham số phải nhất quán giữa lúc index và lúc query</b>; mỗi model một quy ước, hãy đọc model card." };
  return { level: "info", title: "Xem bảng so sánh ở trên", html: "Thử thêm văn bản của bạn và so sánh cosine / dot / euclid." };
}

export default {
  id: "embedding", num: 2, group: "A", title: "Embedding", slides: "3, 5, 8, 10",
  tagline: "Biến text thành vector rồi đo độ giống nhau. Xem từng bước bên trong model và những cạm bẫy khi so sánh.",
  theory: {
    what: `<p><b>Embedding</b> là biểu diễn số được <i>học</i> của một văn bản trong không gian vector liên tục, sao cho các văn bản có đặc điểm tương tự nằm gần nhau.</p>
      <p>Quy trình: Tokenizer → Transformer → Pooling → Normalize → vector 384 chiều.</p>`,
    why: `<p>Khi mọi thứ là vector, bài toán tìm kiếm trở thành <b>tìm điểm gần nhất</b>. Nhưng "gần" phụ thuộc cả <b>model</b> lẫn <b>metric</b>.</p>`,
    points: ["Embedding nắm bắt <b>mẫu thống kê</b>, không phải \"hiểu ngôn ngữ\".",
      "Metric (cosine/dot/euclid) phải khớp cách model được huấn luyện.",
      "<b>Không trộn vector của hai model khác nhau</b>: đổi model = index lại toàn bộ."],
    visual: `<svg viewBox="0 0 760 90" class="theorysvg" role="img" aria-label="Pipeline embedding">
      <rect x="8" y="25" width="120" height="40" rx="6" fill="#fef3c7" stroke="#d97706"/><text x="68" y="50" text-anchor="middle" font-size="13">"reset password"</text>
      <text x="138" y="50" font-size="20">→</text><rect x="160" y="25" width="100" height="40" rx="6" fill="#e0ecff" stroke="#3b82f6"/><text x="210" y="50" text-anchor="middle" font-size="13">Tokenizer</text>
      <text x="270" y="50" font-size="20">→</text><rect x="292" y="25" width="110" height="40" rx="6" fill="#e0ecff" stroke="#3b82f6"/><text x="347" y="50" text-anchor="middle" font-size="13">Transformer</text>
      <text x="412" y="50" font-size="20">→</text><rect x="434" y="25" width="90" height="40" rx="6" fill="#e0ecff" stroke="#3b82f6"/><text x="479" y="50" text-anchor="middle" font-size="13">Pooling</text>
      <text x="534" y="50" font-size="20">→</text><rect x="556" y="25" width="190" height="40" rx="6" fill="#dcfce7" stroke="#16a34a"/><text x="651" y="50" text-anchor="middle" font-size="13">[0.12, −0.44, 0.81, …] ×384</text></svg>`,
  },
  fields: [
    { name: "query", label: "Query", type: "text", wide: true },
    { name: "texts", label: "Các văn bản để so sánh (mỗi dòng một văn bản, tối đa 6)", type: "textarea", rows: 5, wide: true },
    { name: "use_prefix", label: "Dùng prefix \"query: \" / \"passage: \" của e5", type: "check" },
    { name: "scale_last", label: "Phóng to vector của văn bản CUỐI", type: "range", min: 1, max: 10, step: 1, unit: "×" },
  ],
  cases: [
    { id: "statistics", tag: "observe", title: "Embedding không \"hiểu\", nó đo thống kê", slide: "5",
      story: "Không nói \"Embedding hiểu ngôn ngữ\"; nói \"Embedding nắm bắt các mẫu thống kê từ dữ liệu huấn luyện\".",
      expect: "\"I remembered my password\" (ngược nghĩa) lại gần \"I forgot my password\" hơn cả \"How to reset your password\".",
      values: { query: "I forgot my password", texts: TEXTS, use_prefix: true, scale_last: 1 } },
    { id: "dot", tag: "problem", title: "Dot product bị đánh lừa bởi độ lớn vector", slide: "10",
      story: "Dot product bị ảnh hưởng bởi cả hướng VÀ độ lớn; cosine chỉ đo góc. Metric phải khớp cách model được huấn luyện.",
      expect: "Văn bản cuối (không liên quan) bị phóng to ×5: dot và euclid xếp nó #1, cosine vẫn xếp #4.",
      values: { query: "I forgot my password", texts: TEXTS, use_prefix: true, scale_last: 5 },
      fixes: [{ label: "Bỏ phóng to (×1)", patch: { scale_last: 1 } }] },
    { id: "cross", tag: "problem", title: "Query tiếng Anh, tài liệu tiếng Việt", slide: "9",
      story: "Multilingual nếu cần đa ngôn ngữ, nhưng có thể kém hơn monolingual.",
      expect: "Hai văn bản cùng ý nghĩa nhưng bản tiếng Anh có cosine cao hơn bản tiếng Việt.",
      values: { query: "I forgot my password", texts: ["Đặt lại mật khẩu: chọn 'Quên mật khẩu', nhập email công ty.", "How to reset your password: click 'Forgot password'.", "Tạo lệnh sản xuất mới"].join("\n"), use_prefix: true, scale_last: 1 } },
    { id: "prefix", tag: "problem", title: "Quên prefix của e5", slide: "8",
      story: "Mỗi embedding model có quy ước riêng; thay đổi cách dùng = kết quả khác. Phải nhất quán lúc index và lúc query.",
      expect: "Bỏ prefix: điểm cosine của mọi văn bản thay đổi (cột cuối so với lần có prefix).",
      values: { query: "I forgot my password", texts: TEXTS, use_prefix: false, scale_last: 1 },
      fixes: [{ label: "Bật prefix đúng quy ước", patch: { use_prefix: true } }] },
    { id: "free", tag: "free", title: "Tự thử", slide: "5, 10", story: "Tự nhập query và các văn bản.", expect: "So sánh cosine, dot, euclid.",
      values: { query: "quên mật khẩu", texts: ["Đặt lại mật khẩu", "Tạo lệnh sản xuất", "How to reset your password"].join("\n"), use_prefix: true, scale_last: 1 } },
  ],
  async run(v, ctx, current) {
    const texts = v.texts.split("\n").map((t) => t.trim()).filter(Boolean).slice(0, 6);
    if (!texts.length) throw new Error("Cần ít nhất một văn bản.");
    const body = { query: v.query, texts, use_prefix: !!v.use_prefix, scale_last: v.scale_last };
    const d = await post("/api/embed/compare", body);
    const alt = current?.id === "prefix" ? await post("/api/embed/compare", { ...body, use_prefix: !d.use_prefix }) : null;
    const top = d.rows.find((r) => r.rank_cosine === 1), topDot = d.rows.find((r) => r.rank_dot === 1);
    return {
      steps: steps(d, alt, ctx), diagnosis: diagnose(current?.id, d),
      lessons: ["Embedding đo <b>cùng ngữ cảnh/thống kê</b>, không phải đúng-sai hay ngược nghĩa: cần reranker và đánh giá để lọc.",
        "Chọn metric theo cách model được huấn luyện; với vector đã normalize thì cosine = dot.",
        "Prefix, ngôn ngữ, và model đều là <b>tham số của không gian vector</b>: đổi một thứ thì phải index lại."],
      summary: { label: `${v.use_prefix ? "có" : "không"} prefix · scale ×${v.scale_last}`, facts: {
        "Cosine #1": esc(top.text.slice(0, 28)),
        "Dot #1": topDot.rank_cosine === 1 ? esc(topDot.text.slice(0, 28)) : { v: esc(topDot.text.slice(0, 28)), level: "bad" },
        "Cosine văn bản 1": fmt(d.rows[0].cosine, 3) } },
    };
  },
};
