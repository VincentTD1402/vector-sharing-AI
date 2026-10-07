// Bài 3 - Index & Vector DB: vì sao cần ANN, IVF/HNSW chạy ra sao, và các lỗi thật khi lưu vector vào pgvector.
import { get } from "../api.js";
import { callout, kv, ms, sqlBlock, table } from "../lesson/render.js";
import { esc, fmt } from "../util.js";
import { hnswSvg, ivfLegend, ivfSvg, resultLegend } from "./ann-visuals.js";
import { mountExplorer, mountFilterLab, mountStoredData } from "./ann-explorer.js";

const box = (svg, legend = "") => `<div class="toybox">${svg}</div>${legend}`;

function scalingSteps(d) {
  const maxMs = Math.max(...d.rows.map((r) => r.ms));
  return [
    { title: "Chuẩn bị N vector", ms: null,
      explain: `Mỗi vector có ${d.dim} chiều (float32). Một query phải so với <b>mọi</b> vector: số phép nhân-cộng = N × ${d.dim}.`,
      summary: `N từ ${d.rows[0].n.toLocaleString()} đến ${d.rows.at(-1).n.toLocaleString()}`,
      html: table([{ t: "N vector", num: true }, { t: "RAM chứa vector", num: true }, { t: "Phép tính / query", num: true }],
        d.rows.map((r) => [r.n.toLocaleString(), `${r.mb} MB`, `${r.ops_m} triệu`])) },
    { title: "Quét toàn bộ cho 1 query (exact)", ms: ms(d.rows.at(-1).ms),
      explain: "Brute force: nhân ma trận N×D với vector query rồi lấy top-k. Chính xác 100% nhưng <b>thời gian tăng tuyến tính theo N</b>.",
      summary: `${d.rows.at(-1).n.toLocaleString()} vector: ${d.rows.at(-1).ms} ms`,
      html: table([{ t: "N vector", num: true }, { t: "Latency / query", num: true }, { t: "" }],
        d.rows.map((r) => [r.n.toLocaleString(), `<b>${r.ms} ms</b>`, `<div style="min-width:200px"><div class="bar" style="width:${(r.ms / maxMs) * 100}%"></div></div>`])) },
    { title: "Ngoại suy lên quy mô thật", ms: null,
      explain: "Thời gian và RAM tăng tuyến tính nên có thể ngoại suy từ số đo ở trên. Đây là con số mà một hệ thống enterprise thực sự phải đối mặt.",
      summary: `100M vector ≈ ${fmt(d.extrapolated.at(-1).ms / 1000, 1)} s`,
      html: table([{ t: "N vector", num: true }, { t: "Latency / query (ước tính)", num: true }, { t: "RAM", num: true }],
        d.extrapolated.map((r) => [r.n.toLocaleString(), `${r.ms >= 1000 ? fmt(r.ms / 1000, 1) + " giây" : r.ms + " ms"}`, `${r.gb} GB`]),
        (i) => (i === d.extrapolated.length - 1 ? "bad-row" : "")) },
  ];
}

function ivfSteps(d) {
  const pct = ((d.scanned / d.total) * 100).toFixed(1);
  return [
    { title: "Dữ liệu 2D (240 điểm)", ms: null, summary: `${d.total} điểm`,
      explain: "Minh họa bằng dữ liệu 2D để <b>nhìn thấy</b> thuật toán (thực tế là 384 chiều). Exact search phải so query với cả 240 điểm.",
      html: box(ivfSvg(d, "raw")) },
    { title: "Training: K-means chia 8 cụm", ms: null, summary: "8 cụm (cell) + 8 tâm",
      explain: "IVF (Inverted File) chạy <b>K-means</b> tìm 8 tâm cụm (centroid) rồi gán mỗi vector vào tâm gần nhất: mỗi cụm là một \"danh sách đảo\". Bước này chỉ làm một lần lúc build index (nên cần dữ liệu có sẵn).",
      html: box(ivfSvg(d, "cells"), ivfLegend) },
    { title: `Query: chọn ${d.nprobe} cụm gần nhất`, ms: null, summary: `nprobe = ${d.nprobe} → cụm ${d.probed.join(", ")}`,
      explain: `Query (★) chỉ so với 8 tâm cụm (rất ít phép tính) rồi chọn <b>nprobe = ${d.nprobe}</b> cụm gần nhất để quét. Query này cố ý đặt gần <b>ranh giới</b> giữa hai cụm.`,
      html: box(ivfSvg(d, "probe"), ivfLegend) },
    { title: "Chỉ quét các cụm đã chọn", ms: null, summary: `${d.scanned}/${d.total} điểm (${pct}%)`,
      explain: `Chỉ so query với các điểm thuộc cụm được chọn: <b>${d.scanned}/${d.total} điểm (${pct}%)</b> thay vì cả ${d.total}, nhanh hơn khoảng <b>${fmt(d.total / d.scanned, 0)}×</b>. Các điểm mờ không bao giờ được xem tới.`,
      html: box(ivfSvg(d, "scan"), ivfLegend) },
    { title: "So với exact search", ms: null, summary: `recall ${(d.recall * 100).toFixed(0)}%`,
      explain: `Top-${d.top_k} của IVF so với top-${d.top_k} thật. <b>Recall = ${(d.recall * 100).toFixed(0)}%</b>: đây là đánh đổi của ANN.`,
      html: box(ivfSvg(d, "result"), resultLegend) },
  ];
}

function hnswSteps(d) {
  const top = d.top_layer;
  return [
    { title: "Đồ thị nhiều tầng", ms: null, summary: `${top + 1} tầng, ${d.points.length} điểm`,
      explain: "HNSW xếp các điểm vào nhiều tầng: tầng trên <b>thưa</b> (đường cao tốc, bước nhảy xa), tầng dưới <b>dày</b> (đường nội bộ, chính xác). Mỗi điểm nối với vài láng giềng gần nhất trong tầng của nó.",
      html: hnswSvg(d, "layers") },
    { title: "Vào ở tầng trên cùng", ms: null, summary: `điểm vào #${d.entry}`,
      explain: "Tìm kiếm bắt đầu ở một điểm cố định ở tầng cao nhất (vòng đỏ), rồi <b>tham lam</b>: nhảy sang láng giềng gần query (★) hơn cho tới khi không còn láng giềng nào gần hơn.",
      html: hnswSvg(d, "descent") },
    { title: "Xuống tầng dưới, lặp lại", ms: null, summary: `${d.descent.length} bước nhảy`,
      explain: `Hết cải thiện thì <b>hạ xuống tầng kế</b> ngay tại điểm hiện tại và tiếp tục (đường đỏ). Cứ thế tới tầng 0. Tổng ${d.descent.length} điểm ghé qua ở các tầng trên.`,
      html: hnswSvg(d, "descent") },
    { title: `Tầng 0: tìm theo chùm (ef = ${d.ef})`, ms: null, summary: `duyệt ${d.visited.length}/${d.points.length} điểm`,
      explain: `Ở tầng dưới cùng, giữ <b>ef = ${d.ef}</b> ứng viên tốt nhất và mở rộng dần (số cam = thứ tự mở rộng). ef lớn → xem nhiều điểm hơn → recall cao hơn nhưng chậm hơn. Đã duyệt <b>${d.visited.length}/${d.points.length}</b> điểm (brute force phải duyệt hết).`,
      html: hnswSvg(d, "layer0") },
    { title: "So với exact search", ms: null, summary: `recall ${(d.recall * 100).toFixed(0)}%`,
      explain: `Top-${d.top_k} của HNSW so với top-${d.top_k} thật: <b>recall ${(d.recall * 100).toFixed(0)}%</b> với ${d.visited.length}/${d.points.length} điểm được xem.`,
      html: hnswSvg(d, "result") + resultLegend },
  ];
}

const ERR = {
  dim_mismatch: { title: "Trộn vector của hai model khác nhau", explain: [
    "Tạo bảng với cột <code>vector(384)</code>: đúng số chiều của multilingual-e5-small.",
    "Chèn một vector 384 chiều: hợp lệ.",
    "Chèn vector <b>768 chiều</b> (ví dụ từ model khác): pgvector từ chối vì khác số chiều."],
    diag: { level: "bad", title: "Lỗi: không thể trộn vector từ model khác nhau",
      html: "Mỗi model tạo vector với số chiều và <b>không gian riêng</b>. Ngay cả khi hai model cùng 384 chiều, vector của chúng <b>không so sánh được</b> với nhau và pgvector không báo lỗi. Quy tắc: query và tài liệu phải dùng <b>cùng một model</b>; đổi model = re-index toàn bộ." },
    lessons: ["Số chiều khác nhau thì DB báo lỗi rõ ràng, nhưng cùng số chiều khác model thì <b>im lặng cho kết quả vô nghĩa</b>.", "Lưu tên model + phiên bản cùng index để biết cần re-index khi nào."] },
  query_dim: { title: "Query khác chiều với cột lưu", explain: [
    "Tạo bảng <code>vector(384)</code>.", "Chèn một dòng 384 chiều.",
    "Truy vấn bằng vector chỉ <b>3 chiều</b> (ví dụ do embed nhầm model/không đủ chiều): lỗi so sánh."],
    diag: { level: "bad", title: "Lỗi: vector query và vector lưu khác số chiều",
      html: "Đây là lỗi hay gặp khi đổi/nhầm model ở nhánh query nhưng quên index lại. Hãy kiểm tra số chiều ở tầng ứng dụng <b>trước</b> khi gọi DB để báo lỗi dễ hiểu." },
    lessons: ["Kiểm tra <code>len(query_vec) == EMBED_DIM</code> ở tầng ứng dụng.", "Một chuỗi cấu hình duy nhất cho model dùng ở cả index-time và query-time."] },
  hnsw_2048: { title: "vector(2048) + chỉ mục HNSW (ví dụ trong slide)", explain: [
    "Tạo bảng <code>vector(2048)</code> cho Qwen3-Embedding 2048 chiều, như schema trong slide.",
    "Tạo chỉ mục HNSW trên cột đó: pgvector từ chối vì vượt giới hạn <b>2000 chiều</b> cho kiểu vector.",
    "Dọn bảng để thử cách sửa.",
    "Cách sửa: dùng kiểu <code>halfvec(2048)</code> (số 16-bit) nên hỗ trợ index tới 4000 chiều.",
    "Tạo HNSW trên halfvec: thành công."],
    diag: { level: "good", title: "Lỗi đã tái hiện và đã sửa bằng halfvec",
      html: "Chỉ mục HNSW/IVFFlat cho kiểu <code>vector</code> giới hạn <b>2000 chiều</b>. Cách sửa: <code>halfvec</code> (đổi lấy độ chính xác 16-bit), giảm chiều (Matryoshka / PCA), hoặc dùng model ít chiều hơn. Lỗi này có thật trong schema ở slide 29." },
    lessons: ["Kiểm tra giới hạn chiều của chỉ mục <b>trước khi chọn model embedding</b>.", "halfvec giảm một nửa bộ nhớ nhưng giảm độ chính xác số học; hãy đo recall."] },
};

function errorSteps(c, d) {
  const e = ERR[c];
  return d.steps.map((s, i) => ({
    title: s.ok ? `Câu lệnh ${i + 1}` : `Câu lệnh ${i + 1} → LỖI`, ms: null, summary: s.ok ? "thành công" : "BỊ TỪ CHỐI",
    explain: e.explain[i] ?? "",
    html: `${sqlBlock(s.sql)}${s.ok ? callout("good", "Thành công.") : callout("bad", `<b>PostgreSQL trả về lỗi thật:</b><br><code>${esc(s.message)}</code>`)}`,
  }));
}

export default {
  id: "index", num: 3, group: "A", title: "Index & Vector DB", slides: "12, 13, 14, 15, 29",
  tagline: "Lưu vector + metadata vào pgvector và đánh chỉ mục ANN để tìm nhanh. Thấy IVF/HNSW chạy ra sao và các lỗi thật khi lưu vector.",
  theory: {
    what: `<p><b>Exact search</b> so query với mọi vector: O(N × D). <b>ANN</b> (Approximate Nearest Neighbor) dùng cấu trúc dữ liệu để chỉ xem một phần nhỏ.</p>
      <p><b>IVF</b>: chia không gian thành cụm, chỉ quét <code>nprobe</code> cụm. <b>HNSW</b>: đồ thị nhiều tầng, đi tham lam từ tầng thưa xuống tầng dày.</p>`,
    why: `<p>10K vector còn ổn; 1M thì giới hạn; 100M thì mất vài giây và hàng trăm GB RAM: không chấp nhận được. ANN đổi <b>một ít recall</b> lấy tốc độ gấp 10–100 lần.</p>`,
    points: ["<b>Recall@K</b> đo ANN bỏ sót bao nhiêu láng giềng thật; điều chỉnh bằng <code>nprobe</code> / <code>ef_search</code>.",
      "HNSW: recall cao, thêm dữ liệu động, tốn RAM. IVFFlat: nhỏ, build nhanh, cần train.",
      "Giới hạn: chỉ mục HNSW/IVFFlat cho <code>vector</code> tối đa <b>2000 chiều</b>."],
    visual: `<svg viewBox="0 0 760 100" class="theorysvg" role="img" aria-label="Exact vs ANN">
      <rect x="10" y="14" width="220" height="72" rx="8" fill="#fde8e8" stroke="#b91c1c"/><text x="120" y="42" text-anchor="middle" font-size="13" font-weight="700" fill="#b91c1c">Exact</text><text x="120" y="62" text-anchor="middle" font-size="12">so với TẤT CẢ N vector</text><text x="120" y="78" text-anchor="middle" font-size="11" fill="#7f1d1d">100% đúng · chậm khi N lớn</text>
      <rect x="270" y="14" width="220" height="72" rx="8" fill="#fef3c7" stroke="#d97706"/><text x="380" y="42" text-anchor="middle" font-size="13" font-weight="700" fill="#b45309">IVF</text><text x="380" y="62" text-anchor="middle" font-size="12">chỉ quét nprobe cụm</text><text x="380" y="78" text-anchor="middle" font-size="11" fill="#92400e">nhỏ, build nhanh</text>
      <rect x="530" y="14" width="220" height="72" rx="8" fill="#dcfce7" stroke="#16a34a"/><text x="640" y="42" text-anchor="middle" font-size="13" font-weight="700" fill="#15803d">HNSW</text><text x="640" y="62" text-anchor="middle" font-size="12">đồ thị nhiều tầng</text><text x="640" y="78" text-anchor="middle" font-size="11" fill="#166534">recall cao, tốn RAM</text></svg>`,
  },
  fields: [
    { name: "nprobe", label: "nprobe (số cụm được quét)", type: "range", min: 1, max: 8, showFor: ["ivf"] },
    { name: "ef", label: "ef (số ứng viên giữ ở tầng 0)", type: "range", min: 2, max: 20, showFor: ["hnsw"] },
  ],
  cases: [
    { id: "scaling", tag: "problem", title: "Exact search không scale", slide: "12",
      story: "Exact search: chính xác 100% nhưng O(N×D) mỗi query, không scale khi N lớn (100M vector: vài giây, không chấp nhận được).",
      expect: "Latency tăng tuyến tính theo N; ngoại suy 100 triệu vector ≈ 9 giây/query và ~154 GB RAM.",
      values: {}, fixes: [{ label: "Giải pháp 1: IVF (chia cụm)", href: "#/index?case=1" }, { label: "Giải pháp 2: HNSW (đồ thị)", href: "#/index?case=2" }] },
    { id: "ivf", tag: "fix", title: "IVF: chia cụm, chỉ quét vài cụm", slide: "14",
      story: "IVFFlat: K-means chia cụm, query chỉ tìm trong nprobe cụm gần nhất. nprobe = nlist tương đương exact search.",
      expect: "nprobe = 1 chỉ quét ~5% điểm nhưng bỏ sót láng giềng nằm cụm bên cạnh (recall 60%); nprobe = 2 đủ 100%.",
      values: { nprobe: 1 }, fixes: [{ label: "Tăng nprobe lên 2", patch: { nprobe: 2 } }, { label: "nprobe = 8 (quét hết = exact)", patch: { nprobe: 8 } }] },
    { id: "hnsw", tag: "fix", title: "HNSW: đồ thị nhiều tầng", slide: "13",
      story: "HNSW: vào đồ thị ở tầng trên cùng, điều hướng tham lam, xuống tầng dưới khi không còn neighbor gần hơn, tại tầng 0 thu thập top-K. ef_search lớn → recall cao hơn, latency cao hơn.",
      expect: "ef = 2 duyệt rất ít điểm và bỏ sót (recall 40%); ef = 8 đủ 100% mà vẫn không duyệt hết 90 điểm.",
      values: { ef: 2 }, fixes: [{ label: "Tăng ef lên 8", patch: { ef: 8 } }, { label: "Tăng ef lên 16", patch: { ef: 16 } }] },
    { id: "explorer", tag: "observe", title: "Đo thật trên 50.000 vector (pgvector)", slide: "13, 14",
      story: "So sánh HNSW và IVFFlat: recall, memory, build time, khả năng thêm dữ liệu động.",
      expect: "IVFFlat probes=1 chỉ ~40% recall; HNSW ef_search=10 đã ~95% ở vài ms; exact ~85 ms.",
      custom: (el) => mountExplorer(el) },
    { id: "filter", tag: "error", title: "Filter chọn lọc + HNSW trả thiếu", slide: "17",
      story: "Pre-filter: nếu tập sau lọc quá nhỏ → ANN index hoạt động kém. Post-filter: top-K nhỏ → miss results.",
      expect: "WHERE tenant = 7 (1% dữ liệu) với HNSW mặc định chỉ trả 0.3/10 hàng, recall 4%, không báo lỗi.",
      custom: (el) => mountFilterLab(el) },
    { id: "stored", tag: "observe", title: "Dữ liệu đang lưu trong pgvector", slide: "15",
      story: "Vector DB lưu vector + metadata; pgvector là extension của PostgreSQL, hỗ trợ HNSW/IVFFlat và filter bằng SQL WHERE.",
      expect: "Bảng 64 chunk, 3 chỉ mục (HNSW + 2 btree), mỗi dòng có tenant/language/status/department.",
      custom: (el) => mountStoredData(el) },
    { id: "dim_mismatch", tag: "error", title: "Lỗi: trộn vector của model khác nhau", slide: "8, 9",
      story: "Không thể trộn vector từ model khác nhau; thay đổi model = rebuild toàn bộ index.", expect: "PostgreSQL từ chối: expected 384 dimensions, not 768." },
    { id: "query_dim", tag: "error", title: "Lỗi: query khác chiều với cột lưu", slide: "8, 11",
      story: "Query và document phải dùng cùng embedding model.", expect: "PostgreSQL báo: different vector dimensions 384 and 3." },
    { id: "hnsw_2048", tag: "error", title: "Lỗi: vector(2048) + HNSW (schema trong slide)", slide: "29",
      story: "Schema demo: embedding vector(2048) với index HNSW cho Qwen3-Embedding.",
      expect: "CREATE INDEX báo: column cannot have more than 2000 dimensions for hnsw index; sau đó cách sửa bằng halfvec chạy được." },
  ],
  async run(v, ctx, current) {
    const id = current.id;
    if (id === "scaling") {
      const d = await get("/api/lesson/ann/scaling");
      const last = d.extrapolated.at(-1);
      return { steps: scalingSteps(d),
        diagnosis: { level: "bad", title: `100 triệu vector ≈ ${fmt(last.ms / 1000, 1)} giây mỗi query và ${last.gb} GB RAM`,
          html: `Đo thật: ${d.rows.at(-1).n.toLocaleString()} vector mất <b>${d.rows.at(-1).ms} ms</b>/query; thời gian tăng <b>tuyến tính</b> theo N nên không thể scale. Cần chỉ mục ANN.` },
        lessons: ["Exact search vẫn đúng cho tập nhỏ và là <b>chuẩn để đo recall</b> của ANN.", "Chi phí là O(N × D): cả số vector lẫn số chiều đều làm chậm."],
        summary: { label: "exact", facts: {} } };
    }
    if (id === "ivf") {
      const d = await get(`/api/lesson/ann/ivf?nprobe=${v.nprobe}`);
      const miss = d.exact.length - d.found.filter((x) => d.exact.includes(x)).length;
      return { steps: ivfSteps(d),
        diagnosis: d.recall < 1
          ? { level: "warn", title: `Recall ${(d.recall * 100).toFixed(0)}%: bỏ sót ${miss}/${d.top_k} láng giềng thật`, html: `Query nằm sát ranh giới nên một số láng giềng thật thuộc cụm <b>chưa được quét</b>. Đổi lại IVF chỉ xem ${d.scanned}/${d.total} điểm. Tăng <b>nprobe</b> để đổi tốc độ lấy recall.` }
          : { level: "good", title: `Recall 100% chỉ với ${d.scanned}/${d.total} điểm`, html: `Quét ${d.nprobe} cụm là đủ tìm toàn bộ láng giềng thật. Tăng nprobe thêm chỉ làm chậm hơn mà không lợi gì.` },
        lessons: ["<b>nprobe</b> là núm vặn đánh đổi: nhỏ = nhanh nhưng dễ bỏ sót ở ranh giới cụm; nprobe = nlist là exact.", "IVF cần <b>train</b> K-means trên dữ liệu có sẵn; dữ liệu thêm sau có thể làm cụm lệch."],
        summary: { label: `nprobe = ${d.nprobe}`, facts: { "Điểm quét": `${d.scanned}/${d.total}`, "Recall": { v: `${(d.recall * 100).toFixed(0)}%`, level: d.recall === 1 ? "good" : "warn" } } } };
    }
    if (id === "hnsw") {
      const d = await get(`/api/lesson/ann/hnsw?ef=${v.ef}`);
      return { steps: hnswSteps(d),
        diagnosis: d.recall < 1
          ? { level: "warn", title: `Recall ${(d.recall * 100).toFixed(0)}% với ef = ${d.ef}`, html: `Chỉ giữ ${d.ef} ứng viên nên chùm tìm kiếm dừng sớm và bỏ sót láng giềng thật. Tăng <b>ef</b> để duyệt thêm điểm (đổi latency lấy recall).` }
          : { level: "good", title: `Recall 100% sau khi duyệt ${d.visited.length}/${d.points.length} điểm`, html: "Đường đi tham lam ở tầng trên đưa ta tới đúng vùng; tầng 0 chỉ cần xem lân cận." },
        lessons: ["HNSW nhanh nhờ <b>đường tắt ở tầng trên</b>: tới đúng vùng trong vài bước nhảy.", "<b>ef_search</b> là núm vặn recall/latency lúc query; <b>m</b> và <b>ef_construction</b> quyết định chất lượng đồ thị lúc build (và RAM)."],
        summary: { label: `ef = ${d.ef}`, facts: { "Điểm duyệt": `${d.visited.length}/${d.points.length}`, "Recall": { v: `${(d.recall * 100).toFixed(0)}%`, level: d.recall === 1 ? "good" : "warn" } } } };
    }
    const d = await get(`/api/lesson/ann/errors?case=${id}`);
    const e = ERR[id];
    return { steps: errorSteps(id, d), diagnosis: e.diag, lessons: e.lessons, summary: { label: id, facts: {} } };
  },
};
