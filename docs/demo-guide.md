# Demo code cho slide "Embedding, Vector Search & Retrieval Engineering" (Week 6)

13 demo chạy được, bám theo thứ tự slide: từ vector space đến pipeline retrieval hoàn chỉnh.
Mọi số liệu in ra là **đo thật** trên corpus ERP/helpdesk song ngữ Việt-Anh (64 tài liệu, 3 tenant) -
không phải số minh họa trong slide.

## Cài đặt (một lần)

```powershell
uv venv .venv --python 3.14          # hoặc: python -m venv .venv
uv pip install --python .venv\Scripts\python.exe -r requirements.txt   # hoặc: .venv\Scripts\pip install -r requirements.txt
docker compose up -d postgres        # PostgreSQL + pgvector (cổng 5433), cần cho demo 07, 08, 13 chạy bằng venv
                                     # (muốn chạy cả UI: `docker compose up -d --build`, xem mục UI web)
```

- Lần chạy đầu tải 2 model (~0.5 GB mỗi cái): `intfloat/multilingual-e5-small` (embedding, 384 chiều)
  và `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1` (reranker). Sau đó chạy offline từ cache.
- Chạy trên CPU, không cần GPU. Chạy từ thư mục gốc: `.venv\Scripts\python.exe demos\01_vector_space.py`.
- Demo có biểu đồ nhận cờ `--open` để mở HTML trong trình duyệt; file nằm ở `output/`.

## Slide -> demo

| Slide | Demo | Nội dung | Cần DB |
|---|---|---|---|
| 3, 4, 5 | `01_vector_space.py` | Toy 2D như slide; embedding thật chiếu PCA 2D/3D (HTML tương tác), query là ngôi sao đen | |
| 5, 8 | `02_embedding_pipeline.py` | Tự làm lại tokenizer -> transformer -> pooling -> normalize, khớp `model.encode()`; "embedding không hiểu ngôn ngữ"; prefix của e5 | |
| 10 | `03_similarity_metrics.py` | cosine / dot / euclid cho cùng thứ hạng khi norm = 1; tách nhau khi độ lớn khác | |
| 2, 11 | `04_semantic_vs_keyword.py` | BM25 vs vector trên cùng query: paraphrase, đa ngôn ngữ, mã chính xác | |
| 6, 7, 23 | `05_chunking.py` | fixed / overlap / sentence / recursive; đo Recall@1, MRR và số token đưa cho LLM theo chunk size | |
| 12, 14 | `06_exact_vs_ann.py` | Brute force tăng tuyến tính theo N; IVF tự viết ~25 dòng, đường cong recall-latency theo `nprobe` | |
| 13, 14, 15 | `07_pgvector_hnsw_ivfflat.py` | 50K vector: exact vs IVFFlat vs HNSW trong pgvector; `probes` / `ef_search`; kích thước index | x |
| 16, 17 | `08_metadata_filtering.py` | Rò rỉ multi-tenant; post-filter thiếu kết quả; HNSW + WHERE trả thiếu im lặng, cách sửa | x |
| 18, 19 | `09_hybrid_search.py` | Sparse vs dense; Recall theo loại query; đi từng bước RRF; query mà các cách bất đồng | |
| 20 | `10_reranking.py` | Chi phí bi-encoder vs cross-encoder; doc đúng đổi hạng thế nào; số liệu toàn bộ eval | |
| 21, 23, 25, 33 | `11_evaluation.py` | 5 cấu hình retrieval x Recall/Precision/MRR/latency; theo loại query; debug retrieval vs ranking | |
| 27 | `12_query_transformation.py` | rewrite / expansion / multi-query + RRF | |
| 28-32 | `13_end_to_end_pipeline.py` | Pipeline pgvector + BM25 + RRF + rerank; latency budget; log có cấu trúc; embedding cache + semantic cache | x |
| 29, 30 | `api/main.py` | FastAPI `POST /search`: filter bắt buộc, giới hạn `top_k`, whitelist filter, log mỗi request | x |

Chạy API: `.venv\Scripts\python.exe -m uvicorn api.main:app --port 8000`, rồi mở http://localhost:8000/docs.

```json
POST /search
{"query": "tạo lệnh sản xuất", "tenant_id": "company_a", "top_k": 3, "language": "vi", "filters": {"department": "production"}}
```

## UI web (xem quá trình + so sánh) - chạy bằng Docker

```powershell
docker compose up -d --build     # lần đầu build ~vài phút (tải PyTorch CPU + 2 model vào image)
# mở http://localhost:8000          (UI; tài liệu API ở /docs)
docker compose ps                 # app: healthy là sẵn sàng
docker compose logs -f app        # xem log mỗi request (JSON có thời gian từng stage)
docker compose down               # dừng (thêm -v để xóa cả dữ liệu Postgres)
```

- Gồm 2 service: `postgres` (pgvector, cổng host 5433) và `app` (UI + API, cổng 8000). `app` chờ Postgres healthy rồi tự index corpus.
- Model nằm sẵn trong image nên start lên là chạy ngay, không cần mạng. Có `restart: unless-stopped` nên Docker Desktop mở lại là tự lên.
- Sửa code Python/UI thì chạy lại `docker compose up -d --build`.
- Chạy một demo CLI trong container: `docker compose exec app python demos/13_end_to_end_pipeline.py`
  (demo ghi HTML vào `output/` bên trong container; các demo có biểu đồ HTML nên chạy bằng venv trên máy host như phần Cài đặt).
- Chạy không qua Docker (dev): `.venv\Scripts\python.exe -m uvicorn api.main:app --port 8000` rồi mở `http://127.0.0.1:8000/ui/`
  (dùng `127.0.0.1`: uvicorn chỉ nghe IPv4, một số công cụ phân giải `localhost` ra IPv6 trước nên chậm/treo). Khi đó `docker compose up -d postgres` là đủ.

UI là **8 bài học + trang Tổng quan**, đi theo đúng thứ tự một pipeline retrieval (sidebar bên trái; link `#/chunking`, `#/embedding`, …).
Trang **Tổng quan** (`#/overview`) vẽ cả luồng offline/online, mỗi ô nói bước đó làm gì, kiểu lỗi hay gặp, và dẫn tới bài tương ứng.

**Mỗi bài học có cùng 4 phần** (từ lý thuyết ra demo, thấy rõ vấn đề và từng bước chạy):
1. **Lý thuyết:** khái niệm, vì sao quan trọng, điều cần nhớ, một sơ đồ nhỏ.
2. **Chọn tình huống:** các thẻ Vấn đề / Lỗi / Giải pháp / Quan sát lấy từ slide (có ghi slide nào, "slide nói gì", "bạn sẽ thấy gì"). Bấm thẻ chỉ nạp dữ liệu vào form, **chưa chạy gì**; bạn có thể sửa dữ liệu rồi mới bấm chạy.
3. **Chạy từng bước:** thanh luồng hiện toàn bộ các bước ngay từ đầu; bấm ▶ để mở từng bước (hoặc ⏩ chạy hết, hoặc bật tự động phát). Mỗi bước có giải thích, dữ liệu trung gian thật, và thời gian đo thật ở backend (huy hiệu ms).
4. **Chẩn đoán và sửa:** kết luận màu (đỏ/vàng/xanh) vì sao hỏng, bài học rút ra, các nút "thử cách sửa" chạy lại ngay, và **bảng so sánh các lần chạy** (trước/sau khi sửa).

| Nhóm | Bài | Tình huống trong slide (ví dụ) |
|---|---|---|
| A · Offline | 1 Chunking | Đáp án "1,200 units" **bị cắt đôi** ở ranh giới (size 40, overlap 0) → sửa bằng overlap 16 hoặc recursive; chunk quá nhỏ mất ngữ cảnh; chunk quá lớn tốn 555 token cho 1 con số |
| | 2 Embedding | Tokenize → transformer → pooling → normalize (khớp `model.encode()`); embedding không "hiểu" ("I remembered" gần hơn "reset password"); dot product bị đánh lừa bởi độ lớn; đa ngôn ngữ; quên prefix e5 |
| | 3 Index & Vector DB | Exact search không scale (100M vector ≈ 7–9 s và ~154 GB); **IVF** và **HNSW chạy trên dữ liệu 2D** (thấy cụm, nprobe, đường tham lam qua các tầng, recall); đo thật 50.000 vector trên pgvector; filter chọn lọc + HNSW trả thiếu; **3 lỗi pgvector thật**: trộn vector khác chiều, query khác chiều, `vector(2048)` + HNSW (kèm cách sửa `halfvec`) |
| B · Online | 4 Query transformation | "reset pw" #2→#1; "pto request" #7→#1; expansion thêm nhiễu ("stock chk" #1→#4); multi-query |
| | 5 Retrieval & Filter | Mã lỗi `ERR-4102`: BM25 #1, vector #6, hybrid #3; paraphrase: BM25 không tìm được gì; rò rỉ multi-tenant; post-filter trả thiếu; tài liệu lỗi thời; hiện cả câu SQL thật |
| | 6 Reranking | Doc đúng hạng 6 → rerank lên 1; **reranker làm tệ đi** (#1→#3); pool nhỏ thì reranker bất lực; chi phí tăng theo pool |
| C · Đo lường | 7 Đánh giá & Debug | Tính Recall/Precision/MRR bằng tay (ví dụ slide {A,B,C}); chạy bộ eval 36 query (streaming); **debug 8 bước** (slide 33) cho query thất bại, chỉnh pool để chuyển lỗi sang "RETRIEVAL" |
| | 8 Pipeline hoàn chỉnh | Một request qua 8 bước kèm **log JSON có cấu trúc** và ngân sách latency (rerank chiếm 80–90%); **semantic cache** (hit/miss, ms) và lỗi **ngưỡng cache quá thấp trả lời sai** |

Link tiện dùng: `/ui/#/chunking`, `/ui/?case=2#/retrieval` (mở sẵn tình huống số 2, đếm từ 0), `/ui/#/rerank?q=ERR-4102`.

Lưu ý bảo mật: `filter_mode="off"` (tắt filter) chỉ có ở các endpoint `/api/*` của UI demo để minh họa rò rỉ; `POST /search` vẫn bắt buộc tenant.
Thí nghiệm ANN và các lỗi pgvector chạy trên bảng riêng (`ann_bench`) hoặc bảng tạm bị xóa ngay; không đụng bảng `document_chunks`.
Phần IVF/HNSW trên dữ liệu 2D là bản viết tay tối giản chỉ để **giải thích cơ chế**; số đo thật nằm ở thí nghiệm "50.000 vector trên pgvector".

## Cấu trúc

```
data/            corpus.json (64 doc), eval_queries.json (36 query có nhãn), manuals/ + manual_questions.json (chunking)
lib/             config, data, models (embedder + reranker), search (BM25, RRF, metrics), chunking (+ chunking_eval), db (pgvector), ann_bench,
                 pipeline, query_transform, compare (lõi 5 cấu hình), step_* + lesson_* (dữ liệu cho từng bài UI), cache (semantic cache), space_projection (PCA cho bản đồ)
demos/           13 demo, mỗi file một chủ đề
api/             main.py (Search API + phục vụ UI), ui_routes.py + step_routes.py + lesson_routes.py (/api/* cho UI), state.py
ui/              index.html, style.css, lesson.css, js/ (ES modules, không cần build): main.js (router), lesson/ (khung bài học + trình phát từng bước),
                 lessons/ (8 bài), steps/overview.js
docker-compose.yml   PostgreSQL + pgvector
```

## Gợi ý khi giảng - điều đáng nói từ số liệu thật

Số dưới đây từ một lần chạy trên máy phát triển (CPU); chạy lại sẽ lệch chút về latency, còn recall là xác định.
Bộ eval chỉ 36 query, chênh lệch nhỏ (~0.03) là nhiễu - hãy dạy học viên **đọc xu hướng, không đọc từng số**.

- **Demo 01:** bản đồ PCA gốc tách theo **ngôn ngữ**, không phải chủ đề (láng giềng gần nhất cùng ngôn ngữ 95%, cùng chủ đề 59%).
  Bản trừ trung bình theo ngôn ngữ (`01c_*.html`) mới lộ cụm chủ đề (cùng chủ đề 77%). Đây là lý do tìm chéo ngôn ngữ kém hơn.
- **Demo 02:** `"I remembered my password"` gần `"I forgot my password"` hơn cả `"How to reset your password"`
  (0.88 vs 0.85) - embedding đo "cùng ngữ cảnh", không đo ý nghĩa. e5 nén điểm vào ~0.7-1.0, nên đừng bê ngưỡng cosine sang model khác.
- **Demo 04/09:** keyword không vượt được ngôn ngữ (query Anh không ra doc Việt); vector sai ở các mã lỗi na ná
  (`ERR-4102` -> trả `ERR-4210`). Hybrid (RRF) sửa lỗi mã nhưng không tự động thắng ở mọi loại query.
- **Demo 07:** trên 50K vector, exact ~45 ms; HNSW `ef_search=10` đạt ~91% recall ở ~2 ms; IVFFlat `probes=1` chỉ ~43%.
- **Demo 08:** HNSW + `WHERE tenant = ...` (tenant chiếm 1%) trả trung bình **0.3/10** hàng, recall ~4%, không báo lỗi.
  `hnsw.iterative_scan = relaxed_order` hoặc lọc trước rồi exact search đưa về 100%.
- **Demo 11:** rerank là bước cải thiện lớn nhất (Recall@3 ~0.75 -> ~0.96) nhưng cộng ~300 ms trên CPU;
  rerank chiếm ~90% latency của pipeline (demo 13).
- **Demo 05:** corpus chỉ 3 tài liệu nên chênh lệch chunk size nhỏ; hiệu ứng rõ nhất là chunk quá nhỏ (32 token) làm
  Recall@1 giảm, và token đưa cho LLM tăng ~12 lần từ 32 lên 500.

## Khác biệt so với slide (đã kiểm chứng khi dựng demo)

- **Model:** demo dùng `multilingual-e5-small` (384 chiều) cho chạy nhanh trên CPU, thay vì BGE-M3/Qwen3 trong slide.
  Slide 8 ghi BGE-M3 là 1024 chiều nhưng code ví dụ cùng slide in `(768,)` - không nhất quán.
- **Slide 29:** schema `vector(2048)` + chỉ mục HNSW sẽ lỗi trên pgvector: chỉ mục HNSW/IVFFlat cho kiểu `vector`
  giới hạn 2000 chiều (dùng `halfvec` hoặc giảm chiều). Demo dùng 384 chiều nên không dính.
- **Slide 23:** "Hybrid search luôn tốt hơn" chỉ đúng một phần: trên dữ liệu này hybrid nhỉnh hơn vector (Recall@3 ~0.78 vs ~0.75,
  trong ngưỡng nhiễu) và sửa được lỗi mã chính xác, nhưng không thắng ở paraphrase - xem demo 09 và 11. Số trong slide 23 là minh họa.
- **Slide 29:** embedding/reranker chạy in-process (sentence-transformers), không tách container như compose trong slide.
- **HyDE (slide 27)** không có trong demo: cần gọi LLM. Demo 12 làm rewrite/expansion/multi-query bằng từ điển viết tay;
  production thay bằng LLM với cùng cấu trúc.
- **Lexical search:** BM25 chạy in-memory (`rank_bm25`) với mask theo filter, không dùng full-text search của Postgres.
  IDF tính trên toàn corpus nên điểm BM25 hơi khác so với tính riêng từng tenant.
- **API:** `tenant_id` nhận từ body cho dễ demo; production phải lấy từ token xác thực. Một request xử lý tuần tự
  (khóa quanh DB connection và model) - đủ cho demo, không phải cấu hình chịu tải.

## Xử lý sự cố

- `Không kết nối được PostgreSQL`: chạy `docker compose up -d` (Docker Desktop phải đang chạy), kiểm tra cổng 5433.
- Muốn dùng DB khác: đặt biến môi trường `DATABASE_URL`.
- Demo 07 nạp 50K vector lần đầu và build HNSW mất ~20-30 s; lần sau dùng lại bảng `ann_bench`.
