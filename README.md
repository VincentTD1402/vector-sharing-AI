# Embedding, Vector Search & Retrieval Engineering — Demo

Bộ demo cho khóa học: từ không gian vector đến pipeline retrieval hoàn chỉnh (embedding → chunking → pgvector → hybrid → rerank → đánh giá).
Dữ liệu là corpus ERP/helpdesk song ngữ Việt-Anh; model chạy local trên CPU, không cần GPU hay API key.

## Cách nhanh nhất: Docker

Yêu cầu: Docker Desktop (hoặc Docker Engine + Compose v2).

```bash
docker compose up -d --build
```

- Lần build đầu tải model về (vài phút, ~GBs); các lần sau dùng cache.
- Đợi app khởi động (vài chục giây, healthcheck Postgres → app nạp model và index corpus).
- Mở **http://localhost:8000** → giao diện học theo bài (Overview + 8 bài).
- API docs: http://localhost:8000/docs · Health: http://localhost:8000/health
- PostgreSQL + pgvector mở ở cổng host **5433** (user/pass/db: `demo` / `demo` / `semantic_search`).

Dừng: `docker compose down` (thêm `-v` để xóa cả dữ liệu DB).
Xem log: `docker compose logs -f app`.

## UI: 8 bài học

Mỗi bài đi theo: lý thuyết → tình huống trong slide → chạy từng bước → chẩn đoán lỗi → nút sửa → so sánh các lần chạy.
Trang mở ra trống, chỉ chạy khi bấm. Có thể mở thẳng một tình huống bằng `?case=N#/<bài>`.

| Bài | Nội dung |
|-----|----------|
| Chunking | cắt đáp án ở ranh giới, size/overlap, recursive |
| Embedding | vector, prefix `query:`/`passage:`, cosine vs dot |
| Index & Vector DB | exact vs IVF/HNSW, 50K vector thật, lỗi số chiều, filter |
| Query transformation | rewrite / expansion / multi-query |
| Retrieval | BM25 vs vector vs hybrid (RRF), mã lỗi ERR-xxxx |
| Rerank | cross-encoder, pool size, rerank làm tệ đi |
| Evaluate | Recall@K, Precision@K, MRR, debug 8 bước |
| Pipeline | request end-to-end, latency từng bước, semantic cache |

## Chạy demo CLI (không qua Docker)

Yêu cầu: Python 3.12+ và Docker (chỉ cho Postgres).

```bash
# 1. môi trường
python -m venv .venv                       # hoặc: uv venv .venv
.venv\Scripts\pip install -r requirements.txt     # Linux/macOS: .venv/bin/pip

# 2. Postgres + pgvector (cần cho demo 07, 08, 13 và UI)
docker compose up -d postgres

# 3. chạy một demo (từ thư mục gốc)
.venv\Scripts\python.exe demos\01_vector_space.py
```

| Demo | Chủ đề |
|------|--------|
| 01 | trực quan hóa không gian vector (PCA) |
| 02 | pipeline embedding |
| 03 | các thước đo similarity |
| 04 | semantic vs keyword search |
| 05 | chunking |
| 06 | exact vs ANN |
| 07 | pgvector HNSW / IVFFlat |
| 08 | metadata filtering, multi-tenant |
| 09 | hybrid search |
| 10 | reranking |
| 11 | đánh giá (Recall/MRR) |
| 12 | query transformation |
| 13 | pipeline hoàn chỉnh |

Demo có biểu đồ ghi file HTML vào `output/` — mở bằng trình duyệt.

## Chạy API + UI bằng venv (dev)

```bash
docker compose up -d postgres
.venv\Scripts\python.exe -m uvicorn api.main:app --port 8000
```

Mở `http://127.0.0.1:8000/ui/` (dùng `127.0.0.1`, vì uvicorn chỉ nghe IPv4 và `localhost` có thể phân giải ra IPv6 gây chậm).
Biến môi trường `DATABASE_URL` mặc định trỏ tới `localhost:5433`; trong Docker là `postgresql://demo:demo@postgres:5432/semantic_search`.

## Cấu trúc

```
api/        FastAPI: /search, /api/*, /api/lesson/*, phục vụ UI tại /ui
lib/        lõi: models, search (BM25/RRF), chunking, db (pgvector), pipeline, compare, các lesson_*
ui/         giao diện thuần ES-module, không cần build
demos/      13 script demo CLI
data/       corpus.json (64 doc), eval_queries.json (36 query), manuals/
docker-compose.yml, Dockerfile
```

## Lưu ý

- Lần chạy đầu của ANN explorer (Index & Vector DB) tạo 50.000 vector tổng hợp, mất ~20 giây; build HNSW/IVFFlat tốn CPU nên chỉ chạy một lần tại một thời điểm.
- Bộ eval nhỏ (36 query): chênh lệch ±0.03 là nhiễu. Recall của HNSW thật dao động nhẹ giữa các lần chạy.
- HyDE chưa được cài vì cần LLM. IVF/HNSW 2D trong UI là bản viết tay để giải thích cơ chế.
- Cổng 5433 hoặc 8000 bị chiếm: đổi mapping `ports` trong `docker-compose.yml`.
