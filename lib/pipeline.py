"""Pipeline retrieval production trên pgvector (slide 28-31).

query -> embed -> [dense: pgvector + SQL filter] + [lexical: BM25 có mask] -> RRF -> rerank -> top-k

- Filter (tenant, status, language, department) áp dụng ở CẢ hai nhánh. BM25: mask trước khi lấy top-k.
  pgvector: WHERE chính xác khi bảng nhỏ; với chỉ mục HNSW trên bảng lớn cần iterative scan (demo 08 B).
- Mọi bước được đo thời gian; `trace` trả về đủ để ghi log/debug (slide 31, 33).
"""
import threading
import time
from contextlib import contextmanager

import numpy as np
from rank_bm25 import BM25Okapi

from lib.db import connect, index_corpus, vector_search
from lib.models import embed_queries, rerank_scores
from lib.search import rrf, tokenize

ALLOWED_FILTERS = {"department"}  # whitelist: client không được tự chọn cột để lọc


class RetrievalService:
    def __init__(self, pool: int = 20):
        self.pool = pool
        self.conn = connect()
        n = index_corpus(self.conn)  # idempotent: upsert, chạy lại không tạo bản trùng
        self.rows = [
            dict(zip(("id", "content", "tenant_id", "language", "status", "department"), r))
            for r in self.conn.execute(
                "SELECT id, content, tenant_id, language, status, department FROM document_chunks ORDER BY id"
            ).fetchall()
        ]
        self.bm25 = BM25Okapi([tokenize(r["content"]) for r in self.rows])
        self.chunk_count = n
        self._embed_cache: dict[str, np.ndarray] = {}  # cache embedding của query lặp lại
        self._lock = threading.Lock()  # psycopg connection và model không dùng đồng thời an toàn

    def embed_query(self, query: str, trace: dict | None = None) -> np.ndarray:
        vec = self._embed_cache.get(query)  # một lần đọc: an toàn khi cache bị xóa từ luồng khác
        hit = vec is not None
        if not hit:
            vec = embed_queries([query])[0]
            self._embed_cache[query] = vec
        if trace is not None:
            trace["embed_cache_hit"] = hit
        return vec

    @contextmanager
    def _timed(self, trace: dict, stage: str):
        start = time.perf_counter()
        yield
        trace["timings_ms"][stage] = round((time.perf_counter() - start) * 1000, 1)

    def lexical_scored(
        self, query: str, tenant_id: str, languages, department, k: int, filters_off: bool = False,
    ) -> list[tuple[str, float]]:
        """BM25 trên các chunk thỏa filter (mask điểm = -inf cho chunk bị loại).
        filters_off chỉ dành cho UI minh họa rò rỉ: bỏ filter tenant + status."""
        scores = self.bm25.get_scores(tokenize(query)).astype(float)
        keep = np.array([
            (filters_off or (r["tenant_id"] == tenant_id and r["status"] == "current"))
            and (not languages or r["language"] in languages)
            and (not department or r["department"] == department)
            for r in self.rows
        ])
        scores[~keep] = -np.inf
        order = np.argsort(-scores)[:k]
        return [(self.rows[i]["id"], float(scores[i])) for i in order if scores[i] > 0]  # 0 = không khớp từ nào

    def _lexical(self, query: str, tenant_id: str, languages, department, k: int) -> list[str]:
        return [i for i, _ in self.lexical_scored(query, tenant_id, languages, department, k)]

    def search(
        self, query: str, tenant_id: str, top_k: int = 5, languages: list[str] | None = None,
        department: str | None = None, rerank: bool = True,
    ) -> tuple[list[dict], dict]:
        if not tenant_id:
            raise ValueError("tenant_id là bắt buộc")
        trace: dict = {
            "query": query,
            "filters": {"tenant_id": tenant_id, "status": "current", "languages": languages, "department": department},
            "timings_ms": {},
        }
        with self._lock:
            t0 = time.perf_counter()
            with self._timed(trace, "embed"):
                qv = self.embed_query(query, trace)
            with self._timed(trace, "dense_search"):
                dense = vector_search(self.conn, qv, self.pool, tenant_id, "current", languages, department)
            with self._timed(trace, "lexical_search"):
                lexical_ids = self._lexical(query, tenant_id, languages, department, self.pool)
            with self._timed(trace, "fusion"):
                fused = rrf([lexical_ids, [d["id"] for d in dense]])[: self.pool]
            by_id = {r["id"]: r for r in self.rows}

            if rerank and fused:
                with self._timed(trace, "rerank"):
                    scores = rerank_scores(query, [by_id[i]["content"] for i, _ in fused])
                    order = np.argsort(-scores)[:top_k]
                    final = [(fused[i][0], float(scores[i])) for i in order]
            else:
                final = [(i, s) for i, s in fused[:top_k]]
            trace["timings_ms"].setdefault("rerank", 0.0)  # không có ứng viên -> bỏ qua rerank
            trace["timings_ms"]["total"] = round((time.perf_counter() - t0) * 1000, 1)

        trace["dense_ids"] = [d["id"] for d in dense]
        trace["lexical_ids"] = lexical_ids
        trace["final"] = [{"id": i, "score": round(s, 4)} for i, s in final]
        results = [{**by_id[i], "score": s} for i, s in final]
        return results, trace
