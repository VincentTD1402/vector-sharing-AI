"""Bước Index & Vector DB của UI: nội dung bảng pgvector, ANN explorer (exact/IVFFlat/HNSW), filter chọn lọc.

ANN explorer dùng bảng `ann_bench` (50.000 vector tổng hợp có cấu trúc) như demo 07/08 - dữ liệu thật
(64 chunk) quá nhỏ để thấy khác biệt giữa các chỉ mục.
"""
import re
import threading
import time
from collections.abc import Iterator

import numpy as np

from lib.ann_bench import ensure_ann_bench_table, exact_top_k, make_queries
from lib.db import connect, plan_nodes
from lib.pipeline import RetrievalService

K, N, N_QUERIES = 10, 50_000, 30
SEARCH_SQL = "SELECT id FROM ann_bench ORDER BY embedding <=> %s LIMIT %s"
IVF_PROBES, HNSW_EF = (1, 5, 10, 20, 100), (10, 20, 40, 100, 400)
_busy = threading.Lock()  # chỉ một lần chạy ANN tại một thời điểm (build index tốn CPU)
_cache: dict = {}


def index_info(svc: RetrievalService) -> dict:
    with svc._lock:
        c = svc.conn
        rows = c.execute("SELECT count(*), pg_total_relation_size('document_chunks') FROM document_chunks").fetchone()
        groups = c.execute("SELECT tenant_id, status, language, count(*) FROM document_chunks GROUP BY 1,2,3 ORDER BY 1,2,3").fetchall()
        indexes = c.execute("SELECT indexname, indexdef, pg_relation_size(indexname::regclass) FROM pg_indexes "
                            "WHERE tablename = 'document_chunks' ORDER BY indexname").fetchall()
        sample = c.execute("SELECT id, tenant_id, language, status, department, content, embedding "
                           "FROM document_chunks ORDER BY id LIMIT 6").fetchall()
        has_ann = c.execute("SELECT to_regclass('ann_bench')").fetchone()[0] is not None
        ann_rows = c.execute("SELECT count(*) FROM ann_bench").fetchone()[0] if has_ann else 0
    return {
        "rows": rows[0], "total_mb": round(rows[1] / 1e6, 2), "dim": len(sample[0][6].to_numpy()) if sample else 0,
        "groups": [{"tenant_id": g[0], "status": g[1], "language": g[2], "count": g[3]} for g in groups],
        "indexes": [{"name": i[0], "type": (re.search(r"USING (\w+)", i[1]) or [None, "?"])[1],
                     "ddl": i[1], "mb": round(i[2] / 1e6, 3)} for i in indexes],
        "sample": [{"id": s[0], "tenant_id": s[1], "language": s[2], "status": s[3], "department": s[4],
                    "content": s[5][:90], "embedding": [round(float(x), 3) for x in s[6].to_numpy()[:6]]}
                   for s in sample],
        "ann_bench_rows": ann_rows,
    }


def _dataset(conn):
    vectors = ensure_ann_bench_table(conn, N)
    if "truth" not in _cache:
        queries = make_queries(vectors, N_QUERIES)
        _cache.update(queries=queries, truth=[set(exact_top_k(vectors, q, K).tolist()) for q in queries])
    return vectors, _cache["queries"], _cache["truth"]


def _run(conn, queries, truth) -> tuple[float, float]:
    recalls, times = [], []
    for q, t in zip(queries, truth):
        start = time.perf_counter()
        ids = {r[0] for r in conn.execute(SEARCH_SQL, (q, K)).fetchall()}
        times.append((time.perf_counter() - start) * 1000)
        recalls.append(len(ids & t) / K)
    return round(float(np.mean(recalls) * 100), 1), round(float(np.mean(times)), 2)


def _drop_indexes(conn) -> None:
    for name in ("ann_bench_hnsw", "ann_bench_ivf"):
        conn.execute(f"DROP INDEX IF EXISTS {name}")


def _build(conn, kind: str) -> dict:
    ddl = {"ivfflat": "CREATE INDEX ann_bench_ivf ON ann_bench USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100)",
           "hnsw": "CREATE INDEX ann_bench_hnsw ON ann_bench USING hnsw (embedding vector_cosine_ops) WITH (m = 16, ef_construction = 64)"}[kind]
    name = "ann_bench_ivf" if kind == "ivfflat" else "ann_bench_hnsw"
    start = time.perf_counter()
    conn.execute(ddl)
    size = conn.execute("SELECT pg_relation_size(%s::regclass)", (name,)).fetchone()[0]
    return {"seconds": round(time.perf_counter() - start, 1), "mb": round(size / 1e6, 1), "ddl": ddl}


def ann_events(kind: str) -> Iterator[dict]:
    """Sinh sự kiện (status / build / row / done) cho UI cập nhật dần."""
    if not _busy.acquire(blocking=False):
        yield {"error": "Đang có một lần chạy ANN khác, hãy đợi nó xong."}
        return
    conn = None
    try:
        conn = connect()
        yield {"status": f"Chuẩn bị {N:,} vector trong pgvector (lần đầu mất ~20 giây)..."}
        _, queries, truth = _dataset(conn)
        _drop_indexes(conn)
        if kind == "exact":
            yield {"status": "Quét toàn bảng (sequential scan)..."}
            recall, ms = _run(conn, queries, truth)
            yield {"row": {"kind": "exact", "param": "-", "recall": recall, "ms": ms},
                   "plan": plan_nodes(conn, SEARCH_SQL, (queries[0], K))}
        else:
            yield {"status": f"Đang build chỉ mục {kind.upper()}..."}
            yield {"build": {"kind": kind, **_build(conn, kind)}}
            setting, values = ("ivfflat.probes", IVF_PROBES) if kind == "ivfflat" else ("hnsw.ef_search", HNSW_EF)
            for value in values:
                conn.execute(f"SET {setting} = {value}")
                recall, ms = _run(conn, queries, truth)
                yield {"row": {"kind": kind, "param": f"{setting.split('.')[1]}={value}", "recall": recall, "ms": ms}}
            yield {"plan": plan_nodes(conn, SEARCH_SQL, (queries[0], K))}
        yield {"done": True, "kind": kind}
    finally:
        if conn is not None:
            conn.close()
        _busy.release()


def selective_filter() -> dict:
    """HNSW + WHERE chọn lọc (1% dữ liệu): mặc định trả thiếu, iterative scan hoặc lọc trước thì đủ (demo 08 B)."""
    if not _busy.acquire(blocking=False):
        return {"error": "Đang có một lần chạy ANN khác, hãy đợi nó xong."}
    conn = None
    try:
        conn = connect()
        vectors, _, _ = _dataset(conn)
        _drop_indexes(conn)
        conn.execute("CREATE INDEX ann_bench_hnsw ON ann_bench USING hnsw (embedding vector_cosine_ops) WITH (m = 16, ef_construction = 64)")
        tenant, k = 7, 10
        mask = np.arange(len(vectors)) % 100 == tenant
        ids = np.where(mask)[0]
        queries = make_queries(vectors, 20)
        truth = [set(ids[exact_top_k(vectors[mask], q, k)].tolist()) for q in queries]
        plain = "SELECT id FROM ann_bench WHERE tenant = %s ORDER BY embedding <=> %s LIMIT %s"
        pre = ("WITH f AS MATERIALIZED (SELECT id, embedding FROM ann_bench WHERE tenant = %s) "
               "SELECT id FROM f ORDER BY embedding <=> %s LIMIT %s")
        conn.execute("SET hnsw.ef_search = 40")
        rows = []
        for label, sql, setup in (
            ("HNSW + WHERE (mặc định, ef_search=40)", plain, "SET hnsw.iterative_scan = off"),
            ("HNSW + WHERE + iterative_scan", plain, "SET hnsw.iterative_scan = relaxed_order"),
            ("Lọc TRƯỚC (CTE) rồi exact search trong tập nhỏ", pre, "SET hnsw.iterative_scan = off"),
        ):
            conn.execute(setup)
            got, rec = [], []
            for q, t in zip(queries, truth):
                r = {x[0] for x in conn.execute(sql, (tenant, q, k)).fetchall()}
                got.append(len(r)); rec.append(len(r & t) / k)
            rows.append({"label": label, "returned": round(float(np.mean(got)), 1), "recall": round(float(np.mean(rec) * 100))})
        return {"tenant": tenant, "k": k, "rows": rows, "matching_rows": int(mask.sum())}
    finally:
        if conn is not None:
            conn.close()
        _busy.release()
