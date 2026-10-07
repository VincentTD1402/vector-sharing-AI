"""Demo 07 - ANN index trong pgvector: HNSW vs IVFFlat vs không index (slide 13, 14, 15).

Dữ liệu: 50.000 vector 384 chiều (tổng hợp, có cấu trúc cụm) trong bảng ann_bench.
Với mỗi cấu hình đo: recall@10 so với kết quả exact, latency/query, kích thước index.
Tham số điều chỉnh lúc query: hnsw.ef_search, ivfflat.probes.

Cần: docker compose up -d
Chạy:  python demos/07_pgvector_hnsw_ivfflat.py
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from lib.ann_bench import ensure_ann_bench_table, exact_top_k, make_queries
from lib.db import connect, plan_nodes
from lib.viz import banner

K = 10
N = 50_000
N_QUERIES = 30
SEARCH_SQL = "SELECT id FROM ann_bench ORDER BY embedding <=> %s LIMIT %s"


def run_queries(conn, queries, truth, setting_sql: str | None = None) -> tuple[float, float]:
    """Trả về (recall@10 %, ms/query)."""
    if setting_sql:
        conn.execute(setting_sql)
    recalls, times = [], []
    for q, t in zip(queries, truth):
        start = time.perf_counter()
        ids = {r[0] for r in conn.execute(SEARCH_SQL, (q, K)).fetchall()}
        times.append((time.perf_counter() - start) * 1000)
        recalls.append(len(ids & t) / K)
    return float(np.mean(recalls) * 100), float(np.mean(times))


def index_size_mb(conn, name: str) -> float:
    return conn.execute("SELECT pg_relation_size(%s::regclass)", (name,)).fetchone()[0] / 1e6


def build(conn, name: str, ddl: str) -> float:
    conn.execute(f"DROP INDEX IF EXISTS {name}")
    start = time.perf_counter()
    conn.execute(ddl)
    return time.perf_counter() - start


def main() -> None:
    conn = connect()
    vectors = ensure_ann_bench_table(conn, N)
    queries = make_queries(vectors, N_QUERIES)
    truth = [set(exact_top_k(vectors, q, K).tolist()) for q in queries]
    table_mb = conn.execute("SELECT pg_relation_size('ann_bench')").fetchone()[0] / 1e6
    print(f"Bảng ann_bench: {N:,} vector x 384 chiều, dữ liệu {table_mb:.0f} MB. {N_QUERIES} query, k={K}.")
    for idx in ("ann_bench_hnsw", "ann_bench_ivf"):
        conn.execute(f"DROP INDEX IF EXISTS {idx}")

    banner("1. Không có index: sequential scan = exact search", "12")
    recall, ms = run_queries(conn, queries, truth)
    print(f"  recall = {recall:.0f}%   latency = {ms:.1f} ms/query")
    print("  EXPLAIN:", plan_nodes(conn, SEARCH_SQL, (queries[0], K)))

    banner("2. IVFFlat (lists=100): chia cụm bằng K-means, quét `probes` cụm", "14")
    secs = build(conn, "ann_bench_ivf",
                 "CREATE INDEX ann_bench_ivf ON ann_bench USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100)")
    print(f"  build: {secs:.1f}s   kích thước index: {index_size_mb(conn, 'ann_bench_ivf'):.0f} MB")
    print(f"  {'probes':>7} {'recall@10':>10} {'ms/query':>9}")
    for probes in (1, 5, 10, 20, 100):
        recall, ms = run_queries(conn, queries, truth, f"SET ivfflat.probes = {probes}")
        print(f"  {probes:>7} {recall:>9.1f}% {ms:>9.2f}{'   <- = exact' if probes == 100 else ''}")

    banner("3. HNSW (m=16, ef_construction=64): đồ thị nhiều tầng, quét `ef_search` ứng viên", "13")
    conn.execute("DROP INDEX ann_bench_ivf")
    secs = build(conn, "ann_bench_hnsw",
                 "CREATE INDEX ann_bench_hnsw ON ann_bench USING hnsw (embedding vector_cosine_ops) "
                 "WITH (m = 16, ef_construction = 64)")
    print(f"  build: {secs:.1f}s   kích thước index: {index_size_mb(conn, 'ann_bench_hnsw'):.0f} MB")
    print(f"  {'ef_search':>9} {'recall@10':>10} {'ms/query':>9}")
    for ef in (10, 20, 40, 100, 400):
        recall, ms = run_queries(conn, queries, truth, f"SET hnsw.ef_search = {ef}")
        print(f"  {ef:>9} {recall:>9.1f}% {ms:>9.2f}")
    print("  EXPLAIN:", plan_nodes(conn, SEARCH_SQL, (queries[0], K)))

    print("\nĐọc kết quả: HNSW không cần training và thêm dữ liệu động tốt, nhưng index lớn hơn/build chậm hơn.")
    print("IVFFlat nhỏ và build nhanh, nhưng phải train trên dữ liệu có sẵn và cần chỉnh probes.")
    print("Dữ liệu tổng hợp, 50K vector: số tuyệt đối chỉ để minh họa xu hướng. Hãy benchmark trên dữ liệu của bạn.")
    print("(Index HNSW được giữ lại cho demo 08.)")


if __name__ == "__main__":
    main()
