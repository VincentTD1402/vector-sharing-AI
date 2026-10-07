"""Demo 08 - Metadata filtering: pre-filter vs post-filter, multi-tenant isolation (slide 16, 17).

A. Corpus thật (3 tenant, có doc lỗi thời): không filter -> RÒ RỈ dữ liệu tenant khác;
   post-filter -> thiếu kết quả; pre-filter (WHERE) -> đúng và đủ k.
B. Quy mô lớn (50K vector, tenant chiếm 1%): chỉ mục HNSW + WHERE có thể trả THIẾU hàng,
   vì HNSW duyệt `ef_search` ứng viên rồi mới lọc. Cách khắc phục: iterative scan, hoặc
   lọc trước rồi exact search trên tập nhỏ.

Cần: docker compose up -d   (và nên chạy demo 07 trước để có chỉ mục HNSW sẵn)
Chạy:  python demos/08_metadata_filtering.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from lib.ann_bench import ensure_ann_bench_table, exact_top_k, make_queries
from lib.db import connect, index_corpus, plan_nodes, vector_search
from lib.models import embed_queries
from lib.viz import banner

TENANT = "company_a"


def fmt(row: dict) -> str:
    leak = "  <-- RÒ RỈ: tenant khác!" if row["tenant_id"] != TENANT else ""
    stale = "  <-- tài liệu lỗi thời" if row["status"] != "current" else ""
    return (f"    {row['id']} [{row['tenant_id']}/{row['language']}/{row['status']}] "
            f"{row['score']:.3f}  {row['content'][:52]}{leak}{stale}")


def part_a_real_corpus(conn) -> None:
    banner("A. Corpus thật: không filter / post-filter / pre-filter", "16, 17")
    n = index_corpus(conn)
    print(f"  {n} chunk trong pgvector; user thuộc tenant '{TENANT}', chỉ được xem tài liệu 'current'.")
    for query, k in (("How do I approve an invoice?", 3), ("I forgot my password", 4)):
        qv = embed_queries([query])[0]
        print(f"\n  Query: {query!r}  (k={k})")

        print("  1) KHÔNG filter - top-k thuần theo similarity:")
        for r in vector_search(conn, qv, k, allow_cross_tenant=True):  # cố ý: minh họa rò rỉ
            print(fmt(r))

        print(f"  2) POST-filter - lấy top-{k} rồi mới lọc:")
        kept = [r for r in vector_search(conn, qv, k, allow_cross_tenant=True) if r["tenant_id"] == TENANT and r["status"] == "current"]
        for r in kept:
            print(fmt(r))
        print(f"     -> chỉ còn {len(kept)}/{k} kết quả (doc đúng của tenant có thể nằm ngoài top-{k}).")

        print("  3) PRE-filter - WHERE tenant_id AND status (bảng nhỏ -> lọc trước rồi xếp hạng chính xác):")
        full = vector_search(conn, qv, k, tenant_id=TENANT, status="current")
        for r in full:
            print(fmt(r))
        print(f"     -> đủ {len(full)}/{k}, không rò rỉ, không lỗi thời.")
    print("\n  Quy tắc: isolation là ràng buộc bắt buộc - ép tenant_id ở tầng API/DB, không để similarity quyết định.")


def part_b_selective_filter(conn) -> None:
    banner("B. Filter chọn lọc (1% dữ liệu) + chỉ mục HNSW: coi chừng trả thiếu kết quả", "17")
    version = conn.execute("SELECT extversion FROM pg_extension WHERE extname='vector'").fetchone()[0]
    print(f"  pgvector {version}. Bảng ann_bench 50K vector, cột tenant có 100 giá trị -> mỗi tenant ~500 hàng.")
    vectors = ensure_ann_bench_table(conn, 50_000)
    conn.execute("DROP INDEX IF EXISTS ann_bench_ivf")
    conn.execute("DROP INDEX IF EXISTS ann_bench_tenant")
    conn.execute("CREATE INDEX IF NOT EXISTS ann_bench_hnsw ON ann_bench USING hnsw "
                 "(embedding vector_cosine_ops) WITH (m = 16, ef_construction = 64)")
    conn.execute("SET hnsw.ef_search = 40")  # mặc định của pgvector
    conn.execute("SET hnsw.iterative_scan = off")

    tenant, k = 7, 10
    in_tenant = np.arange(len(vectors)) % 100 == tenant
    subset_ids = np.where(in_tenant)[0]
    queries = make_queries(vectors, 20)
    truth = [set(subset_ids[exact_top_k(vectors[in_tenant], q, k)].tolist()) for q in queries]

    plain = "SELECT id FROM ann_bench WHERE tenant = %s ORDER BY embedding <=> %s LIMIT %s"
    prefilter = ("WITH f AS MATERIALIZED (SELECT id, embedding FROM ann_bench WHERE tenant = %s) "
                 "SELECT id FROM f ORDER BY embedding <=> %s LIMIT %s")
    print(f"  Mỗi query xin k={k} kết quả, WHERE tenant={tenant}. Recall so với exact search trong tenant.\n")
    print(f"  {'Cách làm':<52} {'hàng trả về':>11} {'recall':>7}")

    def run(label: str, sql: str, setup: str | None = None) -> None:
        if setup:
            conn.execute(setup)
        rows_returned, recalls = [], []
        for q, t in zip(queries, truth):
            ids = {r[0] for r in conn.execute(sql, (tenant, q, k)).fetchall()}
            rows_returned.append(len(ids))
            recalls.append(len(ids & t) / k)
        print(f"  {label:<52} {np.mean(rows_returned):>8.1f}/{k} {np.mean(recalls) * 100:>6.0f}%")

    run("HNSW + WHERE (ef_search=40, mặc định)", plain)
    print(f"      plan: {plan_nodes(conn, plain, (tenant, queries[0], k))}")
    run("HNSW + WHERE + iterative_scan = relaxed_order", plain, "SET hnsw.iterative_scan = relaxed_order")
    conn.execute("SET hnsw.iterative_scan = off")
    run("Lọc TRƯỚC (CTE) rồi exact search trong tập nhỏ", prefilter)

    print("\n  Đọc kết quả: HNSW mặc định chỉ xét ~40 ứng viên TOÀN BẢNG rồi mới lọc theo tenant -> hầu hết bị loại,")
    print("  trả thiếu (có thể 0 hàng) dù tenant có đủ dữ liệu. Đây là lỗi im lặng, không báo error.")
    print("  Cách xử lý: iterative scan (pgvector >= 0.8), exact search trên tập đã lọc nhỏ,")
    print("  hoặc tách partition / chỉ mục riêng theo tenant (partial index, table partitioning).")


if __name__ == "__main__":
    connection = connect()
    part_a_real_corpus(connection)
    part_b_selective_filter(connection)
