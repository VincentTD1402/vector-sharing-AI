"""pgvector: kết nối, schema, indexing và vector search có metadata filter."""
import json
import re

import numpy as np
import psycopg
from pgvector.psycopg import register_vector

from lib.config import DATABASE_URL, EMBED_DIM

SCHEMA_SQL = f"""
CREATE EXTENSION IF NOT EXISTS vector;
CREATE TABLE IF NOT EXISTS document_chunks (
    id          TEXT PRIMARY KEY,
    document_id TEXT NOT NULL,
    content     TEXT NOT NULL,
    embedding   vector({EMBED_DIM}) NOT NULL,
    tenant_id   TEXT NOT NULL,
    language    TEXT NOT NULL,
    department  TEXT,
    status      TEXT NOT NULL DEFAULT 'current',
    metadata    JSONB DEFAULT '{{}}'
);
-- Chỉ mục ANN cho vector (cosine) + chỉ mục btree cho metadata filter.
CREATE INDEX IF NOT EXISTS document_chunks_embedding_hnsw
    ON document_chunks USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);
CREATE INDEX IF NOT EXISTS document_chunks_tenant_status
    ON document_chunks (tenant_id, status);
"""


def connect() -> psycopg.Connection:
    """Kết nối + đăng ký kiểu vector. Báo lỗi dễ hiểu nếu chưa bật Docker."""
    try:
        conn = psycopg.connect(DATABASE_URL, autocommit=True)
    except psycopg.OperationalError as exc:
        raise SystemExit(
            "Không kết nối được PostgreSQL. Chạy `docker compose up -d` trước.\n"
            f"URL: {DATABASE_URL}\nChi tiết: {exc}"
        )
    conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
    register_vector(conn)
    return conn


def init_schema(conn: psycopg.Connection) -> None:
    conn.execute(SCHEMA_SQL)


def count_chunks(conn: psycopg.Connection) -> int:
    return conn.execute("SELECT count(*) FROM document_chunks").fetchone()[0]


def upsert_chunks(conn: psycopg.Connection, docs: list[dict], vectors: np.ndarray) -> None:
    """Upsert (không insert) -> chạy lại không tạo bản trùng (idempotent indexing)."""
    rows = [
        (d["id"], d.get("document_id", d["id"]), d["text"], v, d["tenant_id"],
         d["language"], d.get("department"), d["status"],
         json.dumps({"topic": d.get("topic")}))
        for d, v in zip(docs, vectors)
    ]
    with conn.cursor() as cur:
        cur.executemany(
            """INSERT INTO document_chunks
                 (id, document_id, content, embedding, tenant_id, language, department, status, metadata)
               VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)
               ON CONFLICT (id) DO UPDATE SET
                 content = EXCLUDED.content, embedding = EXCLUDED.embedding,
                 tenant_id = EXCLUDED.tenant_id, language = EXCLUDED.language,
                 department = EXCLUDED.department, status = EXCLUDED.status,
                 metadata = EXCLUDED.metadata""",
            rows,
        )


def plan_nodes(conn: psycopg.Connection, sql: str, params: tuple) -> str:
    """Tóm tắt kế hoạch thực thi (EXPLAIN) thành tên các node, bỏ literal vector dài."""
    text = "\n".join(r[0] for r in conn.execute("EXPLAIN " + sql, params).fetchall())
    nodes = re.findall(r"(Seq Scan on \w+|Index Scan using \w+ on \w+|Bitmap Heap Scan on \w+|Sort(?= {2}\()|Limit(?= {2}\())", text)
    return " -> ".join(nodes)


def index_corpus(conn: psycopg.Connection) -> int:
    """Pipeline indexing đầy đủ: load -> embed (batch) -> upsert vào pgvector. Trả về số chunk."""
    from lib.data import load_corpus
    from lib.models import embed_passages_cached

    init_schema(conn)
    docs = load_corpus()
    upsert_chunks(conn, docs, embed_passages_cached([d["text"] for d in docs]))
    return count_chunks(conn)


def vector_search(
    conn: psycopg.Connection,
    query_vec: np.ndarray,
    k: int = 10,
    tenant_id: str = "",
    status: str | None = None,
    languages: list[str] | None = None,
    department: str | None = None,
    allow_cross_tenant: bool = False,
) -> list[dict]:
    """Top-k theo cosine với filter trong WHERE. Bảng nhỏ/planner chọn seq scan: lọc trước rồi
    xếp hạng chính xác. Nhưng khi planner dùng chỉ mục HNSW, filter áp dụng SAU khi duyệt ~ef_search
    ứng viên nên có thể trả thiếu hàng (xem demo 08 phần B)."""
    if not tenant_id and not allow_cross_tenant:  # fail-closed: không được lặng lẽ bỏ filter
        raise ValueError("tenant_id là bắt buộc (allow_cross_tenant chỉ dành cho demo minh họa rò rỉ)")
    where, params = [], []
    if tenant_id:
        where.append("tenant_id = %s"); params.append(tenant_id)
    if status:
        where.append("status = %s"); params.append(status)
    if languages:
        where.append("language = ANY(%s)"); params.append(languages)
    if department:
        where.append("department = %s"); params.append(department)
    where_sql = ("WHERE " + " AND ".join(where)) if where else ""
    sql = f"""SELECT id, content, tenant_id, language, status, department,
                     1 - (embedding <=> %s) AS score
              FROM document_chunks {where_sql}
              ORDER BY embedding <=> %s LIMIT %s"""
    rows = conn.execute(sql, [query_vec, *params, query_vec, k]).fetchall()
    cols = ("id", "content", "tenant_id", "language", "status", "department", "score")
    return [dict(zip(cols, r)) for r in rows]
