"""Dữ liệu vector tổng hợp (có cấu trúc cụm) để demo ANN: IVF tự viết và pgvector.

Vector ngẫu nhiên hoàn toàn không có cấu trúc nên ANN gần như vô dụng; embedding
thật thì có cụm theo chủ đề. Vì vậy ta sinh dữ liệu dạng hỗn hợp Gaussian.
"""
import numpy as np
import psycopg

from lib.config import EMBED_DIM


def make_clustered_vectors(
    n: int, dim: int = EMBED_DIM, n_clusters: int = 10, latent_dim: int = 32,
    spread: float = 1.2, noise: float = 0.1, seed: int = 0,
) -> np.ndarray:
    """n vector đã normalize: vài cụm lớn, trong mỗi cụm phân bố liên tục trên một
    không gian con `latent_dim` chiều (giống embedding thật: có cấu trúc nhưng không
    tách cụm sạch). Cụm quá tách bạch thì ANN dễ đến mức recall luôn 100%."""
    rng = np.random.default_rng(seed)
    centers = rng.normal(size=(n_clusters, dim)).astype(np.float32)
    basis = rng.normal(size=(latent_dim, dim)).astype(np.float32) / np.sqrt(latent_dim)
    assign = rng.integers(0, n_clusters, size=n)
    latent = rng.normal(size=(n, latent_dim)).astype(np.float32)
    vecs = centers[assign] + spread * (latent @ basis) + noise * rng.normal(size=(n, dim)).astype(np.float32)
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
    return vecs


def make_queries(vectors: np.ndarray, n: int, noise: float = 0.35, seed: int = 99) -> np.ndarray:
    """Query giống thực tế: nằm gần một vùng có dữ liệu (một vector có sẵn + nhiễu)."""
    rng = np.random.default_rng(seed)
    base = vectors[rng.integers(0, len(vectors), size=n)]
    q = base + noise * rng.normal(size=base.shape).astype(np.float32) / np.sqrt(vectors.shape[1])
    return (q / np.linalg.norm(q, axis=1, keepdims=True)).astype(np.float32)


def exact_top_k(vectors: np.ndarray, query: np.ndarray, k: int) -> np.ndarray:
    """Brute force: so với TẤT CẢ vector (O(N x D)). Dùng làm ground truth."""
    scores = vectors @ query
    top = np.argpartition(-scores, k)[:k]
    return top[np.argsort(-scores[top])]


def _table_matches(conn: psycopg.Connection, vectors: np.ndarray) -> bool:
    """Bảng đã có đúng dữ liệu này chưa? So số dòng VÀ nội dung vector đầu/cuối, để lần
    sau đổi cách sinh dữ liệu (cùng n) không bị dùng nhầm bảng cũ."""
    if not conn.execute("SELECT to_regclass('ann_bench')").fetchone()[0]:
        return False
    if conn.execute("SELECT count(*) FROM ann_bench").fetchone()[0] != len(vectors):
        return False
    for row_id in (0, len(vectors) - 1):
        row = conn.execute("SELECT embedding FROM ann_bench WHERE id = %s", (row_id,)).fetchone()
        if row is None or not np.allclose(row[0].to_numpy(), vectors[row_id], atol=1e-6):
            return False
    return True


def ensure_ann_bench_table(conn: psycopg.Connection, n: int = 50_000) -> np.ndarray:
    """Tạo bảng ann_bench (n vector, cột tenant 100 giá trị) nếu chưa có. Trả về vectors."""
    vectors = make_clustered_vectors(n)
    if _table_matches(conn, vectors):
        return vectors  # sinh lại cùng seed nên giống hệt dữ liệu đã nạp
    conn.execute("DROP TABLE IF EXISTS ann_bench")
    conn.execute(f"CREATE TABLE ann_bench (id int PRIMARY KEY, tenant int, embedding vector({EMBED_DIM}))")
    print(f"Nạp {n:,} vector vào ann_bench (COPY)...")
    with conn.cursor() as cur, cur.copy("COPY ann_bench (id, tenant, embedding) FROM STDIN") as copy:
        copy.set_types(["int4", "int4", "vector"])
        for i in range(n):
            copy.write_row((i, i % 100, vectors[i]))
    conn.execute("ANALYZE ann_bench")
    return vectors
