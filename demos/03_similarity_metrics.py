"""Demo 03 - Similarity metrics: cosine, dot product, euclidean (slide 10).

Ý chính: với vector đã normalize (độ dài 1) cả 3 cho CÙNG thứ hạng.
Chỉ khi độ lớn vector có ý nghĩa/khác nhau thì chúng mới tách nhau ra.

Chạy:  python demos/03_similarity_metrics.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from lib.models import embed_passages, embed_queries
from lib.viz import banner

QUERY = "How do I reset my password?"
DOCS = [
    "How to reset your password",
    "Change account password",
    "Update user profile",
    "Create production order",
]


def cosine(a, b): return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))
def dot(a, b): return float(a @ b)
def euclid(a, b): return float(np.linalg.norm(a - b))


def table(title: str, docs_vec: np.ndarray, q: np.ndarray) -> None:
    print(f"\n{title}")
    print(f"  {'Document':<30} {'cosine':>8} {'dot':>8} {'euclid':>8} | rank(cos/dot/euc)")
    cos = [cosine(q, v) for v in docs_vec]
    dots = [dot(q, v) for v in docs_vec]
    euc = [euclid(q, v) for v in docs_vec]
    rank = lambda scores, rev: {i: r + 1 for r, i in enumerate(sorted(range(len(scores)), key=lambda i: scores[i], reverse=rev))}
    rc, rd, re_ = rank(cos, True), rank(dots, True), rank(euc, False)  # euclid: nhỏ hơn = gần hơn
    for i, d in enumerate(DOCS):
        print(f"  {d:<30} {cos[i]:>8.3f} {dots[i]:>8.3f} {euc[i]:>8.3f} |   #{rc[i]} / #{rd[i]} / #{re_[i]}")


def main() -> None:
    banner("A. Vector đã normalize: 3 metric cho cùng thứ hạng", "10")
    q = embed_queries([QUERY])[0]
    docs = embed_passages(DOCS)
    print(f"Query: {QUERY!r}")
    table("Bảng điểm (norm = 1):", docs, q)

    # Quan hệ toán học khi norm = 1:  euclid^2 = 2 - 2*cosine
    cos0, euc0 = cosine(q, docs[0]), euclid(q, docs[0])
    print(f"\n  Kiểm chứng: euclid^2 = {euc0**2:.4f}  vs  2 - 2*cosine = {2 - 2 * cos0:.4f}")

    banner("B. Độ lớn vector khác nhau: dot product bị 'đánh lừa', cosine thì không", "10")
    scaled = docs.copy()
    scaled[3] *= 5.0  # doc KHÔNG liên quan, nhưng vector bị phóng to gấp 5
    table("Doc cuối (Create production order) bị nhân 5 độ lớn:", scaled, q)
    print("\n  -> cosine bỏ qua độ lớn nên thứ hạng giữ nguyên; dot/euclid thay đổi.")
    print("  Quy tắc: metric phải khớp cách model được huấn luyện (e5/BGE: cosine; một số model: dot).")
    print("  Với vector đã normalize (như demo này) chọn cosine hay dot đều như nhau - dot nhanh hơn một chút.")

    banner("C. Cosine trong SQL (pgvector) - toán tử `<=>` là cosine DISTANCE", "10, 15")
    print("  cosine_distance = 1 - cosine_similarity")
    print("  -> ORDER BY embedding <=> query  (nhỏ nhất trước)  ;  score = 1 - (embedding <=> query)")
    print("  Các toán tử khác: <-> euclid (L2),  <#> negative inner product.")


if __name__ == "__main__":
    main()
