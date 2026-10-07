"""Demo 12 - Query transformation: rewrite, expansion, multi-query (slide 27).

Query người dùng thường ngắn/viết tắt -> embedding kém đại diện. Ta biến đổi query TRƯỚC khi search:
  - rewrite   : mở viết tắt ("reset pw" -> "reset password")
  - expansion : thêm từ đồng nghĩa + từ tiếng Việt
  - multi-query: search bằng nhiều biến thể rồi gộp bằng RRF
Ở đây dùng từ điển viết tay để chạy offline và minh bạch. Thực tế dùng LLM để viết lại -
cùng cấu trúc, chỉ thay hàm `rewrite/expand`.

HyDE (sinh tài liệu giả định rồi embed) cần LLM nên KHÔNG có trong demo offline này.

Chạy:  python demos/12_query_transformation.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from lib.data import active_docs
from lib.query_transform import TERSE_QUERIES, expand, rewrite
from lib.search import InMemoryRetriever, recall_at_k, reciprocal_rank, rrf
from lib.viz import banner

K = 3
TERSE = [(q["query"], q["relevant"]) for q in TERSE_QUERIES]


def search_variants(r: InMemoryRetriever, variants: list[str]) -> list[str]:
    """Multi-query: search từng biến thể rồi gộp bằng RRF."""
    return [i for i, _ in rrf([[i for i, _ in r.vector(v, 10)] for v in variants])]


def main() -> None:
    r = InMemoryRetriever(active_docs("company_a"))
    banner("Query ngắn/viết tắt: so sánh các kỹ thuật biến đổi (vector search)", "27")
    strategies = {
        "gốc": lambda q: [i for i, _ in r.vector(q, 10)],
        "rewrite": lambda q: [i for i, _ in r.vector(rewrite(q), 10)],
        "rewrite + expansion": lambda q: [i for i, _ in r.vector(expand(q), 10)],
        "multi-query (gốc+rewrite+expand)": lambda q: search_variants(r, [q, rewrite(q), expand(q)]),
    }
    print(f"  {'query':<16} {'rewrite':<28} {'expansion'}")
    for q, _ in TERSE[:3]:
        print(f"  {q:<16} {rewrite(q):<28} {expand(q)}")

    print(f"\n  {'kỹ thuật':<34} {'Recall@'+str(K):>9} {'MRR':>6}")
    per_query = {}
    for name, fn in strategies.items():
        ranked = {q: fn(q) for q, _ in TERSE}
        per_query[name] = ranked
        rec = np.mean([recall_at_k(ranked[q], rel, K) for q, rel in TERSE])
        mrr = np.mean([reciprocal_rank(ranked[q], rel) for q, rel in TERSE])
        print(f"  {name:<34} {rec:>9.2f} {mrr:>6.2f}")

    print("\n  Chi tiết: doc đúng đầu tiên xếp hạng mấy (— = ngoài top-10):")
    print(f"  {'query':<16}" + "".join(f"{n[:18]:>20}" for n in strategies))
    for q, rel in TERSE:
        cells = []
        for name in strategies:
            ranked = per_query[name][q]
            pos = next((n for n, i in enumerate(ranked, 1) if i in rel), None)
            cells.append(f"{pos or '—':>20}")
        print(f"  {q:<16}" + "".join(cells))
    print("\n  Lưu ý: 8 query, từ điển do người viết SAU khi biết query nên số liệu lạc quan -> chỉ minh họa cơ chế.")
    print("  Expansion cũng có thể làm tệ đi (thêm từ nhiễu): luôn đo trước khi bật. Mỗi lần biến đổi tốn thêm 1 lần embed")
    print("  (multi-query: N lần) hoặc 1 lần gọi LLM -> cộng vào latency; chỉ dùng khi SLA cho phép.")


if __name__ == "__main__":
    main()
