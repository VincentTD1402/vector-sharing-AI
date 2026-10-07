"""Demo 04 - Keyword search vs Semantic search (slide 2, 11, 23).

Chạy cùng một query qua BM25 (từ khóa) và vector search (ngữ nghĩa), in top-3
để thấy mỗi phương pháp mạnh/yếu ở đâu. Đánh dấu [v] nếu doc nằm trong đáp án đúng.

Chạy:  python demos/04_semantic_vs_keyword.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lib.data import active_docs, load_eval_queries
from lib.search import InMemoryRetriever, tokenize
from lib.viz import banner

# (query, ghi chú giảng viên) - lấy từ bộ eval để có đáp án đúng.
SHOWCASE = {
    "q01": "câu hỏi tiếng Anh, đáp án có cả doc tiếng Việt (d03)",
    "q02": "câu hỏi tiếng Việt, đáp án có cả doc tiếng Anh (d01)",
    "q07": "mã lỗi chính xác",
    "q10": "mã SKU chính xác (có nhiều SKU na ná)",
    "q05": "câu hỏi tự nhiên",
}


def show(label: str, hits: list[tuple[str, float]], relevant: list[str], text_of: dict,
         drop_zero: bool = False) -> None:
    print(f"  {label}")
    # BM25 điểm 0 = KHÔNG khớp từ nào; chỉ là phần đệm cho đủ top-k, không phải kết quả.
    shown = [h for h in hits[:3] if not (drop_zero and h[1] <= 0)]
    for doc_id, score in shown:
        mark = "[v]" if doc_id in relevant else "[ ]"
        print(f"    {mark} {doc_id} {score:7.3f}  {text_of[doc_id][:70]}")
    if len(shown) < 3 and drop_zero:
        print("    (không còn doc nào khớp từ khóa)")


def main() -> None:
    docs = active_docs("company_a")
    retriever = InMemoryRetriever(docs)
    text_of = {d["id"]: d["text"] for d in docs}
    queries = {q["id"]: q for q in load_eval_queries()}

    banner(f"Keyword (BM25) vs Semantic (vector) trên {len(docs)} tài liệu", "11")
    for qid, note in SHOWCASE.items():
        q = queries[qid]
        print(f"\nQuery: {q['query']!r}   ({note})")
        for rel in q["relevant"]:  # doc đúng nào có từ chung với query?
            shared = sorted(set(tokenize(q["query"])) & set(tokenize(text_of[rel])))
            print(f"  Doc đúng {rel} - từ chung với query: {shared or 'KHÔNG CÓ (keyword không thể tìm ra)'}")
        show("Keyword (BM25):", retriever.keyword(q["query"]), q["relevant"], text_of, drop_zero=True)
        show("Semantic     :", retriever.vector(q["query"]), q["relevant"], text_of)

    print("\nĐọc kết quả: keyword chỉ tìm được doc trùng từ (không vượt qua ngôn ngữ/từ đồng nghĩa);")
    print("semantic tìm theo ngữ nghĩa nhưng có thể xếp doc cùng chủ đề (sai) lên trên doc đúng.")
    print("Mỗi cách có kiểu lỗi riêng -> hybrid search (demo 09). Đo trên toàn bộ bộ eval ở demo 11.")


if __name__ == "__main__":
    main()
