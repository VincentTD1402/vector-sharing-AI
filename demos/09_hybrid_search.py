"""Demo 09 - Sparse (BM25) vs Dense vs Hybrid + Reciprocal Rank Fusion (slide 18, 19).

A. Sparse vs dense vector: số chiều khác 0, thứ chúng nắm bắt.
B. Recall@3 theo từng LOẠI query (mã chính xác / tự nhiên / paraphrase) cho keyword, vector, hybrid.
C. Đi từng bước RRF trên một query cụ thể.

Chạy:  python demos/09_hybrid_search.py
"""
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from lib.data import active_docs, load_eval_queries
from lib.search import InMemoryRetriever, recall_at_k, reciprocal_rank, rrf, tokenize
from lib.viz import banner

K = 3


def sparse_vs_dense(r: InMemoryRetriever) -> None:
    banner("A. Sparse vector (từ vựng) vs Dense vector (ngữ nghĩa)", "18")
    vocab = sorted({t for tokens in r.bm25.doc_freqs for t in tokens})
    query = "ERR_CONNECTION_RESET khi dùng VPN"
    tokens = tokenize(query)
    weights = {t: round(r.bm25.idf.get(t, 0.0), 2) for t in tokens}
    print(f"  Sparse: từ điển {len(vocab)} từ -> vector dài {len(vocab)}, query này chỉ {len(tokens)} chiều khác 0:")
    print(f"          {weights}   (giá trị = IDF: từ hiếm -> nặng; từ ngoài từ điển = 0.0)")
    print(f"  Dense : {r.vectors.shape[1]} chiều, MỌI chiều đều có giá trị (không đọc được từng chiều).")
    print("  Sparse nắm bắt 'dùng từ gì chính xác'; dense nắm bắt 'ý định là gì'. Chúng bổ sung cho nhau.")


def by_kind(r: InMemoryRetriever) -> None:
    banner(f"B. Recall@{K} theo loại query ({len(load_eval_queries())} query, tenant company_a)", "19")
    methods = {"keyword": r.keyword, "vector": r.vector, "hybrid": r.hybrid}
    recalls = {m: defaultdict(list) for m in methods}
    for q in load_eval_queries():
        for name, fn in methods.items():
            ids = [i for i, _ in fn(q["query"], 10)]
            recalls[name][q["kind"]].append(recall_at_k(ids, q["relevant"], K))
            recalls[name]["ALL"].append(recall_at_k(ids, q["relevant"], K))
    kinds = ["exact", "natural", "paraphrase", "ALL"]
    counts = {k: sum(1 for q in load_eval_queries() if q["kind"] == k) for k in kinds[:-1]}
    counts["ALL"] = len(load_eval_queries())
    print(f"  {'loại query':<18}" + "".join(f"{m:>10}" for m in methods))
    for kind in kinds:
        row = "".join(f"{np.mean(recalls[m][kind]):>10.2f}" for m in methods)
        print(f"  {kind + f' (n={counts[kind]})':<18}{row}")
    print("  exact = mã lỗi/SKU/tên màn hình; natural = câu hỏi đầy đủ; paraphrase = diễn đạt khác từ trong doc.")
    overall = {m: float(np.mean(recalls[m]["ALL"])) for m in methods}
    print(f"\n  Số đo thật trên bộ này: keyword {overall['keyword']:.2f} | vector {overall['vector']:.2f} | hybrid {overall['hybrid']:.2f}")
    if overall["hybrid"] < max(overall["keyword"], overall["vector"]):
        print("  -> Hybrid KHÔNG tự động thắng. RRF cộng thứ hạng đồng đều nên phương pháp yếu ở một loại query")
        print("     (keyword với paraphrase) kéo nhiễu vào; và khi doc đúng chỉ đứng cao ở MỘT danh sách,")
        print("     nó có thể thua doc trung bình đứng ở cả hai. Cần đo, chỉnh trọng số RRF, hoặc route theo loại query.")
    disagreements(r)


def disagreements(r: InMemoryRetriever) -> None:
    """Liệt kê query mà 3 phương pháp cho Recall@3 khác nhau - chỗ đáng bàn nhất khi giảng."""
    print(f"\n  Các query mà các phương pháp BẤT ĐỒNG về Recall@{K} (tối đa 8):")
    shown = 0
    for q in load_eval_queries():
        tops = {n: [i for i, s in fn(q["query"], 10) if n != "keyword" or s > 0][:K]
                for n, fn in (("keyword", r.keyword), ("vector", r.vector), ("hybrid", r.hybrid))}
        rec = {n: recall_at_k(t, q["relevant"], K) for n, t in tops.items()}
        if len(set(rec.values())) == 1 or shown >= 8:
            continue
        shown += 1
        print(f"    {q['query'][:42]!r:<46} đúng={q['relevant']}")
        for n, t in tops.items():
            print(f"        {n:<8} recall={rec[n]:.1f}  top-{K}={t}")


def rrf_walkthrough(r: InMemoryRetriever) -> None:
    banner("C. Đi từng bước RRF: score(d) = sum 1/(60 + rank)", "19")
    # Chọn query mà hybrid xếp doc đúng cao hơn cả hai phương pháp riêng lẻ (nếu có).
    chosen = None
    for q in load_eval_queries():
        rr = {n: reciprocal_rank([i for i, _ in fn(q["query"], 10)], q["relevant"])
              for n, fn in (("kw", r.keyword), ("vec", r.vector), ("hyb", r.hybrid))}
        if rr["hyb"] > max(rr["kw"], rr["vec"]):
            chosen = q
            break
    if chosen is None:
        chosen = next(q for q in load_eval_queries() if q["id"] == "q01")
        print("  (Không có query nào mà hybrid xếp doc đúng cao hơn CẢ hai phương pháp riêng lẻ trên bộ này;")
        print("   dùng q01 chỉ để minh họa cơ chế cộng điểm.)")
    print(f"  Query: {chosen['query']!r}   đáp án: {chosen['relevant']}")
    kw = [i for i, _ in r.keyword(chosen["query"], 20) if _ > 0][:5]
    vec = [i for i, _ in r.vector(chosen["query"], 20)][:5]
    print(f"  BM25 top-5  : {kw}")
    print(f"  Vector top-5: {vec}")
    print(f"  {'doc':<5} {'rank BM25':>9} {'rank vec':>9} {'1/(60+r1)':>10} {'1/(60+r2)':>10} {'RRF':>8}")
    for doc_id, score in rrf([kw, vec])[:6]:
        r1 = kw.index(doc_id) + 1 if doc_id in kw else None
        r2 = vec.index(doc_id) + 1 if doc_id in vec else None
        f1 = f"{1 / (60 + r1):.4f}" if r1 else "-"
        f2 = f"{1 / (60 + r2):.4f}" if r2 else "-"
        mark = " <- đúng" if doc_id in chosen["relevant"] else ""
        print(f"  {doc_id:<5} {r1 or '-':>9} {r2 or '-':>9} {f1:>10} {f2:>10} {score:>8.4f}{mark}")
    print("  RRF chỉ dùng THỨ HẠNG nên không cần chuẩn hóa điểm BM25 (~0-15) với cosine (~0.7-1.0).")
    print("  Doc xuất hiện ở cả hai danh sách được cộng dồn -> lên đầu.")


if __name__ == "__main__":
    retriever = InMemoryRetriever(active_docs("company_a"))
    sparse_vs_dense(retriever)
    by_kind(retriever)
    rrf_walkthrough(retriever)
