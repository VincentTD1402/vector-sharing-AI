"""Demo 11 - Đánh giá retrieval: Recall@K, Precision@K, MRR, latency (slide 21, 23, 25, 33).

A. So sánh 5 cấu hình retrieval trên bộ eval có nhãn (keyword, vector, hybrid, +rerank).
B. Chia theo loại query để thấy cấu hình nào mạnh ở đâu.
C. Debug một query thất bại: lỗi nằm ở RETRIEVAL (doc đúng không vào pool)
   hay ở RANKING (có trong pool nhưng xếp thấp)?

Chạy:  python demos/11_evaluation.py [--open]
"""
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import plotly.graph_objects as go

from lib.data import active_docs, load_eval_queries
from lib.search import InMemoryRetriever, precision_at_k, recall_at_k, reciprocal_rank, tokenize
from lib.viz import banner, save_figure

K, POOL = 3, 20


def configs(r: InMemoryRetriever) -> dict:
    """Mỗi cấu hình: query -> danh sách doc_id đã xếp hạng."""
    ids = lambda hits: [i for i, _ in hits]
    return {
        "keyword (BM25)": lambda q: ids(r.keyword(q, 10)),
        "vector": lambda q: ids(r.vector(q, 10)),
        "hybrid (RRF)": lambda q: ids(r.hybrid(q, 10)),
        "vector + rerank": lambda q: ids(r.rerank(q, r.vector(q, POOL), 10)),
        "hybrid + rerank": lambda q: ids(r.rerank(q, r.hybrid(q, POOL, pool=POOL), 10)),
    }


def evaluate(fn, queries: list[dict]) -> dict:
    rows = defaultdict(list)
    for q in queries:
        start = time.perf_counter()
        ranked = fn(q["query"])
        ms = (time.perf_counter() - start) * 1000
        recall = recall_at_k(ranked, q["relevant"], K)
        rows["recall"].append(recall)
        rows["precision"].append(precision_at_k(ranked, q["relevant"], K))
        rows["mrr"].append(reciprocal_rank(ranked, q["relevant"]))
        rows["ms"].append(ms)
        rows[f"recall:{q['kind']}"].append(recall)
    return {k: float(np.mean(v)) for k, v in rows.items()}


def compare(r: InMemoryRetriever) -> dict:
    queries = load_eval_queries()
    banner(f"A. {len(queries)} query có nhãn, corpus {len(r.docs)} doc (tenant company_a, status current)", "21, 23")
    cfgs = configs(r)
    for fn in cfgs.values():  # warm-up model để latency không tính thời gian nạp
        fn(queries[0]["query"])
    results = {name: evaluate(fn, queries) for name, fn in cfgs.items()}
    print(f"  {'cấu hình':<18} {'Recall@'+str(K):>9} {'Prec@'+str(K):>8} {'MRR':>6} {'latency':>10}")
    for name, m in results.items():
        print(f"  {name:<18} {m['recall']:>9.3f} {m['precision']:>8.3f} {m['mrr']:>6.3f} {m['ms']:>7.0f} ms")
    print(f"  (Precision@{K} bị chặn trên bởi số doc đúng/query: tối đa 0.67 với query có 2 đáp án.)")
    print("  Bộ eval chỉ 36 query: chênh lệch cỡ 0.03 là nhiễu. Chỉ tin xu hướng lớn.")

    banner(f"B. Recall@{K} theo loại query", "21, 23")
    kinds = ["exact", "natural", "paraphrase"]
    print(f"  {'cấu hình':<18}" + "".join(f"{k:>12}" for k in kinds))
    for name, m in results.items():
        print(f"  {name:<18}" + "".join(f"{m[f'recall:{k}']:>12.2f}" for k in kinds))
    return results


def chart(results: dict) -> None:
    names = list(results)
    fig = go.Figure()
    for metric, label in (("recall", f"Recall@{K}"), ("precision", f"Precision@{K}"), ("mrr", "MRR")):
        fig.add_trace(go.Bar(name=label, x=names, y=[results[n][metric] for n in names],
                             text=[f"{results[n][metric]:.2f}" for n in names], textposition="outside"))
    fig.update_layout(barmode="group", title="So sánh cấu hình retrieval (bộ eval 36 query)",
                      yaxis=dict(range=[0, 1.1]), height=520)
    save_figure(fig, "11_eval_comparison.html")


def debug_failure(r: InMemoryRetriever) -> None:
    banner("C. Debug query thất bại: lỗi ở RETRIEVAL hay RANKING?", "33")
    text_of = dict(zip(r.ids, r.texts))
    for q in load_eval_queries():
        pool = r.hybrid(q["query"], POOL, pool=POOL)
        final = r.rerank(q["query"], pool, 10)
        if recall_at_k([i for i, _ in final], q["relevant"], K) >= 1.0:
            continue
        print(f"  Query: {q['query']!r}   đáp án: {q['relevant']}")
        print(f"  Bước 1 - query tokenized (BM25): {tokenize(q['query'])}")
        pool_ids = [i for i, _ in pool]
        for d in q["relevant"]:
            where = f"hạng {pool_ids.index(d) + 1} trong pool" if d in pool_ids else f"KHÔNG nằm trong pool {POOL}"
            print(f"  Bước 3 - doc đúng {d}: {where}")
        print("  Bước 4/7 - top-3 cuối cùng (điểm reranker):")
        for doc_id, score in final[:3]:
            mark = "[v]" if doc_id in q["relevant"] else "[ ]"
            print(f"       {mark} {doc_id} rerank={score:6.2f}  {text_of[doc_id][:60]}")
        missing = [d for d in q["relevant"] if d not in pool_ids]
        print("  Kết luận:", "lỗi RETRIEVAL (doc đúng không vào pool) -> xem chunking/embedding/hybrid, tăng POOL."
              if missing else "lỗi RANKING (doc đúng có trong pool nhưng xếp thấp) -> xem reranker/ngưỡng top-K.")
        return
    print("  Không có query nào thất bại với hybrid + rerank trên bộ này.")


if __name__ == "__main__":
    retriever = InMemoryRetriever(active_docs("company_a"))
    chart(compare(retriever))
    debug_failure(retriever)
