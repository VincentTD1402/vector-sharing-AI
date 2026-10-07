"""Demo 10 - Reranking: bi-encoder (nhanh, recall) -> cross-encoder (chậm, precision) (slide 20).

A. Vì sao cross-encoder chỉ dùng cho tập nhỏ: so chi phí bi-encoder (vector có sẵn) với
   cross-encoder (chạy model cho TỪNG cặp query-doc).
B. Thứ hạng của doc đúng TRƯỚC và SAU rerank trên các query khó.
C. Số liệu trên toàn bộ bộ eval: MRR, Recall@1, latency.

Chạy:  python demos/10_reranking.py
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from lib.data import active_docs, load_eval_queries
from lib.models import embed_queries, rerank_scores
from lib.search import InMemoryRetriever, reciprocal_rank
from lib.viz import banner

POOL, TOP = 20, 5


def timed(fn):
    start = time.perf_counter()
    out = fn()
    return out, (time.perf_counter() - start) * 1000


def cost_comparison(r: InMemoryRetriever) -> None:
    banner("A. Chi phí: bi-encoder vs cross-encoder", "20")
    query, texts = "How do I reset my password?", r.texts
    rerank_scores(query, texts[:2])  # warm-up: lần chạy đầu tính cả thời gian nạp model
    embed_queries(["warm up"])
    _, bi_ms = timed(lambda: r.vectors @ embed_queries([query])[0])
    print(f"  Bi-encoder   : embed 1 query + dot với {len(texts)} vector có sẵn = {bi_ms:.0f} ms")
    print("                 (vector của doc đã tính sẵn lúc index; thêm doc không làm query chậm hơn nhiều)")
    for n in (10, 20, len(texts)):
        _, ms = timed(lambda n=n: rerank_scores(query, texts[:n]))
        print(f"  Cross-encoder: chấm {n:>3} cặp (query, doc)            = {ms:.0f} ms   ({ms / n:.0f} ms/cặp)")
    print("  Cross-encoder đọc query+doc CÙNG LÚC nên chính xác hơn, nhưng phải chạy lại cho mỗi cặp:")
    print("  chi phí tuyến tính theo số ứng viên -> chỉ dùng cho top-20..50, không phải cả corpus.")


def rank_shift(r: InMemoryRetriever) -> None:
    banner(f"B. Thứ hạng doc đúng: vector top-{POOL} -> rerank", "20")
    rows = []
    for q in load_eval_queries():
        pool = r.vector(q["query"], POOL)
        before = [i for i, _ in pool]
        after = [i for i, _ in r.rerank(q["query"], pool, POOL)]
        pos = lambda ids: next((n for n, i in enumerate(ids, 1) if i in q["relevant"]), None)
        rows.append((q, pos(before), pos(after)))
    shifts = sorted((x for x in rows if x[1] != x[2]), key=lambda x: (x[2] or 99) - (x[1] or 99))
    print(f"  {len(shifts)}/{len(rows)} query bị đổi vị trí doc đúng đầu tiên (sắp theo mức cải thiện):")
    print(f"  {'query':<52} {'trước':>6} {'sau':>5}")
    shown = shifts if len(shifts) <= 6 else shifts[:3] + shifts[-3:]
    for q, b, a in shown:
        print(f"  {q['query'][:50]:<52} {b or '>20':>6} {a or '>20':>5}")
    print("  Reranker không phải phép màu: có query được kéo lên hạng 1 nhưng cũng có query bị đẩy xuống -> luôn đo (demo 11).")


def metrics(r: InMemoryRetriever) -> None:
    banner("C. Toàn bộ bộ eval: vector vs vector + rerank", "20, 23")
    qs = load_eval_queries()
    out = {"vector": ([], [], []), "vector + rerank": ([], [], [])}
    for q in qs:
        pool, t_ret = timed(lambda q=q: r.vector(q["query"], POOL))
        reranked, t_rr = timed(lambda q=q, pool=pool: r.rerank(q["query"], pool, TOP))
        for name, ids, ms in (("vector", [i for i, _ in pool], t_ret),
                              ("vector + rerank", [i for i, _ in reranked], t_ret + t_rr)):
            out[name][0].append(reciprocal_rank(ids, q["relevant"]))
            out[name][1].append(float(ids[0] in q["relevant"]))
            out[name][2].append(ms)
    print(f"  {'cấu hình':<18} {'MRR':>6} {'Recall@1':>9} {'latency/query':>14}")
    for name, (mrr, r1, ms) in out.items():
        print(f"  {name:<18} {np.mean(mrr):>6.3f} {np.mean(r1):>9.2f} {np.mean(ms):>11.0f} ms")
    print(f"  Đánh đổi: rerank thường tăng độ chính xác ở đầu danh sách nhưng cộng thêm latency ({POOL} cặp/query).")


if __name__ == "__main__":
    retriever = InMemoryRetriever(active_docs("company_a"))
    cost_comparison(retriever)
    rank_shift(retriever)
    metrics(retriever)
