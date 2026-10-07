"""Dữ liệu cho các bài online của UI: query transformation, retrieval + filter, reranking (slide 16-20, 27)."""
import time

import numpy as np

from lib.compare import (
    CONFIGS, format_results, latency_of, metrics_of, relevant_for, rrf_table, run_configs,
)
from lib.models import rerank_scores
from lib.pipeline import RetrievalService
from lib.query_transform import MODES, explain, variants
from lib.search import tokenize


def filter_sql(tenant_id: str, languages, department, mode: str, pool: int, post_k: int) -> str:
    """Câu SQL thật (dạng đọc được) mà nhánh vector thực thi theo chế độ lọc."""
    base = "SELECT id, content, 1 - (embedding <=> :query_vec) AS score\nFROM document_chunks"
    tail = "ORDER BY embedding <=> :query_vec"
    if mode == "post":
        return (f"{base}\n{tail}\nLIMIT {post_k};\n"
                f"-- Sau đó ỨNG DỤNG mới loại tenant ≠ '{tenant_id}', status ≠ 'current' khỏi {post_k} dòng trên")
    where = [] if mode == "off" else [f"tenant_id = '{tenant_id}'", "status = 'current'"]
    if languages:
        where.append("language = ANY(" + str(list(languages)).replace('"', "'") + ")")
    if department:
        where.append(f"department = '{department}'")
    clause = ("WHERE " + "\n  AND ".join(where) + "\n") if where else ""
    note = "\n-- KHÔNG có điều kiện tenant/status: kết quả của mọi tenant và cả tài liệu lỗi thời" if mode == "off" else ""
    return f"{base}\n{clause}{tail}\nLIMIT {pool};{note}"


def transform_step(svc: RetrievalService, query: str, tenant_id: str, top_k: int = 5) -> dict:
    """Cùng một query ngắn/viết tắt đi qua 4 cách biến đổi; retrieve bằng VECTOR search để thấy rõ tác động."""
    relevant = relevant_for(query, tenant_id)
    runs = []
    for mode, label in MODES.items():
        vs = variants(query, mode)
        run = run_configs(svc, query, tenant_id, filter_mode="pre", rerank=False, variants=vs)
        ranked = run["ranked"]["vector"]
        entry = {
            "mode": mode, "label": label, "variants": vs, "ms": round(run["ms"]["embed"] + run["ms"]["dense"], 1),
            "results": format_results(svc, ranked, top_k, tenant_id, relevant),
            "metrics": metrics_of([i for i, _ in ranked], relevant),
        }
        if mode == "multi":  # kết quả riêng của từng biến thể trước khi gộp RRF
            entry["per_variant"] = [{"variant": v, "results": format_results(
                svc, [(d["id"], d["score"]) for d in items], 3, tenant_id, relevant)}
                for v, items in zip(vs, run["per_variant"])]
        runs.append(entry)
    return {"query": query, "relevant": relevant, "rules": explain(query), "runs": runs}


def _first_rank(ids: list[str], relevant: list[str] | None) -> int | None:
    if not relevant:
        return None
    return next((n for n, i in enumerate(ids, 1) if i in relevant), None)


def retrieve_step(
    svc: RetrievalService, query: str, tenant_id: str, top_k: int, languages, department, filter_mode: str,
) -> dict:
    """Keyword | Vector | Hybrid với filter_mode pre/post/off (chưa rerank), kèm dữ liệu từng nhánh để UI kể chuyện."""
    run = run_configs(svc, query, tenant_id, languages, department, filter_mode, rerank=False, post_k=top_k)
    relevant = relevant_for(query, tenant_id)
    by_id = {r["id"]: r for r in svc.rows}
    configs = []
    for key, label in CONFIGS[:3]:
        ranked = run["ranked"][key]
        results = format_results(svc, ranked, top_k, tenant_id, relevant)
        configs.append({"key": key, "label": label, "latency_ms": latency_of(key, run["ms"]), "results": results,
                        "returned": len(results), "metrics": metrics_of([i for i, _ in ranked], relevant)})
    q_tokens = set(tokenize(query))
    bm25 = [{"id": i, "score": round(s, 3), "matched": sorted(q_tokens & set(tokenize(by_id[i]["content"]))),
             "content": by_id[i]["content"], "language": by_id[i]["language"],
             "relevant": None if relevant is None else i in relevant}
            for i, s in run["lists"]["lexical"][:8]]
    dense = [{"id": d["id"], "score": round(d["score"], 4), "content": d["content"], "language": d["language"],
              "tenant_id": d["tenant_id"], "status": d["status"],
              "relevant": None if relevant is None else d["id"] in relevant,
              "leaked": d["tenant_id"] != tenant_id or d["status"] != "current"}
             for d in run["lists"]["dense"][:8]]
    return {
        "query": query, "relevant": relevant, "top_k": top_k, "bm25_tokens": tokenize(query),
        "filters": {"tenant_id": tenant_id, "languages": languages, "department": department, "filter_mode": filter_mode},
        "sql": filter_sql(tenant_id, languages, department, filter_mode, svc.pool, top_k),
        "stages_ms": run["ms"], "counts": {k: len(v) for k, v in run["lists"].items()},
        "bm25": bm25, "dense": dense,
        "ranks": {key: _first_rank([i for i, _ in run["ranked"][key]], relevant) for key, _ in CONFIGS[:3]},
        "configs": configs, "rrf": rrf_table(svc, run),
    }


def rerank_step(
    svc: RetrievalService, query: str, tenant_id: str, source: str, pool: int, languages=None, department=None,
) -> dict:
    """Lấy `pool` ứng viên từ retriever (vector hoặc hybrid) rồi chấm lại bằng cross-encoder."""
    run = run_configs(svc, query, tenant_id, languages, department, "pre", rerank=False, pool=pool)
    candidates = run["ranked"][source][:pool]
    by_id = {r["id"]: r for r in svc.rows}
    relevant = relevant_for(query, tenant_id)
    if not candidates:
        return {"query": query, "source": source, "relevant": relevant, "candidates": [], "ms": {}, "metrics": None}

    with svc._lock:
        start = time.perf_counter()
        scores = rerank_scores(query, [by_id[i]["content"] for i, _ in candidates])
        rerank_ms = (time.perf_counter() - start) * 1000
    order = list(np.argsort(-scores))
    after_rank = {candidates[i][0]: n for n, i in enumerate(order, 1)}
    rows = []
    for before_rank, (doc_id, before_score) in enumerate(candidates, 1):
        row = by_id[doc_id]
        rows.append({
            "id": doc_id, "content": row["content"], "language": row["language"],
            "before_rank": before_rank, "before_score": round(before_score, 4),
            "after_rank": after_rank[doc_id], "after_score": round(float(scores[before_rank - 1]), 3),
            "relevant": None if relevant is None else doc_id in relevant,
        })
    rows.sort(key=lambda r: r["after_rank"])
    retrieve_ms = run["ms"]["embed"] + run["ms"]["dense"] + (run["ms"]["lexical"] + run["ms"]["fusion"] if source == "hybrid" else 0)
    return {
        "query": query, "source": source, "relevant": relevant, "pairs": len(candidates), "candidates": rows,
        "ms": {"retrieve": round(retrieve_ms, 1), "rerank": round(rerank_ms, 1),
               "per_pair": round(rerank_ms / len(candidates), 1)},
        "metrics": {"before": metrics_of([i for i, _ in candidates], relevant),
                    "after": metrics_of([candidates[i][0] for i in order], relevant)},
    }
