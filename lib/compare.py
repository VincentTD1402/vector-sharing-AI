"""Lõi retrieval cho UI: chạy dense + lexical + RRF + rerank trên pgvector và suy ra 5 cấu hình
(keyword | vector | hybrid | vector + rerank | hybrid + rerank). Các bước của UI (retrieval, rerank,
query transformation, pipeline đầy đủ) đều dựa trên `run_configs` + `format_results`.

filter_mode:
  pre  - WHERE tenant/status/language/department trong SQL + mask BM25 (đúng và đủ k)
  post - lấy top-k thuần similarity rồi mới lọc (có thể trả thiếu)
  off  - bỏ filter tenant/status (CHỈ để minh họa rò rỉ)
"""
import time

import numpy as np

from lib.data import load_eval_queries
from lib.db import vector_search
from lib.models import embed_queries, rerank_scores
from lib.pipeline import RetrievalService
from lib.query_transform import TERSE_QUERIES
from lib.search import precision_at_k, recall_at_k, reciprocal_rank, rrf, tokenize

CONFIGS = [
    ("keyword", "Keyword (BM25)"), ("vector", "Vector"), ("hybrid", "Hybrid (RRF)"),
    ("vector_rerank", "Vector + Rerank"), ("hybrid_rerank", "Hybrid + Rerank"),
]
K_METRIC = 3
BASE_OF = {"vector_rerank": "vector", "hybrid_rerank": "hybrid"}  # cấu hình rerank -> cấu hình gốc
FILTER_MODES = ("pre", "post", "off")


def relevant_for(query: str, tenant_id: str) -> list[str] | None:
    """Đáp án đúng nếu query trùng một query có nhãn (bộ eval hoặc query ngắn của bước transformation)."""
    if tenant_id != "company_a":  # nhãn chỉ có cho tenant company_a
        return None
    key = query.strip().lower()
    for q in [*load_eval_queries(), *TERSE_QUERIES]:
        if q["query"].strip().lower() == key:
            return q["relevant"]
    return None


def _ms(start: float) -> float:
    return round((time.perf_counter() - start) * 1000, 1)


def _allowed(row: dict, tenant_id: str, languages, department) -> bool:
    return (row["tenant_id"] == tenant_id and row["status"] == "current"
            and (not languages or row["language"] in languages)
            and (not department or row["department"] == department))


def _dense(svc, qvec, tenant_id, languages, department, mode, pool, post_k) -> list[dict]:
    if mode == "pre":
        return vector_search(svc.conn, qvec, pool, tenant_id, "current", languages, department)
    if mode == "off":
        return vector_search(svc.conn, qvec, pool, "", None, languages, department, allow_cross_tenant=True)
    raw = vector_search(svc.conn, qvec, post_k, "", None, None, None, allow_cross_tenant=True)
    return [d for d in raw if _allowed(d, tenant_id, languages, department)]


def _lexical(svc, text, tenant_id, languages, department, mode, pool, post_k) -> list[tuple[str, float]]:
    if mode == "pre":
        return svc.lexical_scored(text, tenant_id, languages, department, pool)
    if mode == "off":
        return svc.lexical_scored(text, tenant_id, languages, department, pool, filters_off=True)
    by_id = {r["id"]: r for r in svc.rows}
    raw = svc.lexical_scored(text, tenant_id, None, None, post_k, filters_off=True)
    return [(i, s) for i, s in raw if _allowed(by_id[i], tenant_id, languages, department)]


def _fuse_dense(lists: list[list[dict]]) -> list[dict]:
    """Multi-query: gộp danh sách dense của từng biến thể bằng RRF; điểm hiển thị = cosine cao nhất."""
    best: dict[str, float] = {}
    for items in lists:
        for d in items:
            best[d["id"]] = max(best.get(d["id"], -1.0), d["score"])
    return [{"id": i, "score": best[i]} for i, _ in rrf([[d["id"] for d in items] for items in lists])]


def run_configs(
    svc: RetrievalService, query: str, tenant_id: str, languages=None, department=None,
    filter_mode: str = "pre", rerank: bool = True, variants: list[str] | None = None,
    pool: int | None = None, post_k: int = 5,
) -> dict:
    """Chạy tất cả bước và trả về danh sách xếp hạng đầy đủ (pool) của từng cấu hình + thời gian.
    Rerank luôn chấm theo query GỐC (ý định người dùng), không theo bản đã mở rộng."""
    pool = pool or svc.pool
    variants = variants or [query]
    lexical_text = " ".join(dict.fromkeys(t for v in variants for t in tokenize(v)))
    with svc._lock:  # connection DB và model không dùng đồng thời an toàn
        t = time.perf_counter()
        qvecs = embed_queries(variants)  # không qua cache để latency embed phản ánh thật
        ms_embed = _ms(t)

        t = time.perf_counter()
        per_variant = [_dense(svc, qv, tenant_id, languages, department, filter_mode, pool, post_k) for qv in qvecs]
        dense = per_variant[0] if len(per_variant) == 1 else _fuse_dense(per_variant)
        ms_dense = _ms(t)

        t = time.perf_counter()
        lexical = _lexical(svc, lexical_text, tenant_id, languages, department, filter_mode, pool, post_k)
        ms_lex = _ms(t)

        t = time.perf_counter()
        fused = rrf([[i for i, _ in lexical], [d["id"] for d in dense]])[:pool]
        ms_fuse = _ms(t)

        by_id = {r["id"]: r for r in svc.rows}
        reranked, rerank_ms = {}, {}
        if rerank:
            for key, pool_ids in (("vector_rerank", [d["id"] for d in dense]), ("hybrid_rerank", [i for i, _ in fused])):
                t = time.perf_counter()
                if pool_ids:
                    scores = rerank_scores(query, [by_id[i]["content"] for i in pool_ids])
                    reranked[key] = [(pool_ids[i], float(scores[i])) for i in np.argsort(-scores)]
                else:
                    reranked[key] = []
                rerank_ms[key] = _ms(t)

    return {
        "ranked": {"keyword": lexical, "vector": [(d["id"], d["score"]) for d in dense], "hybrid": fused, **reranked},
        "ms": {"embed": ms_embed, "dense": ms_dense, "lexical": ms_lex, "fusion": ms_fuse, **rerank_ms},
        "lists": {"dense": dense, "lexical": lexical, "fused": fused}, "per_variant": per_variant,
        "variants": variants, "lexical_text": lexical_text,
    }


def latency_of(key: str, ms: dict) -> float:
    """Latency tích lũy của từng cấu hình (đã gồm các bước nó phụ thuộc)."""
    base = {"keyword": ms["lexical"], "vector": ms["embed"] + ms["dense"],
            "hybrid": ms["embed"] + ms["dense"] + ms["lexical"] + ms["fusion"]}
    if key in base:
        return round(base[key], 1)
    return round(base[key.split("_")[0]] + ms.get(key, 0.0), 1)


def metrics_of(ids: list[str], relevant: list[str] | None) -> dict | None:
    if relevant is None:
        return None
    first = next((n for n, i in enumerate(ids, 1) if i in relevant), None)
    return {"recall": round(recall_at_k(ids, relevant, K_METRIC), 3),
            "precision": round(precision_at_k(ids, relevant, K_METRIC), 3),
            "mrr": round(reciprocal_rank(ids, relevant), 3), "first_hit": first}


def format_results(
    svc: RetrievalService, ranked: list[tuple[str, float]], top_k: int, tenant_id: str,
    relevant: list[str] | None, base_ranked: list[tuple[str, float]] | None = None,
) -> list[dict]:
    """Top-k của một danh sách xếp hạng -> dict cho UI (có cờ đúng/sai theo nhãn và cờ rò rỉ)."""
    by_id = {r["id"]: r for r in svc.rows}
    base_rank = {i: n for n, (i, _) in enumerate(base_ranked, 1)} if base_ranked is not None else {}
    out = []
    for rank, (doc_id, score) in enumerate(ranked[:top_k], 1):
        row = by_id[doc_id]
        out.append({
            "rank": rank, "id": doc_id, "score": round(score, 4), "content": row["content"],
            "base_rank": base_rank.get(doc_id) if base_ranked is not None else None,
            "tenant_id": row["tenant_id"], "language": row["language"], "status": row["status"],
            "department": row["department"],
            "relevant": None if relevant is None else doc_id in relevant,
            "leaked": row["tenant_id"] != tenant_id or row["status"] != "current",
        })
    return out


def rrf_table(svc: RetrievalService, run: dict, n: int = 8) -> list[dict]:
    by_id = {r["id"]: r for r in svc.rows}
    dense_rank = {d["id"]: k for k, d in enumerate(run["lists"]["dense"], 1)}
    lex_rank = {i: k for k, (i, _) in enumerate(run["lists"]["lexical"], 1)}
    return [{"id": i, "score": round(s, 5), "rank_dense": dense_rank.get(i), "rank_lexical": lex_rank.get(i),
             "content": by_id[i]["content"]} for i, s in run["lists"]["fused"][:n]]


def compare(
    svc: RetrievalService, query: str, tenant_id: str, top_k: int = 5, languages=None,
    department=None, filter_mode: str = "pre", variants: list[str] | None = None,
) -> dict:
    run = run_configs(svc, query, tenant_id, languages, department, filter_mode, True, variants, post_k=top_k)
    relevant = relevant_for(query, tenant_id)
    configs = []
    for key, label in CONFIGS:
        ranked = run["ranked"][key]
        base_key = BASE_OF.get(key)
        configs.append({
            "key": key, "label": label, "latency_ms": latency_of(key, run["ms"]),
            "results": format_results(svc, ranked, top_k, tenant_id, relevant,
                                      run["ranked"][base_key] if base_key else None),
            "metrics": metrics_of([i for i, _ in ranked], relevant),
        })
    return {
        "query": query, "relevant": relevant, "variants": run["variants"],
        "filters": {"tenant_id": tenant_id, "languages": languages, "department": department,
                    "filter_mode": filter_mode},
        "stages_ms": run["ms"],
        "counts": {k: len(v) for k, v in run["lists"].items()},
        "configs": configs, "rrf": rrf_table(svc, run),
    }


def evaluate_query(svc: RetrievalService, q: dict) -> dict:
    """Một query của bộ eval -> metric của 5 cấu hình (tenant company_a, filter bật)."""
    run = run_configs(svc, q["query"], "company_a")
    return {key: {**metrics_of([i for i, _ in run["ranked"][key]], q["relevant"]),
                  "ms": latency_of(key, run["ms"])} for key, _ in CONFIGS}
