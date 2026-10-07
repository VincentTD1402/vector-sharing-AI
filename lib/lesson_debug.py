"""Bài học Đánh giá: debug MỘT query theo quy trình 8 bước của slide 33 và liệt kê các query đang thất bại.

Mục tiêu: chỉ ra lỗi nằm ở đâu - RETRIEVAL (doc đúng không vào pool), FILTER (bị loại), hay RANKING
(có trong pool nhưng xếp thấp) - thay vì đổ cho LLM.
"""
import numpy as np

from lib.compare import CONFIGS, latency_of, metrics_of, run_configs
from lib.data import load_eval_queries
from lib.models import embed_queries
from lib.pipeline import RetrievalService
from lib.search import tokenize
from lib.step_retrieval import filter_sql

TENANT = "company_a"
_failing_cache: list[dict] | None = None


def _rank_in(ranked: list[tuple[str, float]], doc_id: str) -> int | None:
    return next((n for n, (i, _) in enumerate(ranked, 1) if i == doc_id), None)


def failing_queries(svc: RetrievalService) -> list[dict]:
    """Các query có cấu hình retrieval (chưa rerank) xếp doc đúng ngoài hạng 1."""
    global _failing_cache
    if _failing_cache is not None:
        return _failing_cache
    out = []
    for q in load_eval_queries():
        run = run_configs(svc, q["query"], TENANT, rerank=False)
        first = {key: next((n for n, (i, _) in enumerate(run["ranked"][key], 1) if i in q["relevant"]), None)
                 for key in ("keyword", "vector", "hybrid")}
        if any(v != 1 for v in first.values()):
            out.append({"id": q["id"], "query": q["query"], "kind": q["kind"], "ranks": first})
    _failing_cache = out
    return out


def debug_query(svc: RetrievalService, qid: str, pool: int = 20) -> dict:
    q = next((x for x in load_eval_queries() if x["id"] == qid), None)
    if q is None:
        raise ValueError(f"không có query '{qid}'")
    by_id = {r["id"]: r for r in svc.rows}
    rel = q["relevant"]
    run = run_configs(svc, q["query"], TENANT, pool=pool)
    ranked, lists = run["ranked"], run["lists"]
    dense = lists["dense"]
    scores = [d["score"] for d in dense]
    best_rel = max((d["score"] for d in dense if d["id"] in rel), default=None)

    rank = {name: {r: _rank_in(ranked[name], r) for r in rel} for name in
            ("keyword", "vector", "hybrid", "vector_rerank", "hybrid_rerank")}
    first = lambda name: min((v for v in rank[name].values() if v is not None), default=None)
    pool_of = lambda name: first(name) is not None
    final = first("hybrid_rerank")
    if not pool_of("hybrid"):
        stage, detail = "retrieval", "Doc đúng KHÔNG nằm trong pool ứng viên của hybrid: lỗi ở RETRIEVAL (chunking/embedding/hybrid/pool)."
    elif final is not None and final > 3:
        stage, detail = "ranking", f"Doc đúng có trong pool (hạng {first('hybrid')} sau hybrid) nhưng chỉ đứng #{final} sau rerank: lỗi ở RANKING."
    elif final is not None and final > min(v for v in (first("keyword"), first("vector"), first("hybrid")) if v is not None):
        stage, detail = "regressed", f"Retrieval đã xếp doc đúng ở hạng {min(v for v in (first('keyword'), first('vector'), first('hybrid')) if v is not None)} nhưng RERANK đẩy xuống #{final}: reranker làm tệ đi."
    elif first("vector") not in (1, None) or first("keyword") not in (1, None):
        stage, detail = "recovered", "Một nhánh retrieval xếp thấp nhưng các bước sau (hybrid/rerank) đã cứu được."
    else:
        stage, detail = "ok", "Mọi bước đều đưa doc đúng lên đầu."
    qv = embed_queries([q["query"]])[0]
    return {
        "id": q["id"], "query": q["query"], "kind": q["kind"], "relevant": rel,
        "relevant_docs": [{"id": r, "content": by_id[r]["content"], "language": by_id[r]["language"]} for r in rel],
        "tokens": tokenize(q["query"]),
        "embedding": {"dim": int(qv.shape[0]), "norm": round(float(np.linalg.norm(qv)), 3),
                      "dims": [round(float(x), 4) for x in qv[:48]], "ms": run["ms"]["embed"]},
        "filter": {"tenant_id": TENANT, "sql": filter_sql(TENANT, None, None, "pre", pool, 5), "pool": pool,
                   "pool_size": len(dense),
                   "relevant_passes": {r: (by_id[r]["tenant_id"] == TENANT and by_id[r]["status"] == "current") for r in rel}},
        "dense": [{"id": d["id"], "score": round(d["score"], 4), "content": d["content"], "relevant": d["id"] in rel} for d in dense[:8]],
        "gap": {"top1": round(scores[0], 4) if scores else None, "tenth": round(scores[9], 4) if len(scores) > 9 else None,
                "relevant": round(best_rel, 4) if best_rel is not None else None},
        "lexical": [{"id": i, "score": round(s, 3), "relevant": i in rel} for i, s in lists["lexical"][:6]],
        "rerank": [{"id": i, "score": round(s, 3), "relevant": i in rel, "content": by_id[i]["content"]}
                   for i, s in ranked["hybrid_rerank"][:6]],
        "ranks": rank, "stage": stage, "detail": detail,
        "metrics": {key: metrics_of([i for i, _ in ranked[key]], rel) for key, _ in CONFIGS},
        "ms": run["ms"],
    }
