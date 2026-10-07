"""Endpoint /api/lesson/* : dữ liệu từng bước cho các BÀI HỌC của UI (mỗi bài một tình huống trong slide)."""
from typing import Annotated, Literal

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field, model_validator

from api.state import services
from lib import lesson_ann, lesson_chunking, lesson_debug
from lib.cache import SemanticCache

router = APIRouter(prefix="/api/lesson")


class ChunkLessonRequest(BaseModel):
    source: str = "manual:production_planning_handbook_en"
    custom: str = Field(default="", max_length=20_000)
    strategy: Literal["fixed", "sentence", "recursive"] = "fixed"
    size: int = Field(default=40, ge=8, le=500)
    overlap: int = Field(default=0, ge=0, le=250)
    query: str = Field(min_length=1, max_length=500)
    answer: str = Field(default="", max_length=200)

    @model_validator(mode="after")
    def _overlap_below_size(self):
        if self.size - self.overlap < 4:
            raise ValueError("size - overlap phải ≥ 4 (bước trượt quá nhỏ sinh quá nhiều chunk)")
        return self


@router.post("/chunk")
def chunk_lesson(req: ChunkLessonRequest) -> dict:
    try:
        return lesson_chunking.run(req.source, req.strategy, req.size, req.overlap, req.query, req.answer, req.custom)
    except ValueError as exc:
        raise HTTPException(422, str(exc))


@router.get("/ann/scaling")
def ann_scaling() -> dict:
    return lesson_ann.scaling()


@router.get("/ann/ivf")
def ann_ivf(nprobe: int = Query(1, ge=1, le=8)) -> dict:
    return lesson_ann.ivf_toy(nprobe=nprobe)


@router.get("/ann/hnsw")
def ann_hnsw(ef: int = Query(6, ge=1, le=30)) -> dict:
    return lesson_ann.hnsw_toy(ef=ef)


@router.get("/ann/errors")
def ann_errors(case: Literal["dim_mismatch", "query_dim", "hnsw_2048"]) -> dict:
    return lesson_ann.errors(case)


@router.get("/debug/failing")
def debug_failing() -> list[dict]:
    return lesson_debug.failing_queries(services["main"])


@router.get("/debug/run")
def debug_run(qid: str = Query(min_length=2, max_length=10), pool: int = Query(20, ge=5, le=30)) -> dict:
    try:
        return lesson_debug.debug_query(services["main"], qid, pool)
    except ValueError as exc:
        raise HTTPException(404, str(exc))


class CacheRequest(BaseModel):
    queries: list[Annotated[str, Field(max_length=500)]] = Field(min_length=1, max_length=12)
    threshold: float = Field(default=0.95, ge=0.5, le=0.999)
    tenant_id: str = Field(default="company_a", min_length=1, max_length=64)


@router.post("/cache")
def cache_lab(req: CacheRequest) -> dict:
    """Chạy tuần tự các query qua một semantic cache MỚI; mỗi dòng cho biết hit/miss, thời gian và độ giống."""
    import time

    svc = services["main"]
    svc._embed_cache.clear()
    cache = SemanticCache(svc, req.threshold)
    rows = []
    for q in [x.strip() for x in req.queries if x.strip()]:
        t = time.perf_counter()
        hit, sim, matched = cache.lookup(q, req.tenant_id)
        if hit is not None:
            results, outcome = hit, "hit"
        else:
            results, _ = svc.search(q, req.tenant_id, top_k=3)
            cache.put(q, req.tenant_id, results)
            outcome = "miss"
        elapsed = round((time.perf_counter() - t) * 1000, 1)
        correct = True
        if outcome == "hit":  # kiểm chứng ngoài phép đo: kết quả cache có giống kết quả tìm thật không?
            fresh, _ = svc.search(q, req.tenant_id, top_k=3)
            correct = bool(fresh) and bool(results) and fresh[0]["id"] == results[0]["id"]
        rows.append({"query": q, "outcome": outcome, "ms": elapsed, "correct": correct,
                     "similarity": None if sim is None else round(sim, 4), "matched_query": matched if outcome == "hit" else None,
                     "top": [{"id": r["id"], "content": r["content"][:70]} for r in results[:2]]})
    return {"threshold": req.threshold, "rows": rows}
