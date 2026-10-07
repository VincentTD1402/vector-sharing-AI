"""Endpoint /api/* cho từng bài của UI: embedding, index, transformation, retrieval, rerank."""
import json
from typing import Literal

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from api.state import services
from lib import step_embedding, step_index, step_retrieval

router = APIRouter(prefix="/api")
FilterMode = Literal["pre", "post", "off"]


class EmbedRequest(BaseModel):
    query: str = Field(min_length=1, max_length=300)
    texts: list[str] = Field(min_length=1, max_length=step_embedding.MAX_TEXTS)
    use_prefix: bool = True
    scale_last: float = Field(default=1.0, ge=0.1, le=10.0)


@router.post("/embed/compare")
def embed_compare(req: EmbedRequest) -> dict:
    texts = [t.strip()[:300] for t in req.texts if t.strip()]
    if not texts:
        raise HTTPException(422, "Cần ít nhất một văn bản")
    space = services["space"]
    result = step_embedding.compare_texts(req.query, texts, req.use_prefix, req.scale_last)
    result["trace"] = step_embedding.embed_trace(req.query)
    result["points"] = {"query": space.project_query(req.query), "texts": space.project_texts(texts)}
    return result


@router.get("/index/info")
def index_info() -> dict:
    return step_index.index_info(services["main"])


@router.get("/ann/stream")
def ann_stream(kind: Literal["exact", "ivfflat", "hnsw"]) -> StreamingResponse:
    return StreamingResponse((json.dumps(e, ensure_ascii=False) + "\n" for e in step_index.ann_events(kind)),
                             media_type="application/x-ndjson")


@router.post("/ann/filter")
def ann_filter() -> dict:
    return step_index.selective_filter()


class QueryRequest(BaseModel):
    query: str = Field(min_length=1, max_length=500)
    tenant_id: str = Field(default="company_a", min_length=1, max_length=64)
    top_k: int = Field(default=5, ge=1, le=10)
    language: str | None = None
    department: str | None = None
    filter_mode: FilterMode = "pre"


@router.post("/transform")
def transform(req: QueryRequest) -> dict:
    return step_retrieval.transform_step(services["main"], req.query, req.tenant_id, req.top_k)


@router.post("/retrieve")
def retrieve(req: QueryRequest) -> dict:
    return step_retrieval.retrieve_step(
        services["main"], req.query, req.tenant_id, req.top_k,
        [req.language] if req.language else None, req.department, req.filter_mode)


class RerankRequest(BaseModel):
    query: str = Field(min_length=1, max_length=500)
    tenant_id: str = Field(default="company_a", min_length=1, max_length=64)
    source: Literal["vector", "hybrid"] = "hybrid"
    pool: int = Field(default=20, ge=3, le=30)


@router.post("/rerank")
def rerank(req: RerankRequest) -> dict:
    return step_retrieval.rerank_step(services["main"], req.query, req.tenant_id, req.source, req.pool)
