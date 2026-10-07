"""Endpoint /api/* phục vụ UI: so sánh cấu hình, bản đồ không gian vector, đánh giá streaming.

/search (api/main.py) vẫn là API an toàn. Ở đây `filter_mode="off"` chỉ để UI MINH HỌA rò rỉ
multi-tenant, nên chỉ có trong router của demo.
"""
import json
from collections import defaultdict
from typing import Literal

import numpy as np
from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from api.state import services
from lib.compare import CONFIGS, K_METRIC, compare, evaluate_query
from lib import step_chunking
from lib.data import load_corpus, load_eval_queries
from lib.query_transform import MODES, TERSE_QUERIES, variants
from lib.space_projection import embedding_info

router = APIRouter(prefix="/api")


class CompareRequest(BaseModel):
    query: str = Field(min_length=1, max_length=500)
    tenant_id: str = "company_a"
    top_k: int = Field(default=5, ge=1, le=10)
    language: str | None = None
    department: str | None = None
    filter_mode: Literal["pre", "post", "off"] = "pre"
    transform: Literal["none", "rewrite", "expand", "multi"] = "none"


@router.get("/meta")
def meta() -> dict:
    corpus = load_corpus()
    return {
        "chunks": services["main"].chunk_count,
        "tenants": sorted({d["tenant_id"] for d in corpus}),
        "departments": sorted({d["department"] for d in corpus if d["department"]}),
        "configs": [{"key": k, "label": label} for k, label in CONFIGS],
        "examples": [{"query": q["query"], "kind": q["kind"]} for q in load_eval_queries()],
        "terse_examples": [{"query": q["query"], "kind": "ngắn/viết tắt"} for q in TERSE_QUERIES],
        "manuals": step_chunking.catalog(),
        "transform_modes": [{"key": k, "label": v} for k, v in MODES.items()],
        "metric_k": K_METRIC,
    }


@router.post("/compare")
def compare_endpoint(req: CompareRequest) -> dict:
    result = compare(
        services["main"], req.query, req.tenant_id, req.top_k,
        [req.language] if req.language else None, req.department, req.filter_mode,
        variants(req.query, req.transform),
    )
    result["embedding"] = embedding_info(req.query)
    result["query_point"] = services["space"].project_query(req.query)
    return result


@router.get("/space")
def space() -> dict:
    return services["space"].points()


def _summarize(rows: list[dict]) -> dict:
    """Gộp metric theo cấu hình và theo loại query."""
    keys = [k for k, _ in CONFIGS]
    summary = {}
    for key in keys:
        entry = {m: float(np.mean([r["configs"][key][m] for r in rows])) for m in ("recall", "precision", "mrr", "ms")}
        by_kind = defaultdict(list)
        for r in rows:
            by_kind[r["kind"]].append(r["configs"][key]["recall"])
        entry["recall_by_kind"] = {k: float(np.mean(v)) for k, v in by_kind.items()}
        summary[key] = entry
    return summary


@router.get("/eval/stream")
def eval_stream() -> StreamingResponse:
    """NDJSON: mỗi dòng là kết quả một query (UI cập nhật tiến độ), dòng cuối là tổng hợp."""
    queries = load_eval_queries()

    def generate():
        rows = []
        for i, q in enumerate(queries, 1):
            row = {"i": i, "n": len(queries), "query": q["query"], "kind": q["kind"],
                   "configs": evaluate_query(services["main"], q)}
            rows.append(row)
            yield json.dumps(row, ensure_ascii=False) + "\n"
        yield json.dumps({"done": True, "summary": _summarize(rows), "metric_k": K_METRIC}, ensure_ascii=False) + "\n"

    return StreamingResponse(generate(), media_type="application/x-ndjson")
