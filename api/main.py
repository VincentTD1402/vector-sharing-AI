"""Search API (slide 30): POST /search với filter bắt buộc, giới hạn top_k, log từng request.

Chạy:  uvicorn api.main:app --port 8000      (từ thư mục gốc dự án; cần docker compose up -d)
Docs : http://localhost:8000/docs
"""
import json
import logging
import uuid
from contextlib import asynccontextmanager

from pathlib import Path

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from api.state import services
from api.lesson_routes import router as lesson_router
from api.step_routes import router as step_router
from api.ui_routes import router as ui_router
from lib.pipeline import ALLOWED_FILTERS, RetrievalService
from lib.space_projection import SpaceProjector

MAX_TOP_K = 20  # chặn client xin quá nhiều kết quả
UI_DIR = Path(__file__).resolve().parent.parent / "ui"
logger = logging.getLogger("search")
logging.basicConfig(level=logging.INFO, format="%(message)s")
logging.getLogger("httpx").setLevel(logging.WARNING)  # bớt nhiễu log HTTP của thư viện model


@asynccontextmanager
async def lifespan(app: FastAPI):
    services["main"] = RetrievalService()  # index corpus + nạp BM25 + warm-up model
    services["main"].search("warm up", "company_a", top_k=1)
    services["space"] = SpaceProjector()  # PCA cho bản đồ không gian vector của UI
    yield
    services.clear()


app = FastAPI(title="Semantic Search API", lifespan=lifespan)
app.include_router(ui_router)
app.include_router(step_router)
app.include_router(lesson_router)


class SearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=500)
    top_k: int = Field(default=5, ge=1, le=MAX_TOP_K)
    # Production: tenant_id phải lấy từ token xác thực, không nhận từ client. Demo nhận qua body.
    tenant_id: str = Field(min_length=1)
    language: str | None = None
    filters: dict[str, str] = {}


class SearchResult(BaseModel):
    id: str
    content: str
    score: float
    metadata: dict


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "chunks": services["main"].chunk_count}


@app.post("/search", response_model=list[SearchResult])
def search(req: SearchRequest, response: Response) -> list[SearchResult]:
    unknown = set(req.filters) - ALLOWED_FILTERS
    if unknown:  # whitelist: không cho client lọc theo cột tùy ý
        raise HTTPException(422, f"filter không được hỗ trợ: {sorted(unknown)}; cho phép: {sorted(ALLOWED_FILTERS)}")

    request_id = uuid.uuid4().hex[:12]
    results, trace = services["main"].search(
        req.query, req.tenant_id, req.top_k,
        languages=[req.language] if req.language else None,
        department=req.filters.get("department"),
    )
    # Log có cấu trúc để debug/monitor (slide 31): query, filter, thời gian, kết quả.
    logger.info(json.dumps({"request_id": request_id, **trace}, ensure_ascii=False))
    response.headers["X-Request-Id"] = request_id
    return [
        SearchResult(
            id=r["id"], content=r["content"], score=round(r["score"], 4),
            metadata={"tenant_id": r["tenant_id"], "language": r["language"], "department": r["department"]},
        )
        for r in results
    ]


@app.get("/", include_in_schema=False)
def root(request: Request) -> RedirectResponse:
    query = f"?{request.url.query}" if request.url.query else ""  # giữ ?q=... khi chuyển sang UI
    return RedirectResponse(f"/ui/{query}")


# Đặt cuối cùng: mount "/ui" không được che các route API phía trên.
app.mount("/ui", StaticFiles(directory=UI_DIR, html=True), name="ui")
