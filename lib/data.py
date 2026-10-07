"""Nạp corpus ERP/helpdesk, bộ câu hỏi đánh giá và các sổ tay dài."""
import json
from functools import lru_cache

from lib.config import DATA_DIR


@lru_cache
def load_corpus() -> list[dict]:
    return json.loads((DATA_DIR / "corpus.json").read_text(encoding="utf-8"))


def active_docs(tenant_id: str = "company_a", status: str = "current") -> list[dict]:
    """Tập tài liệu 'hợp lệ' cho một tenant: đúng tenant + còn hiệu lực."""
    return [
        d for d in load_corpus()
        if d["tenant_id"] == tenant_id and d["status"] == status
    ]


@lru_cache
def load_eval_queries() -> list[dict]:
    return json.loads((DATA_DIR / "eval_queries.json").read_text(encoding="utf-8"))


@lru_cache
def load_manuals() -> dict[str, str]:
    """Tên file -> nội dung. Dùng cho demo chunking."""
    return {
        p.stem: p.read_text(encoding="utf-8")
        for p in sorted((DATA_DIR / "manuals").glob("*.txt"))
    }


@lru_cache
def load_manual_questions() -> list[dict]:
    return json.loads((DATA_DIR / "manual_questions.json").read_text(encoding="utf-8"))
