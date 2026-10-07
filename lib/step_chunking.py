"""Danh mục tài liệu cho bài Chunking."""
from lib.chunking import count_tokens
from lib.data import load_manuals

MAX_CUSTOM_CHARS = 20_000


def catalog() -> list[dict]:
    return [{"name": name, "chars": len(text), "tokens": count_tokens(text)} for name, text in load_manuals().items()]


def resolve_text(source: str, custom: str) -> str:
    if source.startswith("manual:"):
        manuals = load_manuals()
        name = source.removeprefix("manual:")
        if name not in manuals:
            raise ValueError(f"không có tài liệu '{name}'")
        return manuals[name]
    return custom[:MAX_CUSTOM_CHARS]
