"""Embedding model + reranker (chạy local trên CPU)."""
import hashlib
from functools import lru_cache

import numpy as np

from lib.config import CACHE_DIR, EMBED_MODEL, RERANK_MODEL


def _load(model_cls, name: str):
    """Ưu tiên cache cục bộ (nhanh, chạy được khi mất mạng/wifi lớp học yếu);
    chỉ khi chưa có mới tải từ HuggingFace Hub (lần chạy đầu)."""
    try:
        return model_cls(name, device="cpu", local_files_only=True)
    except OSError:
        print(f"[models] Lần đầu dùng {name}: đang tải từ HuggingFace Hub (~0.5 GB)...")
        return model_cls(name, device="cpu")


@lru_cache
def get_embedder():
    from sentence_transformers import SentenceTransformer

    return _load(SentenceTransformer, EMBED_MODEL)


@lru_cache
def get_reranker():
    from sentence_transformers import CrossEncoder

    return _load(CrossEncoder, RERANK_MODEL)


def _encode(texts: list[str], prefix: str, batch_size: int = 32) -> np.ndarray:
    # normalize_embeddings=True -> vector độ dài 1, nên cosine == dot product.
    return get_embedder().encode(
        [prefix + t for t in texts],
        normalize_embeddings=True,
        batch_size=batch_size,
        show_progress_bar=False,
        convert_to_numpy=True,
    ).astype(np.float32)


def embed_passages(texts: list[str]) -> np.ndarray:
    """Embed tài liệu (prefix 'passage: ' theo yêu cầu của e5)."""
    return _encode(texts, "passage: ")


def embed_queries(texts: list[str]) -> np.ndarray:
    """Embed câu hỏi (prefix 'query: ')."""
    return _encode(texts, "query: ")


def embed_passages_cached(texts: list[str]) -> np.ndarray:
    """Như embed_passages nhưng cache ra đĩa theo hash nội dung + tên model."""
    key = hashlib.sha1((EMBED_MODEL + "\x00".join(texts)).encode("utf-8")).hexdigest()[:16]
    path = CACHE_DIR / f"passages_{key}.npy"
    if path.exists():
        return np.load(path)
    vectors = embed_passages(texts)
    np.save(path, vectors)
    return vectors


def rerank_scores(query: str, passages: list[str]) -> np.ndarray:
    """Cross-encoder chấm điểm từng cặp (query, passage)."""
    return np.asarray(get_reranker().predict([(query, p) for p in passages]))
