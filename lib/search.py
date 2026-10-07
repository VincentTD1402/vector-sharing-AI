"""Các khối retrieval in-memory: BM25, dense search, RRF, reranking, metrics.

Dùng cho các demo giải thích thuật toán (không cần DB). Pipeline production
trên pgvector nằm ở lib/pipeline.py và tái sử dụng chính các hàm này.
"""
import re

import numpy as np
from rank_bm25 import BM25Okapi

from lib.models import embed_passages_cached, embed_queries, rerank_scores


# Từ dừng (rất phổ biến, không mang nghĩa): nếu giữ lại, "to/the/from/của" sẽ khớp gần như mọi doc.
STOPWORDS = frozenset("""
a an the to of in on for from with at by is are was be it its this that these those my your our
i you we do does did how what when why who which can could and or as if so not no
và của là các một những được cho trong khi để có không này đã sẽ với tôi bạn thì mà
""".split())


def tokenize(text: str) -> list[str]:
    """Tách từ cho BM25: chữ thường, chuỗi ký tự \\w (giữ nguyên ERR_CONNECTION_RESET), bỏ stopword."""
    return [t for t in re.findall(r"\w+", text.lower()) if t not in STOPWORDS]


class InMemoryRetriever:
    """BM25 + dense search + hybrid (RRF) + rerank trên một danh sách docs."""

    def __init__(self, docs: list[dict]):
        self.docs = docs
        self.ids = [d["id"] for d in docs]
        self.texts = [d["text"] for d in docs]
        self.vectors = embed_passages_cached(self.texts)  # (N, 384), đã normalize
        self.bm25 = BM25Okapi([tokenize(t) for t in self.texts])

    # --- từng phương pháp trả về list[(doc_id, score)] đã sắp xếp giảm dần -----
    def keyword(self, query: str, k: int = 10) -> list[tuple[str, float]]:
        scores = self.bm25.get_scores(tokenize(query))
        # BM25 điểm 0 = không khớp từ nào: không phải kết quả, bỏ đi (thứ tự đệm là tùy ý).
        return [(i, s) for i, s in self._top(scores, k) if s > 0]

    def vector(self, query: str, k: int = 10) -> list[tuple[str, float]]:
        qv = embed_queries([query])[0]
        return self._top(self.vectors @ qv, k)

    def hybrid(self, query: str, k: int = 10, pool: int = 20) -> list[tuple[str, float]]:
        # BM25 điểm 0 = không khớp từ nào: không được tính vào danh sách lexical, nếu không
        # các doc đệm (thứ tự tùy ý) sẽ nhận điểm RRF và gây nhiễu.
        fused = rrf([
            [i for i, score in self.keyword(query, pool) if score > 0],
            [i for i, _ in self.vector(query, pool)],
        ])
        return fused[:k]

    def rerank(self, query: str, candidates: list[tuple[str, float]], k: int = 5):
        """Chấm lại các ứng viên bằng cross-encoder -> top-k."""
        text_of = dict(zip(self.ids, self.texts))
        scores = rerank_scores(query, [text_of[i] for i, _ in candidates])
        order = np.argsort(-scores)[:k]
        return [(candidates[i][0], float(scores[i])) for i in order]

    def _top(self, scores: np.ndarray, k: int) -> list[tuple[str, float]]:
        order = np.argsort(-scores)[:k]
        return [(self.ids[i], float(scores[i])) for i in order]


def rrf(rankings: list[list[str]], k: int = 60) -> list[tuple[str, float]]:
    """Reciprocal Rank Fusion: score(d) = sum 1 / (k + rank). Chỉ dùng thứ hạng,
    nên trộn được BM25 (thang điểm ~0-20) với cosine (thang ~0-1) mà không cần chuẩn hóa."""
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank + 1)
    return sorted(scores.items(), key=lambda kv: kv[1], reverse=True)


# --- metrics (slide 21) ------------------------------------------------------
def recall_at_k(retrieved: list[str], relevant: list[str], k: int) -> float:
    """Trong các doc liên quan, bao nhiêu % nằm trong top-k."""
    return len(set(retrieved[:k]) & set(relevant)) / len(relevant)


def precision_at_k(retrieved: list[str], relevant: list[str], k: int) -> float:
    """Trong top-k, bao nhiêu % thực sự liên quan."""
    return len(set(retrieved[:k]) & set(relevant)) / k


def reciprocal_rank(retrieved: list[str], relevant: list[str]) -> float:
    """1 / vị trí của doc liên quan đầu tiên (0 nếu không có)."""
    for rank, doc_id in enumerate(retrieved, start=1):
        if doc_id in relevant:
            return 1.0 / rank
    return 0.0
