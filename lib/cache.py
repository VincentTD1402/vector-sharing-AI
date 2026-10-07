"""Semantic cache (slide 32): trả lại kết quả của một query ĐÃ gặp nếu query mới đủ giống về ngữ nghĩa.

Chỉ dùng khi chấp nhận sai lệch nhỏ. Ngưỡng quá thấp -> trả kết quả của query khác ý; quá cao -> không bao giờ hit.
Khóa cache phải gồm tenant (không để tenant này nhận kết quả đã cache của tenant khác).
"""
import numpy as np

from lib.pipeline import RetrievalService


class SemanticCache:
    def __init__(self, svc: RetrievalService, threshold: float = 0.95):
        self.svc, self.threshold = svc, threshold
        self.vectors: list[np.ndarray] = []
        self.entries: list[tuple[str, str, list[dict]]] = []  # (tenant, query gốc, kết quả)

    def lookup(self, query: str, tenant_id: str):
        """Trả về (kết quả | None, độ giống cao nhất với một mục cùng tenant, query của mục đó)."""
        qv = self.svc.embed_query(query)
        best, best_query, hit = -1.0, None, None
        for vec, (tenant, cached_query, results) in zip(self.vectors, self.entries):
            if tenant != tenant_id:
                continue
            sim = float(vec @ qv)
            if sim > best:
                best, best_query = sim, cached_query
                hit = results if sim >= self.threshold else None
        return hit, (best if best_query else None), best_query

    def put(self, query: str, tenant_id: str, results: list[dict]) -> None:
        self.vectors.append(self.svc.embed_query(query))
        self.entries.append((tenant_id, query, results))
