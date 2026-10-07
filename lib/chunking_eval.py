"""Đo ảnh hưởng của chunking tới retrieval trên các sổ tay + bộ câu hỏi có đáp án (dùng chung CLI và UI).

Hit = chunk CHỨA chuỗi đáp án. Mỗi câu chỉ có 1-2 chunk chứa đáp án nên Precision@k bị chặn
trên 1/k, không phân biệt được cấu hình -> dùng Recall@1 và MRR, kèm số token phải đưa cho LLM.
"""
import numpy as np

from lib.chunking import chunk_spans, count_tokens, spans_to_texts
from lib.data import load_manual_questions, load_manuals
from lib.models import embed_passages, embed_queries

K = 3
# (nhãn, chiến lược, size, overlap): các cấu hình so sánh trong sweep. 500 chừa chỗ cho prefix + token đặc biệt (max 512).
SWEEP = [(f"fixed {s} tok", "fixed", s, 0) for s in (32, 64, 128, 256, 500)] + [
    ("fixed 128 tok + overlap 32", "fixed", 128, 32),
    ("sentence ≤128 tok", "sentence", 128, 0),
    ("recursive ≤128 tok", "recursive", 128, 0),
]


def evaluate(strategy) -> dict:
    """Index toàn bộ sổ tay bằng `strategy(text) -> list[str]`, rồi hỏi từng câu."""
    chunks = [c for text in load_manuals().values() for c in strategy(text)]
    token_counts = np.array([count_tokens(c) for c in chunks])
    vectors = embed_passages(chunks)
    questions = load_manual_questions()
    qvecs = embed_queries([q["query"] for q in questions])
    r1, r3, rr, ctx = [], [], [], []
    for q, qv in zip(questions, qvecs):
        order = np.argsort(-(vectors @ qv))
        has_answer = [q["answer"].lower() in chunks[i].lower() for i in order]
        first = has_answer.index(True) + 1 if any(has_answer) else None
        r1.append(first == 1)
        r3.append(first is not None and first <= K)
        rr.append(1 / first if first else 0.0)
        ctx.append(token_counts[order[:K]].sum())  # token phải đưa cho LLM nếu lấy top-K
    return {
        "chunks": len(chunks), "avg_tokens": float(token_counts.mean()),
        "recall@1": float(np.mean(r1)), f"recall@{K}": float(np.mean(r3)),
        "mrr": float(np.mean(rr)), "ctx_tokens": float(np.mean(ctx)),
    }


def evaluate_config(strategy: str, size: int, overlap: int = 0) -> dict:
    return evaluate(lambda text: spans_to_texts(text, chunk_spans(text, strategy, size, overlap)))


def sweep() -> list[dict]:
    return [{"label": label, "strategy": s, "size": size, "overlap": ov, **evaluate_config(s, size, ov)}
            for label, s, size, ov in SWEEP]
