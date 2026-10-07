"""Bài học Chunking: chạy một tình huống (tài liệu + cấu hình + câu hỏi + đáp án) và trả dữ liệu từng bước.

Điểm mấu chốt: đáp án có NGUYÊN VẸN trong ít nhất một chunk không? Nếu bị cắt đôi ở ranh giới thì dù
retriever chọn đúng, LLM cũng chỉ thấy một nửa đáp án.
"""
import time

import numpy as np

from lib.chunking import chunk_spans, count_tokens, get_tokenizer
from lib.models import embed_passages, embed_queries
from lib.step_chunking import resolve_text


def _ms(start: float) -> float:
    return round((time.perf_counter() - start) * 1000, 1)


def locate_answer(text: str, answer: str, chunks: list[dict]) -> dict:
    """Vị trí đáp án và tình trạng so với ranh giới chunk: intact | split | missing | none."""
    if not answer.strip():
        return {"status": "none", "text": ""}
    start = text.lower().find(answer.lower())
    if start < 0:
        return {"status": "missing", "text": answer}
    end = start + len(answer)
    contained = [c["i"] for c in chunks if c["start"] <= start and c["end"] >= end]
    touching = [c["i"] for c in chunks if c["start"] < end and c["end"] > start]
    # ranh giới (chỗ chunk này kết thúc / chunk khác bắt đầu) nằm LỌT vào giữa đáp án
    inside = sorted({p for c in chunks for p in (c["start"], c["end"]) if start < p < end})
    return {"status": "intact" if contained else "split", "text": answer, "start": start, "end": end,
            "contained_in": contained, "touching": touching, "boundaries_inside": inside}


def run(source: str, strategy: str, size: int, overlap: int, query: str, answer: str, custom: str = "") -> dict:
    text = resolve_text(source, custom)
    if not text.strip():
        raise ValueError("Văn bản trống")

    t = time.perf_counter()
    ids = get_tokenizer()(text, add_special_tokens=False)["input_ids"]
    tokens_preview = get_tokenizer().convert_ids_to_tokens(ids[:14])
    ms_tok = _ms(t)

    t = time.perf_counter()
    spans = chunk_spans(text, strategy, size, overlap)
    chunks = [{"i": i, "start": a, "end": b, "tokens": count_tokens(text[a:b]), "text": text[a:b]}
              for i, (a, b) in enumerate(spans)]
    ms_chunk = _ms(t)

    t = time.perf_counter()
    vectors = embed_passages([c["text"] for c in chunks])
    ms_embed = _ms(t)

    t = time.perf_counter()
    qv = embed_queries([query])[0]
    scores = vectors @ qv
    ms_query = _ms(t)
    order = [int(i) for i in np.argsort(-scores)]
    rank_of = {i: r for r, i in enumerate(order, 1)}

    ans = locate_answer(text, answer, chunks)
    contained = ans.get("contained_in", [])
    return {
        "doc": {"source": source, "chars": len(text), "tokens": len(ids), "head": text[:260]},
        "tokens_preview": tokens_preview,
        "text": text,
        "config": {"strategy": strategy, "size": size, "overlap": overlap},
        "chunks": chunks,
        "answer": ans,
        "ranking": [{"i": i, "rank": rank_of[i], "score": round(float(scores[i]), 4),
                     "has_full": i in contained, "has_part": i in ans.get("touching", []) and i not in contained}
                    for i in order],
        "answer_rank": min((rank_of[i] for i in contained), default=None),
        "ctx_tokens_top3": int(sum(chunks[i]["tokens"] for i in order[:3])),
        "avg_tokens": round(float(np.mean([c["tokens"] for c in chunks])), 1),
        "dim": int(vectors.shape[1]),
        "ms": {"tokenize": ms_tok, "chunk": ms_chunk, "embed_chunks": ms_embed, "embed_query": ms_query},
    }
