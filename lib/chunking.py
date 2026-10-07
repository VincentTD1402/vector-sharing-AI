"""Các chiến lược chunking: fixed-size (theo token, có overlap), sentence, recursive.

Mỗi chiến lược trả về SPAN (start, end) trên văn bản gốc - chunk luôn là chuỗi nguyên văn
`text[start:end]`, nên UI có thể tô màu từng chunk và thấy rõ ranh giới/overlap.
"""
import re
from functools import lru_cache

from lib.models import get_embedder

SEPARATORS = ("\n\n", "\n", ". ", " ")
Span = tuple[int, int]


@lru_cache
def get_tokenizer():
    return get_embedder().tokenizer


def count_tokens(text: str) -> int:
    return len(get_tokenizer()(text, add_special_tokens=False)["input_ids"])


def _strip_span(text: str, start: int, end: int) -> Span | None:
    piece = text[start:end]
    stripped = piece.strip()
    if not stripped:
        return None
    lead = len(piece) - len(piece.lstrip())
    return start + lead, start + lead + len(stripped)


def fixed_size_spans(text: str, size: int = 256, overlap: int = 0) -> list[Span]:
    """Cửa sổ trượt `size` token, bước (size - overlap)."""
    offsets = get_tokenizer()(text, add_special_tokens=False, return_offsets_mapping=True)["offset_mapping"]
    step = max(1, size - overlap)
    spans = []
    for start in range(0, len(offsets), step):
        end = min(start + size, len(offsets))
        spans.append((offsets[start][0], offsets[end - 1][1]))
        if end == len(offsets):
            break
    return spans


def sentence_spans(text: str, max_tokens: int = 128) -> list[Span]:
    """Gom các câu liên tiếp cho tới khi sắp vượt `max_tokens` - không bao giờ cắt giữa câu."""
    sentences, cursor = [], 0
    for sentence in re.split(r"(?<=[.!?])\s+|\n+", text):
        sentence = sentence.strip()
        if not sentence:
            continue
        start = text.index(sentence, cursor)
        sentences.append((start, start + len(sentence)))
        cursor = start + len(sentence)
    spans, current, current_tokens = [], None, 0
    for start, end in sentences:
        n = count_tokens(text[start:end])
        if current and current_tokens + n > max_tokens:
            spans.append(current)
            current, current_tokens = None, 0
        current = (current[0], end) if current else (start, end)
        current_tokens += n
    if current:
        spans.append(current)
    return spans


def recursive_spans(text: str, max_tokens: int = 128, separators=SEPARATORS) -> list[Span]:
    """Thử tách theo separator lớn nhất (đoạn) trước; phần nào còn quá dài thì đệ quy xuống
    separator nhỏ hơn (dòng -> câu -> từ). Cùng ý tưởng RecursiveCharacterTextSplitter."""
    if count_tokens(text) <= max_tokens or not separators:
        span = _strip_span(text, 0, len(text))
        return [span] if span else []
    sep, rest = separators[0], separators[1:]
    pieces, pos = [], 0
    for piece in text.split(sep):
        pieces.append((pos, pos + len(piece)))
        pos += len(piece) + len(sep)
    spans, current = [], None
    for start, end in pieces:
        candidate = (current[0], end) if current else (start, end)
        if count_tokens(text[candidate[0]:candidate[1]]) <= max_tokens:
            current = candidate
            continue
        if current:
            spans.append(current)
        if count_tokens(text[start:end]) > max_tokens:  # tự nó đã quá dài -> đệ quy
            spans.extend((start + s, start + e) for s, e in recursive_spans(text[start:end], max_tokens, rest))
            current = None
        else:
            current = (start, end)
    if current:
        spans.append(current)
    return [s for s in (_strip_span(text, a, b) for a, b in spans) if s]


def chunk_spans(text: str, strategy: str, size: int, overlap: int = 0) -> list[Span]:
    if strategy == "fixed":
        return fixed_size_spans(text, size, overlap)
    if strategy == "sentence":
        return sentence_spans(text, size)
    if strategy == "recursive":
        return recursive_spans(text, size)
    raise ValueError(f"chiến lược không hỗ trợ: {strategy}")


def spans_to_texts(text: str, spans: list[Span]) -> list[str]:
    return [t for t in (text[a:b].strip() for a, b in spans) if t]


def fixed_size_chunks(text: str, size: int = 256, overlap: int = 0) -> list[str]:
    return spans_to_texts(text, fixed_size_spans(text, size, overlap))


def sentence_chunks(text: str, max_tokens: int = 128) -> list[str]:
    return spans_to_texts(text, sentence_spans(text, max_tokens))


def recursive_chunks(text: str, max_tokens: int = 128) -> list[str]:
    return spans_to_texts(text, recursive_spans(text, max_tokens))
