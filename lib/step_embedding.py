"""Bước Embedding của UI: tách từng bước bên trong model.encode() và so sánh similarity (slide 5, 8, 10)."""
import time

import numpy as np

from lib.models import embed_passages, embed_queries, get_embedder
from lib.space_projection import detect_language

MAX_TEXTS = 6


def embed_trace(text: str, as_query: bool = True, n_dims: int = 48) -> dict:
    """Tokenizer -> Transformer -> mean pooling -> normalize, kiểm tra khớp với model.encode()."""
    import torch

    st = get_embedder()
    transformer = st[0].auto_model.eval()
    prefixed = ("query: " if as_query else "passage: ") + text

    t = time.perf_counter()
    enc = st.tokenizer(prefixed, return_tensors="pt", truncation=True, max_length=st.max_seq_length)
    ms_tok = (time.perf_counter() - t) * 1000
    tokens = st.tokenizer.convert_ids_to_tokens(enc["input_ids"][0])

    t = time.perf_counter()
    with torch.no_grad():
        hidden = transformer(**enc).last_hidden_state[0]  # (số_token, 384): mỗi token một vector ngữ cảnh
    ms_tf = (time.perf_counter() - t) * 1000

    t = time.perf_counter()
    pooled = hidden.mean(dim=0)
    final = (pooled / pooled.norm()).numpy()
    ms_pool = (time.perf_counter() - t) * 1000

    reference = (embed_queries if as_query else embed_passages)([text])[0]
    return {
        "tokens": tokens, "token_ids": enc["input_ids"][0].tolist(), "hidden_shape": list(hidden.shape),
        "pooled_norm": round(float(pooled.norm()), 3), "dim": int(final.shape[0]),
        "dims": [round(float(x), 4) for x in final[:n_dims]], "norm": round(float(np.linalg.norm(final)), 3),
        "matches_encode": bool(np.allclose(final, reference, atol=1e-4)),
        "ms": {"tokenize": round(ms_tok, 2), "transformer": round(ms_tf, 1), "pooling": round(ms_pool, 2)},
    }


def compare_texts(query: str, texts: list[str], use_prefix: bool = True, scale_last: float = 1.0) -> dict:
    """Xếp hạng `texts` theo cosine / dot / euclid so với `query`; `scale_last` phóng to vector doc cuối
    để thấy dot/euclid bị đánh lừa còn cosine thì không."""
    texts = texts[:MAX_TEXTS]
    st = get_embedder()
    enc = (lambda xs, fn: fn(xs)) if use_prefix else (
        lambda xs, fn: st.encode(xs, normalize_embeddings=True, show_progress_bar=False).astype(np.float32))
    q = enc([query], embed_queries)[0]
    docs = enc(texts, embed_passages).copy()
    if scale_last != 1.0 and len(docs):
        docs[-1] = docs[-1] * scale_last
    cosine = docs @ q / (np.linalg.norm(docs, axis=1) * np.linalg.norm(q))
    dot = docs @ q
    euclid = np.linalg.norm(docs - q, axis=1)

    def ranks(values, descending):
        order = np.argsort(-values if descending else values)
        out = np.empty(len(values), dtype=int)
        out[order] = np.arange(1, len(values) + 1)
        return out

    rc, rd, re_ = ranks(cosine, True), ranks(dot, True), ranks(euclid, False)
    return {
        "query": query, "use_prefix": use_prefix, "scale_last": scale_last,
        "rows": [{"text": t, "language": detect_language(t), "norm": round(float(np.linalg.norm(docs[i])), 2),
                  "cosine": round(float(cosine[i]), 4), "dot": round(float(dot[i]), 4),
                  "euclid": round(float(euclid[i]), 4),
                  "rank_cosine": int(rc[i]), "rank_dot": int(rd[i]), "rank_euclid": int(re_[i])}
                 for i, t in enumerate(texts)],
    }
