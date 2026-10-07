"""Chiếu embedding xuống 2D (PCA) để UI vẽ vị trí tài liệu và query trong không gian vector (slide 3, 5).

Hai chế độ: "raw" (PCA gốc - trục lớn nhất là NGÔN NGỮ) và "centered" (đã trừ trung bình theo
ngôn ngữ - lộ cấu trúc chủ đề). Xem demos/01_vector_space.py.
"""
import re

import numpy as np
from sklearn.decomposition import PCA

from lib.data import load_corpus
from lib.models import embed_passages, embed_passages_cached, embed_queries, get_embedder

_VI_CHARS = re.compile(r"[àáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđ]", re.I)


def detect_language(text: str) -> str:
    return "vi" if _VI_CHARS.search(text) else "en"


def embedding_info(query: str, n_dims: int = 48) -> dict:
    """Token hóa + vector của query: dữ liệu cho panel 'Embedding'."""
    tokenizer = get_embedder().tokenizer
    tokens = tokenizer.convert_ids_to_tokens(tokenizer("query: " + query)["input_ids"])
    vec = embed_queries([query])[0]
    return {"tokens": tokens, "dims": [round(float(x), 4) for x in vec[:n_dims]],
            "dim": int(vec.shape[0]), "norm": round(float(np.linalg.norm(vec)), 3)}


class SpaceProjector:
    def __init__(self):
        self.docs = load_corpus()
        vectors = embed_passages_cached([d["text"] for d in self.docs])
        langs = np.array([d["language"] for d in self.docs])
        self.lang_mean = {lang: vectors[langs == lang].mean(axis=0) for lang in set(langs)}
        centered = vectors - np.stack([self.lang_mean[lang] for lang in langs])
        self.pca = {
            "raw": PCA(n_components=2, random_state=0).fit(vectors),
            "centered": PCA(n_components=2, random_state=0).fit(centered),
        }
        self.coords = {"raw": self.pca["raw"].transform(vectors),
                       "centered": self.pca["centered"].transform(centered)}

    def points(self) -> dict:
        bounds = {}
        for mode, xy in self.coords.items():
            lo, hi = xy.min(axis=0), xy.max(axis=0)
            pad = (hi - lo) * 0.12
            bounds[mode] = [float(lo[0] - pad[0]), float(hi[0] + pad[0]), float(lo[1] - pad[1]), float(hi[1] + pad[1])]
        return {
            "bounds": bounds,
            "docs": [{"id": d["id"], "topic": d["topic"], "language": d["language"], "tenant_id": d["tenant_id"],
                      "status": d["status"], "text": d["text"][:110],
                      "raw": [float(x) for x in self.coords["raw"][i]],
                      "centered": [float(x) for x in self.coords["centered"][i]]}
                     for i, d in enumerate(self.docs)],
        }

    def project_vector(self, vec: np.ndarray, text: str) -> dict:
        lang = detect_language(text)
        return {
            "language": lang,
            "raw": [float(x) for x in self.pca["raw"].transform([vec])[0]],
            "centered": [float(x) for x in self.pca["centered"].transform([vec - self.lang_mean[lang]])[0]],
        }

    def project_query(self, query: str) -> dict:
        return self.project_vector(embed_queries([query])[0], query)

    def project_texts(self, texts: list[str]) -> list[dict]:
        """Chiếu nhiều văn bản (embed như passage) - dùng cho bước Embedding."""
        vecs = embed_passages(texts)
        return [self.project_vector(v, t) for v, t in zip(vecs, texts)]
