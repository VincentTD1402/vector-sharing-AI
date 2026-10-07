"""Demo 02 - Từ text đến vector: tokenizer -> transformer -> pooling (slide 5, 8).

Tự làm lại từng bước bên trong model.encode() rồi kiểm tra kết quả khớp.
Cuối cùng minh họa: embedding = mẫu thống kê, KHÔNG phải "hiểu ngôn ngữ".

Chạy:  python demos/02_embedding_pipeline.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

from lib.config import EMBED_MODEL
from lib.data import active_docs, load_eval_queries
from lib.models import embed_passages, embed_queries, get_embedder
from lib.search import recall_at_k
from lib.viz import banner


def step_by_step(text: str) -> None:
    banner("A. Pipeline chi tiết: Text -> Tokenizer -> Transformer -> Pooling -> Vector", "8")
    st = get_embedder()
    tokenizer, transformer = st.tokenizer, st[0].auto_model.eval()
    text = "query: " + text  # e5 cần prefix

    # 1) Tokenizer: text -> token (subword) -> id
    enc = tokenizer(text, return_tensors="pt")
    tokens = tokenizer.convert_ids_to_tokens(enc["input_ids"][0])
    print(f"1) Tokenizer  : {tokens}")
    print(f"   token ids  : {enc['input_ids'][0].tolist()}")

    # 2) Transformer: mỗi token -> 1 vector ngữ cảnh (contextual)
    with torch.no_grad():
        hidden = transformer(**enc).last_hidden_state[0]  # (số_token, 384)
    print(f"2) Transformer: {hidden.shape[0]} token x {hidden.shape[1]} chiều (mỗi token một vector)")

    # 3) Mean pooling: gộp nhiều vector token thành 1 vector cho cả câu
    pooled = hidden.mean(dim=0)
    print(f"3) Mean pooling: -> shape {tuple(pooled.shape)}")

    # 4) Normalize: độ dài = 1 để cosine == dot product
    final = (pooled / pooled.norm()).numpy()
    print(f"4) Normalize  : norm = {np.linalg.norm(final):.3f};  8 số đầu = {np.round(final[:8], 3)}")

    reference = embed_queries([text.removeprefix('query: ')])[0]
    print(f"\nKhớp với model.encode()? {np.allclose(final, reference, atol=1e-4)}"
          f"  (max lệch = {np.abs(final - reference).max():.2e})")


def model_info() -> None:
    banner("B. Thông số model đang dùng", "8, 9")
    st = get_embedder()
    print(f"  Model        : {EMBED_MODEL}")
    print(f"  Số chiều     : {st.get_embedding_dimension()}")
    print(f"  Max tokens   : {st.max_seq_length}  (tài liệu dài hơn phải chunking)")
    print(f"  Số tham số   : {sum(p.numel() for p in st.parameters()) / 1e6:.0f}M")
    print("  Nguyên tắc   : không trộn vector của 2 model khác nhau - đổi model = re-index toàn bộ.")


def embedding_is_statistics() -> None:
    banner("C. Embedding nắm bắt mẫu thống kê, không phải 'hiểu'", "5")
    anchor = "I forgot my password"
    candidates = [
        "How to reset your password",          # đồng nghĩa, nên gần
        "I remembered my password",            # ngược nghĩa nhưng cùng từ vựng/ngữ cảnh
        "I never want to see my password again",
        "Create a production order",           # khác chủ đề
    ]
    qv = embed_queries([anchor])[0]
    scores = embed_passages(candidates) @ qv
    print(f"  Anchor: {anchor!r}")
    for c, s in sorted(zip(candidates, scores), key=lambda x: -x[1]):
        print(f"    cosine = {s:.3f}   {c}")
    print("  -> 'ngược nghĩa' vẫn rất gần vì cùng chủ đề. Embedding đo 'cùng ngữ cảnh', không đo 'đúng/sai'.")
    print("  -> Lưu ý: e5 nén điểm vào dải hẹp (~0.7-1.0), kể cả doc khác chủ đề vẫn ~0.75.")
    print("     Đừng bê ngưỡng cosine từ slide/model khác sang; hãy hiệu chỉnh trên dữ liệu của bạn.")


def prefix_matters() -> None:
    banner("D. Chi tiết dễ sai: prefix 'query:' / 'passage:' của e5 - đo thật trên bộ eval", "8")
    docs = active_docs("company_a")
    queries = load_eval_queries()
    st = get_embedder()
    doc_texts = [d["text"] for d in docs]
    ids = [d["id"] for d in docs]

    def mean_recall(doc_vecs, query_vecs) -> float:
        total = 0.0
        for q, qv in zip(queries, query_vecs):
            top = [ids[i] for i in np.argsort(-(doc_vecs @ qv))[:3]]
            total += recall_at_k(top, q["relevant"], 3)
        return total / len(queries)

    q_text = [q["query"] for q in queries]
    with_prefix = mean_recall(embed_passages(doc_texts), embed_queries(q_text))
    no_prefix = mean_recall(
        st.encode(doc_texts, normalize_embeddings=True, show_progress_bar=False),
        st.encode(q_text, normalize_embeddings=True, show_progress_bar=False),
    )
    print(f"  Recall@3 trung bình trên {len(queries)} query:")
    print(f"    có prefix  : {with_prefix:.3f}")
    print(f"    thiếu prefix: {no_prefix:.3f}")
    print("  Chênh lệch này nằm trong mức nhiễu của 32 query (1 query ~ 3 điểm %), nên KHÔNG kết luận được")
    print("  prefix 'tốt' hay 'xấu' trên corpus nhỏ này. Bài học: quy ước prefix là tham số phải kiểm chứng")
    print("  bằng bộ eval đủ lớn, và nhất quán giữa index-time và query-time. Mỗi model một quy ước: đọc model card.")


if __name__ == "__main__":
    step_by_step("How to reset your password?")
    model_info()
    embedding_is_statistics()
    prefix_matters()
