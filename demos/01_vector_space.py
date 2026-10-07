"""Demo 01 - Vector space: vị trí có ý nghĩa (slide 3, 4, 5).

A. Toy 2D: vài điểm đặt tay để thấy "gần nhau = giống nhau".
B. Embedding thật (384 chiều) của corpus, chiếu xuống 2D/3D bằng PCA, kèm số đo láng giềng gần nhất.
   Quan sát thật: với model đa ngôn ngữ nhỏ này, NGÔN NGỮ là trục biến thiên lớn nhất
   (tiếng Việt và tiếng Anh tách thành hai vùng); chủ đề là cấu trúc nhỏ hơn bên trong mỗi vùng.
C. Trừ trung bình theo ngôn ngữ rồi chiếu lại: cấu trúc chủ đề lộ ra rõ hơn.

Chạy:  python demos/01_vector_space.py [--open]
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import plotly.graph_objects as go
from sklearn.decomposition import PCA

from lib.data import load_corpus
from lib.models import embed_passages_cached, embed_queries
from lib.viz import banner, save_figure

QUERIES = ["I forgot my password", "How do I create a production order?"]


def toy_2d() -> None:
    banner("A. Vector là điểm trong không gian - toy 2D", "3")
    points = {  # (x, y, vị trí nhãn)
        "reset password": (4.5, 4.2, "bottom center"),
        "forgot password": (4.0, 4.8, "top center"),
        "change password": (4.8, 3.7, "bottom center"),
        "create order": (1.0, 1.2, "top center"),
        "production order": (1.7, 0.8, "bottom center"),
    }
    names = list(points)
    xy = np.array([points[n][:2] for n in names])

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=xy[:, 0], y=xy[:, 1], mode="markers+text", text=names,
        textposition=[points[n][2] for n in names],
        marker=dict(size=16, color=["#2a9d8f"] * 3 + ["#e76f51"] * 2),
    ))
    for a, b in [("reset password", "forgot password"), ("reset password", "create order")]:
        pa, pb = np.array(points[a][:2]), np.array(points[b][:2])
        dist = np.linalg.norm(pa - pb)
        print(f"  khoảng cách({a!r}, {b!r}) = {dist:.2f}")
        fig.add_shape(type="line", x0=pa[0], y0=pa[1], x1=pb[0], y1=pb[1], line=dict(dash="dot", color="gray"))
        fig.add_annotation(x=(pa[0] + pb[0]) / 2, y=(pa[1] + pb[1]) / 2, text=f"d = {dist:.2f}",
                           showarrow=False, bgcolor="white", yshift=14)
    fig.update_layout(title="Toy 2D: vị trí tương đối = quan hệ đo được",
                      xaxis_title="x", yaxis_title="y", height=520, showlegend=False)
    save_figure(fig, "01a_toy_2d.html")


def neighbor_stats(docs: list[dict], vectors: np.ndarray) -> None:
    """Láng giềng gần nhất (cosine) của mỗi doc: cùng ngôn ngữ? cùng chủ đề?"""
    sims = vectors @ vectors.T
    np.fill_diagonal(sims, -1)
    nn = sims.argmax(axis=1)
    same_lang = np.mean([docs[i]["language"] == docs[j]["language"] for i, j in enumerate(nn)])
    same_topic = np.mean([docs[i]["topic"] == docs[j]["topic"] for i, j in enumerate(nn)])
    print(f"  Láng giềng gần nhất của mỗi doc: cùng ngôn ngữ {same_lang:.0%}, cùng chủ đề {same_topic:.0%}")


def plot_map(docs, coords, query_coords, dims: int, title: str, filename: str) -> None:
    fig = go.Figure()
    scatter = go.Scatter if dims == 2 else go.Scatter3d
    axes = "xyz"[:dims]
    for topic in sorted({d["topic"] for d in docs}):
        idx = [i for i, d in enumerate(docs) if d["topic"] == topic]
        fig.add_trace(scatter(
            **{a: coords[idx, j] for j, a in enumerate(axes)}, mode="markers", name=topic,
            marker=dict(size=9 if dims == 2 else 5,
                        symbol=["circle" if docs[i]["language"] == "vi" else "diamond" for i in idx]),
            text=[f"[{docs[i]['language']}] {docs[i]['text'][:90]}" for i in idx],
            hovertemplate="%{text}<extra>" + topic + "</extra>",
        ))
    fig.add_trace(scatter(
        **{a: query_coords[:, j] for j, a in enumerate(axes)}, mode="markers+text", name="query",
        text=QUERIES, textposition="top center",
        marker=dict(size=14 if dims == 2 else 8, color="black", symbol="star" if dims == 2 else "x"),
    ))
    fig.update_layout(title=title, height=680)
    save_figure(fig, filename)


def real_embeddings() -> None:
    banner("B. Embedding thật: 384 chiều -> chiếu xuống 2D/3D", "3, 5")
    docs = load_corpus()
    vectors = embed_passages_cached([d["text"] for d in docs])
    qvecs = embed_queries(QUERIES)
    print(f"  Một vector: shape={vectors[0].shape}, độ dài (norm)={np.linalg.norm(vectors[0]):.3f}")
    print(f"  8 số đầu của '{docs[0]['text'][:40]}...': {np.round(vectors[0][:8], 3)}")
    neighbor_stats(docs, vectors)
    print("  -> Ngôn ngữ chi phối hình học: tìm chéo ngôn ngữ (query Anh -> doc Việt) kém hơn cùng ngôn ngữ.")

    for dims in (2, 3):  # PCA fit trên docs, rồi chiếu thêm query vào cùng không gian
        pca = PCA(n_components=dims, random_state=0).fit(vectors)
        kept = pca.explained_variance_ratio_.sum() * 100
        print(f"  PCA {dims}D giữ {kept:.1f}% phương sai -> hình chỉ là minh họa, mất phần lớn thông tin.")
        plot_map(docs, pca.transform(vectors), pca.transform(qvecs), dims,
                 f"Embedding space ({dims}D PCA, giữ {kept:.0f}% phương sai) - tròn: tiếng Việt, thoi: tiếng Anh",
                 f"01b_embedding_map_{dims}d.html")


def language_centered() -> None:
    banner("C. Trừ trung bình theo ngôn ngữ: để lộ cấu trúc CHỦ ĐỀ", "5")
    docs = load_corpus()
    vectors = embed_passages_cached([d["text"] for d in docs])
    qvecs = embed_queries(QUERIES)
    langs = np.array([d["language"] for d in docs])
    centered = vectors.copy()
    for lang in set(langs):
        centered[langs == lang] -= vectors[langs == lang].mean(axis=0)
    # Query tiếng Anh: trừ trung bình của nhóm tiếng Anh.
    q_centered = qvecs - vectors[langs == "en"].mean(axis=0)
    pca = PCA(n_components=2, random_state=0).fit(centered)
    kept = pca.explained_variance_ratio_.sum() * 100
    print(f"  Sau khi bỏ 'hướng ngôn ngữ', PCA 2D giữ {kept:.1f}% phương sai. So với bước B:")
    neighbor_stats(docs, centered / np.linalg.norm(centered, axis=1, keepdims=True))
    plot_map(docs, pca.transform(centered), pca.transform(q_centered), 2,
             "Embedding space đã trừ trung bình theo ngôn ngữ (2D PCA) - các cụm chủ đề hiện rõ hơn, vi/en trộn lẫn",
             "01c_embedding_map_language_centered.html")


if __name__ == "__main__":
    toy_2d()
    real_embeddings()
    language_centered()
    print("\nQuan sát: query (sao đen) rơi gần cụm đúng chủ đề; bản đồ gốc tách theo ngôn ngữ, bản trừ-ngôn-ngữ tách theo chủ đề.")
