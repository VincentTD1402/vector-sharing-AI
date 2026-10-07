"""Demo 06 - Exact search không scale -> ANN, IVF tự viết (slide 12, 14).

A. Brute force: thời gian mỗi query tăng tuyến tính theo N (O(N x D)).
B. IVF (Inverted File) viết tay ~25 dòng: K-means chia không gian thành nlist cụm,
   mỗi query chỉ quét `nprobe` cụm gần nhất. Vẽ đường cong recall vs latency.
   -> nprobe = nlist chính là exact search; nprobe nhỏ = nhanh nhưng có thể sót.

Không cần model/DB: dùng vector tổng hợp có cấu trúc cụm (lib/ann_bench.py).

Chạy:  python demos/06_exact_vs_ann.py [--open]
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import plotly.graph_objects as go
from sklearn.cluster import MiniBatchKMeans

from lib.ann_bench import exact_top_k, make_clustered_vectors, make_queries
from lib.config import EMBED_DIM
from lib.viz import banner, save_figure

K = 10
N_QUERIES = 50


def timed_ms(fn, repeat: int = 1) -> float:
    start = time.perf_counter()
    for _ in range(repeat):
        fn()
    return (time.perf_counter() - start) / repeat * 1000


def brute_force_scaling() -> None:
    banner("A. Brute force: latency tăng tuyến tính theo số vector", "12")
    print(f"  D = {EMBED_DIM} chiều. Mỗi query = N x D phép nhân-cộng.")
    print(f"  {'N vector':>10} {'phép tính/query':>18} {'latency/query':>15} {'RAM vectors':>12}")
    for n in (10_000, 100_000, 400_000):
        vecs = make_clustered_vectors(n, seed=1)
        q = vecs[0]
        ms = timed_ms(lambda: exact_top_k(vecs, q, K), repeat=5)
        print(f"  {n:>10,} {n * EMBED_DIM / 1e6:>15,.0f} M {ms:>12.1f} ms {vecs.nbytes / 1e6:>9.0f} MB")
    print("  Ngoại suy: 100M vector ~ 250x của 400K => hàng giây/query và ~150 GB RAM -> không chấp nhận được.")


class IVFIndex:
    """IVFFlat tối giản: centroid (K-means) + danh sách vector (inverted list) cho mỗi cụm."""

    def __init__(self, vectors: np.ndarray, nlist: int):
        # Training: K-means tìm nlist tâm cụm
        km = MiniBatchKMeans(n_clusters=nlist, random_state=0, n_init=1, batch_size=4096).fit(vectors)
        self.centroids = km.cluster_centers_.astype(np.float32)
        # Indexing: gán mỗi vector vào centroid gần nhất; lưu vector của mỗi cụm liền kề
        # trong bộ nhớ (như inverted list thật) để quét một cụm không phải copy dữ liệu.
        self.ids = [np.where(km.labels_ == c)[0] for c in range(nlist)]
        self.cells = [vectors[ids] for ids in self.ids]

    def search(self, query: np.ndarray, k: int, nprobe: int) -> np.ndarray:
        # 1) tìm nprobe centroid gần query nhất (chỉ nlist phép so sánh)
        nearest_cells = np.argsort(-(self.centroids @ query))[:nprobe]
        # 2) chỉ quét vector thuộc các cụm đó
        scores = np.concatenate([self.cells[c] @ query for c in nearest_cells])
        cand = np.concatenate([self.ids[c] for c in nearest_cells])
        return cand[np.argsort(-scores)[:k]]


def ivf_recall_vs_latency() -> None:
    banner("B. IVF tự viết: đánh đổi recall - latency qua tham số nprobe", "12, 14")
    n, nlist = 200_000, 256
    vecs = make_clustered_vectors(n, seed=2)
    queries = make_queries(vecs, N_QUERIES)  # query nằm gần vùng có dữ liệu, như thực tế
    print(f"  N = {n:,} vector, nlist = {nlist} cụm. Đang train K-means...")
    index = IVFIndex(vecs, nlist)
    truth = [set(exact_top_k(vecs, q, K)) for q in queries]
    exact_ms = np.mean([timed_ms(lambda q=q: exact_top_k(vecs, q, K)) for q in queries])
    print(f"  Exact search: {exact_ms:.1f} ms/query (recall = 100%)\n")
    print(f"  {'nprobe':>7} {'recall@10':>10} {'ms/query':>10} {'nhanh hơn':>10}")

    rows = []
    for nprobe in (1, 2, 4, 8, 16, 32, 64, nlist):
        recall = np.mean([len(set(index.search(q, K, nprobe)) & t) / K for q, t in zip(queries, truth)])
        ms = np.mean([timed_ms(lambda q=q: index.search(q, K, nprobe)) for q in queries])
        rows.append((nprobe, recall * 100, ms))
        note = "  <- quét hết = exact (chậm hơn vì 256 phép nhân nhỏ + overhead Python)" if nprobe == nlist else ""
        print(f"  {nprobe:>7} {recall * 100:>9.1f}% {ms:>10.2f} {exact_ms / ms:>9.1f}x{note}")

    fig = go.Figure(go.Scatter(
        x=[r[2] for r in rows], y=[r[1] for r in rows], mode="lines+markers+text",
        text=[f"nprobe={r[0]}" for r in rows], textposition="bottom right",
    ))
    fig.add_vline(x=exact_ms, line_dash="dot", annotation_text="exact search")
    fig.update_layout(title="IVF: recall vs latency (mỗi điểm = một giá trị nprobe)",
                      xaxis_title="latency (ms/query)", yaxis_title="recall@10 (%)", height=520)
    save_figure(fig, "06_ivf_recall_vs_latency.html")
    print("  ANN = đổi một ít recall lấy tốc độ. Tăng nprobe -> recall tăng, latency tăng.")


if __name__ == "__main__":
    brute_force_scaling()
    ivf_recall_vs_latency()
