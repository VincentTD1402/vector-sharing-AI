"""Bài học Index & Vector DB: minh họa ANN (exact không scale, IVF, HNSW) trên dữ liệu 2D nhỏ để NHÌN thấy
thuật toán, cùng các lỗi thật của pgvector (slide 12-15, 29).

IVF/HNSW ở đây là bản tối giản viết tay (không dùng thư viện) chỉ để giải thích cơ chế; số đo thật trên 50.000
vector pgvector nằm ở lib/step_index.py.
"""
import math
import threading
import time

import numpy as np
import psycopg

from lib.db import connect


# ---------------------------------------------------------------- 1. exact search không scale
_scaling_lock = threading.Lock()  # sinh tới ~600 MB vector: chạy một lần, các lần sau dùng kết quả đã đo
_scaling_result: dict = {}


def scaling() -> dict:
    with _scaling_lock:
        if not _scaling_result:
            _scaling_result.update(_measure_scaling())
        return _scaling_result


def _measure_scaling(sizes=(10_000, 50_000, 200_000, 400_000), dim: int = 384) -> dict:
    rng = np.random.default_rng(0)
    rows = []
    for n in sizes:
        vecs = rng.standard_normal((n, dim), dtype=np.float32)
        vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
        q = vecs[0]
        times = []
        for _ in range(5):
            t = time.perf_counter()
            scores = vecs @ q
            np.argpartition(-scores, 10)[:10]
            times.append((time.perf_counter() - t) * 1000)
        rows.append({"n": n, "ms": round(float(np.median(times)), 2), "mb": round(vecs.nbytes / 1e6),
                     "ops_m": round(n * dim / 1e6)})
        del vecs
    per_vec_ms = rows[-1]["ms"] / rows[-1]["n"]
    extra = [{"n": n, "ms": round(per_vec_ms * n, 0), "gb": round(n * dim * 4 / 1e9, 1)} for n in (1_000_000, 10_000_000, 100_000_000)]
    return {"dim": dim, "rows": rows, "extrapolated": extra}


# ---------------------------------------------------------------- 2. IVF toy
def _kmeans(points: np.ndarray, k: int, rng, iters: int = 12):
    cent = points[rng.choice(len(points), k, replace=False)].copy()
    for _ in range(iters):
        assign = np.argmin(((points[:, None] - cent[None]) ** 2).sum(-1), axis=1)
        for c in range(k):
            if (assign == c).any():
                cent[c] = points[assign == c].mean(axis=0)
    return cent, np.argmin(((points[:, None] - cent[None]) ** 2).sum(-1), axis=1)


def ivf_toy(nprobe: int = 1, k_cells: int = 8, top_k: int = 5, seed: int = 3) -> dict:
    rng = np.random.default_rng(seed)
    centers = rng.uniform(0.12, 0.88, size=(k_cells, 2))
    pts = np.clip(np.concatenate([c + rng.normal(0, 0.055, size=(30, 2)) for c in centers]), 0.01, 0.99)
    cent, assign = _kmeans(pts, k_cells, rng)
    # query đặt ngay sát RANH GIỚI giữa hai cụm gần nhất để thấy trường hợp bỏ sót
    d = np.linalg.norm(cent[:, None] - cent[None], axis=-1) + np.eye(k_cells) * 9
    a, b = np.unravel_index(np.argmin(d), d.shape)
    q = cent[a] * 0.55 + cent[b] * 0.45
    order_cells = np.argsort(np.linalg.norm(cent - q, axis=1))
    probed = [int(c) for c in order_cells[:nprobe]]
    cand = np.where(np.isin(assign, probed))[0]
    found = cand[np.argsort(np.linalg.norm(pts[cand] - q, axis=1))[:top_k]]
    exact = np.argsort(np.linalg.norm(pts - q, axis=1))[:top_k]
    return {
        "points": pts.round(4).tolist(), "cell": assign.tolist(), "centroids": cent.round(4).tolist(), "query": q.round(4).tolist(),
        "cells_by_distance": [int(c) for c in order_cells], "probed": probed, "scanned": int(len(cand)), "total": len(pts),
        "found": [int(i) for i in found], "exact": [int(i) for i in exact],
        "recall": round(len(set(found.tolist()) & set(exact.tolist())) / top_k, 2),
        "nprobe": nprobe, "top_k": top_k,
    }


# ---------------------------------------------------------------- 3. HNSW toy
def hnsw_toy(ef: int = 6, top_k: int = 5, n: int = 90, m: int = 4, seed: int = 11) -> dict:
    rng = np.random.default_rng(seed)
    pts = rng.uniform(0.04, 0.96, size=(n, 2))
    ml = 1 / math.log(m)
    level = np.minimum(np.floor(-np.log(rng.uniform(1e-9, 1, n)) * ml).astype(int), 3)
    level[0] = max(level.max(), 2)  # node 0 luôn ở tầng cao nhất: điểm vào
    top = int(level.max())
    layers = []
    for l in range(top + 1):
        nodes = [int(i) for i in np.where(level >= l)[0]]
        edges = set()
        for i in nodes:
            others = sorted((j for j in nodes if j != i), key=lambda j: np.linalg.norm(pts[i] - pts[j]))[:m]
            edges.update((min(i, j), max(i, j)) for j in others)
        layers.append({"nodes": nodes, "edges": [list(e) for e in sorted(edges)]})
    nbrs = [{i: set() for i in L["nodes"]} for L in layers]
    for l, L in enumerate(layers):
        for a, b in L["edges"]:
            nbrs[l][a].add(b); nbrs[l][b].add(a)

    q = rng.uniform(0.2, 0.8, size=2)
    dist = lambda i: float(np.linalg.norm(pts[i] - q))
    path, cur = [], 0
    for l in range(top, 0, -1):  # tầng trên: tham lam, nhảy tới hàng xóm gần query nhất cho tới khi không cải thiện
        path.append({"layer": l, "node": cur, "dist": round(dist(cur), 4)})
        while True:
            best = min(nbrs[l][cur], key=dist, default=cur)
            if dist(best) >= dist(cur):
                break
            cur = best
            path.append({"layer": l, "node": cur, "dist": round(dist(cur), 4)})
    # tầng 0: tìm theo chùm (beam) với ef ứng viên
    visited, cand, result = {cur}, [cur], [cur]
    base_path = list(path)
    order = []
    while cand:
        c = min(cand, key=dist)
        cand.remove(c)
        worst = sorted(result, key=dist)[:ef][-1] if len(result) >= ef else None
        if worst is not None and dist(c) > dist(worst):
            break
        order.append(c)
        for nb in nbrs[0][c]:
            if nb not in visited:
                visited.add(nb); cand.append(nb); result.append(nb)
        result = sorted(set(result), key=dist)[:ef]
    found = sorted(result, key=dist)[:top_k]
    exact = [int(i) for i in np.argsort(np.linalg.norm(pts - q, axis=1))[:top_k]]
    return {
        "points": pts.round(4).tolist(), "level": level.tolist(), "layers": layers, "query": q.round(4).tolist(),
        "descent": base_path, "layer0_order": order, "visited": sorted(visited), "found": [int(i) for i in found], "exact": exact,
        "recall": round(len(set(found) & set(exact)) / top_k, 2), "ef": ef, "top_k": top_k, "entry": 0, "top_layer": top,
    }


# ---------------------------------------------------------------- 4. lỗi thật của pgvector
def _run(conn, sql: str, shown: str | None = None) -> dict:
    try:
        conn.execute(sql)
        return {"sql": shown or sql, "ok": True, "message": None}
    except psycopg.Error as exc:
        return {"sql": shown or sql, "ok": False, "message": str(exc).strip().splitlines()[0], "kind": type(exc).__name__}


def errors(case: str) -> dict:
    conn = connect()
    steps = []
    long_vec = lambda dim: "'[" + ",".join(["0.1"] * dim) + "]'"
    try:
        if case == "dim_mismatch":
            steps += [_run(conn, "CREATE TEMP TABLE lesson_err (id int, embedding vector(384))"),
                      _run(conn, f"INSERT INTO lesson_err VALUES (1, {long_vec(384)}::vector)", "INSERT INTO lesson_err VALUES (1, '[0.1, 0.1, …]'::vector);  -- 384 chiều (đúng)"),
                      _run(conn, f"INSERT INTO lesson_err VALUES (2, {long_vec(768)}::vector)", "INSERT INTO lesson_err VALUES (2, '[0.1, 0.1, …]'::vector);  -- 768 chiều (vector của model khác)")]
        elif case == "query_dim":
            steps += [_run(conn, "CREATE TEMP TABLE lesson_err (id int, embedding vector(384))"),
                      _run(conn, f"INSERT INTO lesson_err VALUES (1, {long_vec(384)}::vector)", "INSERT INTO lesson_err VALUES (1, '[0.1, …]'::vector);  -- 384 chiều"),
                      _run(conn, "SELECT id FROM lesson_err ORDER BY embedding <=> '[1,2,3]' LIMIT 5")]
        elif case == "hnsw_2048":
            steps += [_run(conn, "CREATE TEMP TABLE lesson_err (id int, embedding vector(2048))"),
                      _run(conn, "CREATE INDEX ON lesson_err USING hnsw (embedding vector_cosine_ops) WITH (m = 16, ef_construction = 64)"),
                      _run(conn, "DROP TABLE lesson_err"),
                      _run(conn, "CREATE TEMP TABLE lesson_err (id int, embedding halfvec(2048))", "-- Cách sửa 1: dùng halfvec (độ chính xác 16-bit, hỗ trợ tới 4000 chiều có index)\nCREATE TABLE ... (id int, embedding halfvec(2048));"),
                      _run(conn, "CREATE INDEX ON lesson_err USING hnsw (embedding halfvec_cosine_ops) WITH (m = 16, ef_construction = 64)",
                           "CREATE INDEX ON ... USING hnsw (embedding halfvec_cosine_ops) WITH (m = 16, ef_construction = 64);")]
        else:
            raise ValueError(f"tình huống lỗi không hỗ trợ: {case}")
    finally:
        try:
            conn.execute("DROP TABLE IF EXISTS lesson_err")
        finally:
            conn.close()
    return {"case": case, "steps": steps}
