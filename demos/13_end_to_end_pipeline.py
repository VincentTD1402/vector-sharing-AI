"""Demo 13 - Pipeline retrieval hoàn chỉnh trên pgvector + monitoring + caching (slide 28-32).

Document ingestion -> query -> hybrid (BM25 + pgvector, pre-filter) -> RRF -> rerank -> top-k.
A. Một request chạy hết pipeline, kèm trace từng bước. Cùng query, khác tenant -> kết quả khác.
B. Latency budget trung bình theo từng stage (slide 32): chỗ nào đáng tối ưu?
C. Log có cấu trúc cho mỗi request (slide 31).
D. Caching: embedding cache + semantic cache (slide 32).

Cần: docker compose up -d
Chạy:  python demos/13_end_to_end_pipeline.py
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from lib.cache import SemanticCache
from lib.data import load_eval_queries
from lib.pipeline import RetrievalService
from lib.viz import banner

STAGES = ["embed", "dense_search", "lexical_search", "fusion", "rerank", "total"]


def one_request(svc: RetrievalService) -> None:
    banner("A. Một request đi qua toàn bộ pipeline", "28")
    query = "How do I approve an invoice?"
    for tenant in ("company_a", "company_b"):
        results, trace = svc.search(query, tenant, top_k=3)
        print(f"\n  Query: {query!r}  tenant={tenant}")
        for r in results:
            print(f"    {r['id']} [{r['tenant_id']}] rerank={r['score']:6.2f}  {r['content'][:62]}")
        print(f"    dense top: {trace['dense_ids'][:4]}  lexical top: {trace['lexical_ids'][:4]}")
    print("\n  Cùng query, mỗi tenant chỉ thấy tài liệu của mình: filter nằm trong SQL + mask BM25, không phụ thuộc similarity.")


def latency_budget(svc: RetrievalService) -> None:
    banner("B. Latency budget trung bình (ms) trên bộ eval", "32")
    queries = [q["query"] for q in load_eval_queries()]
    svc.search(queries[0], "company_a")  # warm-up
    traces = [svc.search(q, "company_a")[1] for q in queries]
    print(f"  {len(queries)} query, tenant company_a, pool={svc.pool}, top_k=5 (CPU, không GPU)")
    avg = {s: np.mean([t["timings_ms"][s] for t in traces]) for s in STAGES}
    p95 = np.percentile([t["timings_ms"]["total"] for t in traces], 95)
    for s in STAGES[:-1]:
        bar = "#" * int(avg[s] / avg["total"] * 40)
        print(f"    {s:<15} {avg[s]:>7.1f} ms  {bar}")
    print(f"    {'TỔNG (trung bình)':<15} {avg['total']:>7.1f} ms   p95 = {p95:.0f} ms")
    print("  -> Rerank (cross-encoder) thường chiếm phần lớn: giảm pool, rerank model nhỏ hơn, hoặc batch/GPU.")
    print("  Không đo từng stage thì không biết tối ưu chỗ nào (slide 32: 'không thể tối ưu những gì không đo được').")


def structured_log(svc: RetrievalService) -> None:
    banner("C. Log có cấu trúc mỗi request - dữ liệu cần để debug production", "31")
    _, trace = svc.search("quên mật khẩu", "company_a", top_k=3)
    record = {"request_id": "demo-0001", **trace}
    for key in ("dense_ids", "lexical_ids"):  # rút gọn khi in; log thật giữ đủ danh sách
        record[key] = record[key][:5] + (["..."] if len(record[key]) > 5 else [])
    print(json.dumps(record, ensure_ascii=False))
    print("  Dùng log này để: tính zero-result rate, phát hiện score tụt (embedding drift), debug filter quá chặt.")


def caching(svc: RetrievalService) -> None:
    banner("D. Caching: query lặp lại và query tương tự", "32")
    tenant = "company_a"
    workload = ["how do I reset my password", "how do I reset my password", "How do I reset my password?",
                "how can I reset my password", "create production order", "create production order"]
    cache = SemanticCache(svc, threshold=0.95)
    svc._embed_cache.clear()
    print(f"  {'query':<34} {'kết quả':<24} {'thời gian':>10}")
    for q in workload:
        start = time.perf_counter()
        cached, _, _ = cache.lookup(q, tenant)
        if cached is not None:
            outcome = "semantic cache HIT"
        else:
            results, trace = svc.search(q, tenant, top_k=3)
            cache.put(q, tenant, results)
            # Lần tra semantic cache đã embed query; pipeline tái dùng vector đó nên không embed lần 2.
            outcome = "MISS -> chạy pipeline" + (" (dùng lại embedding)" if trace["embed_cache_hit"] else "")
        print(f"  {q:<34} {outcome:<24} {(time.perf_counter() - start) * 1000:>7.1f} ms")
    print("\n  Ngưỡng 0.95 với e5 (điểm nén vào dải ~0.7-1.0) phải tự hiệu chỉnh: quá thấp -> trả lời sai cho query khác ý;")
    print("  quá cao -> không bao giờ hit. Cache theo tenant (tránh rò rỉ dữ liệu) và đặt TTL ngắn khi dữ liệu thay đổi.")
    print("  Khóa cache ở demo chỉ gồm (vector query, tenant): production phải thêm top_k, language, department.")


if __name__ == "__main__":
    service = RetrievalService()
    print(f"Đã index {service.chunk_count} chunk vào pgvector (upsert, chạy lại không trùng).")
    one_request(service)
    latency_budget(service)
    structured_log(service)
    caching(service)
