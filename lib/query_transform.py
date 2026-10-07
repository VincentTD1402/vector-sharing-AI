"""Query transformation bằng từ điển viết tay: rewrite, expansion, multi-query (slide 27).

Dùng từ điển để chạy offline và minh bạch. Thực tế dùng LLM để viết lại - cùng cấu trúc,
chỉ thay hàm `rewrite/expand`. HyDE (sinh tài liệu giả định rồi embed) cần LLM nên không có ở đây.
"""
ABBREVIATIONS = {"pw": "password", "inv": "invoice", "pr": "purchase request", "pto": "annual leave",
                 "sx": "sản xuất", "2fa": "two-factor authentication", "2": "to", "chk": "check"}
SYNONYMS = {
    "password": ["credentials", "login", "mật khẩu"], "reset": ["recover", "đặt lại"],
    "invoice": ["bill", "hóa đơn"], "approve": ["authorize", "duyệt"], "purchase": ["mua hàng"],
    "leave": ["vacation", "nghỉ phép"], "stock": ["inventory", "tồn kho"], "request": ["yêu cầu"],
    "sản": ["production"], "xuất": ["production order"], "lệnh": ["order"], "check": ["view", "kiểm tra"],
}
# Query ngắn/viết tắt + doc đúng (corpus company_a). Từ điển ở trên được viết SAU khi biết các query này,
# nên số đo trên chúng lạc quan: chỉ minh họa cơ chế.
TERSE_QUERIES = [
    {"query": "reset pw", "relevant": ["d01", "d03"]}, {"query": "inv approve", "relevant": ["d14", "d15"]},
    {"query": "create pr", "relevant": ["d24"]}, {"query": "pto request", "relevant": ["d34", "d35"]},
    {"query": "lệnh sx mới", "relevant": ["d08", "d07"]}, {"query": "2fa setup", "relevant": ["d05"]},
    {"query": "stock chk", "relevant": ["d27"]}, {"query": "how 2 get vpn", "relevant": ["d45"]},
]
MODES = {
    "none": "Không biến đổi", "rewrite": "Rewrite (mở viết tắt)",
    "expand": "Rewrite + Expansion (thêm từ đồng nghĩa)", "multi": "Multi-query (gốc + rewrite + expansion)",
}


def rewrite(query: str) -> str:
    return " ".join(ABBREVIATIONS.get(t, t) for t in query.lower().split())


def expand(query: str) -> str:
    base = rewrite(query)
    extra = [s for t in base.split() for s in SYNONYMS.get(t, [])]
    return f"{base} {' '.join(extra)}".strip()


def variants(query: str, mode: str) -> list[str]:
    """Danh sách câu truy vấn dùng để retrieve theo chế độ biến đổi."""
    if mode == "none":
        return [query]
    if mode == "rewrite":
        return [rewrite(query)]
    if mode == "expand":
        return [expand(query)]
    if mode == "multi":
        return list(dict.fromkeys([query, rewrite(query), expand(query)]))  # bỏ trùng, giữ thứ tự
    raise ValueError(f"chế độ không hỗ trợ: {mode}")


def explain(query: str) -> dict:
    """Luật nào đã được áp dụng cho query này (để UI giải thích vì sao câu bị đổi)."""
    base = rewrite(query)
    return {
        "abbreviations": {t: ABBREVIATIONS[t] for t in query.lower().split() if t in ABBREVIATIONS},
        "synonyms": {t: SYNONYMS[t] for t in base.split() if t in SYNONYMS},
    }
