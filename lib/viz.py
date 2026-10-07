"""Xuất biểu đồ Plotly ra HTML tĩnh (mở bằng trình duyệt, không cần server)."""
import sys
import webbrowser

from lib.config import OUTPUT_DIR


def save_figure(fig, filename: str) -> str:
    """Lưu figure vào output/. plotly.min.js dùng chung 1 bản -> chạy offline được."""
    path = OUTPUT_DIR / filename
    fig.write_html(path, include_plotlyjs="directory")
    print(f"\n[HTML] {path}")
    if "--open" in sys.argv:
        webbrowser.open(path.as_uri())
    return str(path)


def banner(title: str, slide: str = "") -> None:
    line = "=" * 78
    suffix = f"   (slide {slide})" if slide else ""
    print(f"\n{line}\n{title}{suffix}\n{line}")
