"""Đường dẫn, tên model và kết nối DB dùng chung cho mọi demo."""
import os
import sys
from pathlib import Path

# Giảm nhiễu log của HuggingFace/transformers khi demo trên lớp (đặt TRƯỚC khi import chúng).
os.environ.setdefault("HF_HUB_VERBOSITY", "error")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# Windows console mặc định cp1252 -> tiếng Việt bị lỗi. Ép UTF-8 cho mọi demo.
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "output"  # HTML/biểu đồ do demo sinh ra
CACHE_DIR = ROOT / ".cache"  # embedding đã tính, tránh chạy lại model

# multilingual-e5-small: 384 chiều, tiếng Việt + Anh, chạy tốt trên CPU.
# Lưu ý: e5 yêu cầu prefix "query: " / "passage: " (xem lib/models.py).
EMBED_MODEL = "intfloat/multilingual-e5-small"
EMBED_DIM = 384
# Cross-encoder đa ngôn ngữ cho bước reranking.
RERANK_MODEL = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"

DATABASE_URL = os.getenv(
    "DATABASE_URL", "postgresql://demo:demo@localhost:5433/semantic_search"
)

OUTPUT_DIR.mkdir(exist_ok=True)
CACHE_DIR.mkdir(exist_ok=True)
