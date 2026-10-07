# Image chạy UI + Search API. Model được tải sẵn lúc build nên container start là chạy ngay, không cần mạng.
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1
WORKDIR /app

# PyTorch bản CPU (~200 MB) thay vì bản CUDA mặc định (~2.5 GB+); máy demo không có GPU.
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN pip install -r requirements.txt

# Tải 2 model vào image (layer riêng, ít thay đổi nên được cache khi sửa code).
COPY lib/__init__.py lib/config.py lib/
RUN python -c "from sentence_transformers import SentenceTransformer, CrossEncoder; \
from lib.config import EMBED_MODEL, RERANK_MODEL; \
SentenceTransformer(EMBED_MODEL); CrossEncoder(RERANK_MODEL); print('models cached')"

COPY lib lib
COPY api api
COPY ui ui
COPY data data
COPY demos demos

EXPOSE 8000
HEALTHCHECK --interval=10s --timeout=5s --start-period=60s --retries=12 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4)"
# 0.0.0.0 để cổng được publish ra máy host.
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
