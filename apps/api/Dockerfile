FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/app/models/huggingface \
    TRANSFORMERS_OFFLINE=0

RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates curl tesseract-ocr \
    && rm -rf /var/lib/apt/lists/*

# 安装 uv 用于快速依赖解析和安装
RUN pip install --no-cache-dir uv

WORKDIR /app/apps/api

# 复制项目元数据
COPY apps/api/pyproject.toml ./

# 使用 uv pip install --system 直接安装到系统 Python，不创建虚拟环境
# 安装 [rerank] 和 dev extra 到系统 Python：支持 Cross-Encoder 与容器内 pytest 验收
RUN uv pip install --system --no-cache -e ".[rerank,dev,ocr]"

# 复制应用代码
COPY apps/api/app ./app
COPY apps/api/tests ./tests
COPY apps/worker/worker_app /app/apps/worker/worker_app
COPY .env.example /app/.env.example

# 默认不在构建阶段访问 Hugging Face，避免常规 rebuild 卡在网络下载。
# 如需把 reranker 模型预热进镜像，构建时传入：
#   docker build --build-arg PRELOAD_RERANK_MODEL=true ...
ARG PRELOAD_RERANK_MODEL=false
RUN if [ "$PRELOAD_RERANK_MODEL" = "true" ]; then \
      python -c "import os; os.environ['TRANSFORMERS_OFFLINE']='0'; from sentence_transformers import CrossEncoder; CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2', max_length=512); print('Cross-encoder model pre-downloaded')"; \
    else \
      echo "Model pre-download skipped; set PRELOAD_RERANK_MODEL=true to bake it into the image"; \
    fi

EXPOSE 8000

HEALTHCHECK --interval=15s --timeout=5s --retries=20 \
    CMD curl -f http://127.0.0.1:8000/api/health || exit 1

CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
