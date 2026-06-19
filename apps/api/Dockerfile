FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates curl tesseract-ocr \
    && rm -rf /var/lib/apt/lists/*

# 安装 uv 用于快速依赖解析和安装
RUN pip install --no-cache-dir uv

WORKDIR /app/apps/api

# 复制项目元数据
COPY apps/api/pyproject.toml ./

# 使用 uv pip install --system 直接安装到系统 Python，不创建虚拟环境
# 安装 dev/ocr extra 到系统 Python：支持容器内 pytest 验收和 OCR 解析
RUN uv pip install --system --no-cache -e ".[dev,ocr]"

# 复制应用代码
COPY apps/api/app ./app
COPY apps/api/alembic.ini ./alembic.ini
COPY apps/api/migrations ./migrations
COPY apps/api/tests ./tests
COPY apps/worker/worker_app /app/apps/worker/worker_app
COPY .env.example /app/.env.example

EXPOSE 8000

HEALTHCHECK --interval=15s --timeout=5s --retries=20 \
    CMD curl -f http://127.0.0.1:8000/api/health || exit 1

CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
