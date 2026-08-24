FROM python:3.13-slim@sha256:a0779d7c12fc20be6ec6b4ddc901a4fd7657b8a6bc9def9d3fde89ed5efe0a3d

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    VIRTUAL_ENV=/app/apps/api/.venv \
    PATH="/app/apps/api/.venv/bin:${PATH}"

RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    set -eu; \
    apt-get -o Acquire::Retries=5 update; \
    for attempt in 1 2 3; do \
        if apt-get -o Acquire::Retries=5 install -y --no-install-recommends \
            ca-certificates=20250419 \
            curl=8.14.1-2+deb13u4 \
            tesseract-ocr=5.5.0-1+b1; then \
            break; \
        fi; \
        if [ "$attempt" = 3 ]; then \
            exit 1; \
        fi; \
        apt-get -o Acquire::Retries=5 update; \
    done; \
    rm -rf /var/lib/apt/lists/*

# 安装 uv 用于快速依赖解析和安装
RUN pip install --no-cache-dir uv==0.12.3

WORKDIR /app/apps/api

# 复制项目元数据
COPY apps/api/pyproject.toml ./
COPY apps/api/uv.lock ./uv.lock

# 从 uv.lock 同步精确的 Python 3.13/Linux 依赖，并把容器运行入口固定到该虚拟环境。
# 项目本身是非 package 源码，后续 COPY；dev/ocr 用于容器内验收和 OCR 解析。
RUN --mount=type=cache,target=/root/.cache/uv,sharing=locked \
    uv sync --locked --no-install-project --extra dev --extra ocr \
    && uv sync --locked --check --no-install-project --extra dev --extra ocr

# 复制应用代码
COPY apps/api/app ./app
COPY apps/api/Dockerfile ./Dockerfile
COPY apps/api/alembic.ini ./alembic.ini
COPY apps/api/migrations ./migrations
COPY apps/api/tests ./tests
COPY apps/web/src /app/apps/web/src
COPY packages/shared/src /app/packages/shared/src
COPY apps/worker/worker_app /app/apps/worker/worker_app
COPY apps/worker/pyproject.toml /app/apps/worker/pyproject.toml
COPY apps/worker/uv.lock /app/apps/worker/uv.lock
COPY scripts /app/scripts
COPY infra/model-bridge /app/infra/model-bridge
COPY infra/docker-compose.yml /app/infra/docker-compose.yml
COPY infra/README.md /app/infra/README.md
COPY start-app.ps1 /app/start-app.ps1
COPY start-app.bat /app/start-app.bat
COPY rebuild-images.ps1 /app/rebuild-images.ps1
COPY rebuild-images.bat /app/rebuild-images.bat
COPY package.json /app/package.json
COPY package-lock.json /app/package-lock.json
COPY docs/technical-spec.md /app/docs/technical-spec.md

EXPOSE 8000

HEALTHCHECK --interval=15s --timeout=5s --retries=20 \
    CMD curl -f http://127.0.0.1:8000/api/ready || exit 1

CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
