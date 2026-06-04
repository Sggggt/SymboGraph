# SymboGraph Background Worker

An asynchronous task processor and filesystem watcher designed to offload long-running, CPU-intensive computations from the main FastAPI application thread and synchronize local source documents.

---

## 🛠️ Key Roles & Components

### 1. Asynchronous Ingestion (Celery Worker)
To prevent blocking the core API event loop and maintain low latency for interactive client RAG requests, heavy computing loads are offloaded to **Celery**:
- **Bulk Document Parsing**: Processing extensive PDF layouts, Jupyter notebook structures, and Office slides.
- **High-volume Embedding Generation**: Performing batch vector embeddings across thousands of semantic text chunks.
- **LLM Graph Extraction**: Managing long-running LLM extract-and-merge pipelines.
- **Auto HPO Trials**: Running sequential Optuna TPE simulations over candidate parameters.
- **Phase-aware cancellation**: Preserving committed parse results when cancellation happens during graph work, while rolling back parse-phase writes when cancellation happens before parse commit.

### 2. Filesystem Synchronization (Course Watcher)
A lightweight monitoring service using `watchdog` to track file events within your course directories (`DATA_ROOT`):
- Automatically triggers a sync ingestion batch when course files (`.pdf`, `.ipynb`, `.md`, etc.) are added, updated, or removed from local storage folders.
- Uses file hashes (checksums) to dynamically deduplicate file writes and prevent redundant pipeline triggers.

---

## 📂 Project Structure

```
apps/worker/
├── worker_app/
│   ├── celery_app.py   # Celery application registration and task broker configuration
│   ├── tasks.py        # Shared background tasks (ingestion, graph rebuild)
│   └── watcher.py       # Event-driven filesystem watchdog monitoring course storage
├── pyproject.toml      # Modern uv package manager configuration
└── README.md           # Worker documentation
```

---

## 🚀 Running the Worker

> [!NOTE]
> The Celery worker is part of the standard `infra/docker-compose.yml` stack and consumes the `course-kg-main-ingestion` queue. The filesystem watcher remains optional and is not started by the default compose command.

Ensure that PostgreSQL, Qdrant, and Redis are running and reachable before launching the worker services.

### Starting the Celery Background Task Worker
To run the background task handler:
```bash
# Sync local virtualenv using uv
uv sync

# Launch Celery worker
uv run celery -A worker_app.celery_app worker --loglevel=info --queues=course-kg-main-ingestion --concurrency=${WORKER_CONCURRENCY:-2}
```

### Starting the Filesystem Directory Watcher
To run the filesystem watchdog listener:
```bash
# Launch watcher module
uv run python -m worker_app.watcher
```
