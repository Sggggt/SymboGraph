from __future__ import annotations

import time
from pathlib import Path

from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer

from worker_app.bootstrap import API_ROOT  # noqa: F401
from worker_app.tasks import ingest_path
from app.core.config import get_settings
from app.services.ingestion import should_include_file
from app.services.storage import (
    ensure_storage_durability_ready,
    validate_source_content_path,
)


class KnowledgeBaseEventHandler(FileSystemEventHandler):
    def __init__(self) -> None:
        super().__init__()
        self.settings = get_settings()
        self.cache: dict[str, tuple[float, str]] = {}

    def on_created(self, event) -> None:
        if not event.is_directory:
            self._handle(Path(event.src_path))

    def on_modified(self, event) -> None:
        if not event.is_directory:
            self._handle(Path(event.src_path))

    def _handle(self, path: Path) -> None:
        if not should_include_file(
            path,
            authorized_root=self.settings.storage_root_path,
        ) or not path.exists():
            return
        stat = path.stat()
        validation = validate_source_content_path(
            path,
            self.settings.storage_root_path,
            max_bytes=self.settings.upload_max_bytes,
        )
        checksum = validation.checksum
        cached = self.cache.get(str(path))
        snapshot = (stat.st_mtime, checksum)
        if cached == snapshot:
            return
        self.cache[str(path)] = snapshot
        ingest_path.delay(str(path), trigger_source="watchdog")


def main() -> None:
    settings = get_settings()
    ensure_storage_durability_ready(settings=settings, force_probe=True)
    storage_root = settings.storage_root_path
    observer = Observer()
    handler = KnowledgeBaseEventHandler()
    observer.schedule(handler, str(storage_root), recursive=True)
    observer.start()
    print(f"Watching {storage_root}")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()


if __name__ == "__main__":
    main()
