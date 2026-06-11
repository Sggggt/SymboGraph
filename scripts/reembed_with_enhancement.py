#!/usr/bin/env python3
"""Compatibility entrypoint for active chunk contextual re-embedding.

Run inside the API container:
    python /app/scripts/reembed_with_enhancement.py --knowledge-base-name "Knowledge Base" --dry-run

This delegates to reembed_all_chunks.py, which now operates only on active_chunks
and vector_records.
"""
from __future__ import annotations

import asyncio

from reembed_all_chunks import main


if __name__ == "__main__":
    asyncio.run(main())
