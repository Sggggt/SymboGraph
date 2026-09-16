"""Coalesce redundant, argument-free periodic scans; preserve all user jobs."""
from __future__ import annotations

import argparse
import base64
import json

from _context_graph_maintenance import API_ROOT

ALLOWLIST = frozenset({"reconcile_interrupted_ingestion_batches", "reconcile_profile_lifecycle", "reconcile_vector_store", "update_policy_backfill"})


def plan_messages(newest_first):
    retained, duplicates, invalid = set(), [], 0
    for raw in newest_first:
        try:
            message = json.loads(raw)
            headers = message.get("headers", {})
            task = headers.get("task")
            if task not in ALLOWLIST:
                continue
            body = json.loads(base64.b64decode(message["body"], validate=True))
            if not isinstance(body,list) or len(body) != 3 or not isinstance(body[2],(dict,type(None))):
                invalid += 1
                continue
            if body[0] != [] or body[1] != {} or any((body[2] or {}).values()):
                continue
            if task in retained:
                duplicates.append({"task":task,"id":headers["id"],"raw":raw})
            else:
                retained.add(task)
        except (ValueError, TypeError, KeyError, AttributeError):
            invalid += 1
    return duplicates, invalid


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    import redis
    from app.core.config import get_settings
    settings = get_settings()
    client = redis.Redis.from_url(settings.redis_url)
    queue = settings.ingestion_task_queue
    messages = client.lrange(queue,0,999)
    duplicates, invalid = plan_messages(messages)
    targets = [{"task":item["task"],"id":item["id"]} for item in duplicates]
    print(json.dumps({"execute":args.execute,"scanned":len(messages),"invalid":invalid,"targets":targets,
                      "impact":"retain newest argument-free scan per known periodic task; user jobs and parameterized tasks are preserved"}),flush=True)
    if invalid:
        raise RuntimeError("Queue contains malformed maintenance messages; no changes applied")
    if args.execute:
        removed = 0
        for start in range(0,len(duplicates),100):
            with client.pipeline(transaction=True) as pipeline:
                for item in duplicates[start:start+100]:
                    pipeline.lrem(queue,1,item["raw"])
                removed += sum(pipeline.execute())
        print(json.dumps({"removed":removed,"remaining":client.llen(queue)}),flush=True)


if __name__ == "__main__":
    main()
