"""Read Docker resource samples for the named SymboGraph stack only."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
import subprocess
import threading
import queue
import re
import time
from pathlib import Path

SERVICES = [f"course-kg-{name}" for name in ("api", "worker", "beat", "web", "model-bridge", "postgres", "redis", "qdrant")]


def parse_stats_line(line):
    # Docker on Windows emits ANSI clear-line codes even into a pipe.
    cleaned = re.sub(r"\x1b\[[0-?]*[ -/]*[@-~]", "", line).strip()
    return json.loads(cleaned) if cleaned else None


def memory_bytes(value):
    import re
    match = re.fullmatch(r"\s*([\d.]+)\s*([kKMGT]?i?B)\s*", value)
    if not match:
        raise ValueError("Unrecognized Docker memory unit")
    units = {"B":1,"kB":1000,"MB":1000**2,"GB":1000**3,"TB":1000**4,
             "KiB":1024,"MiB":1024**2,"GiB":1024**3,"TiB":1024**4}
    return int(float(match[1])*units[match[2]])


def resource_row(item, available):
    if item["Name"] not in SERVICES:
        raise ValueError("Unexpected Docker service in resource stream")
    cpu = float(item["CPUPerc"].rstrip("%"))
    if not math.isfinite(cpu) or cpu < 0:
        raise ValueError("Invalid Docker CPU observation")
    return {"service":item["Name"], "cpu_cores":cpu/100,
            "cpu_percent_of_available":cpu/available,
            "memory_bytes":memory_bytes(item["MemUsage"].split("/")[0]),
            "block_io":item["BlockIO"], "network_io":item["NetIO"],
            "pids":int(item["PIDs"])}


def distribution(values):
    ordered = sorted(values)
    return {"sample_count":len(ordered), "method":"nearest_rank", **{
        name:ordered[max(0,math.ceil(len(ordered)*q)-1)] if ordered else None
        for name,q in (("p50",.5),("p95",.95),("p99",.99))}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--duration-seconds", type=int, default=2100)
    parser.add_argument("--output", default="output/build-resources.json")
    parser.add_argument("--stop-file", default="output/stop-build-resources")
    args = parser.parse_args()
    root = (Path(__file__).resolve().parents[1]/"output").resolve()
    output, stop = Path(args.output).resolve(), Path(args.stop_file).resolve()
    if not output.is_relative_to(root) or not stop.is_relative_to(root):
        raise ValueError("Resource artifacts must remain inside repository output")
    if not args.execute:
        print(json.dumps({"services":SERVICES,"duration_seconds":args.duration_seconds,"writes":False}))
        return
    if stop.exists():
        raise ValueError("Choose an absent stop-file for the new observation")
    if not 1 <= args.duration_seconds <= 43_200:
        raise ValueError("Resource observation duration must be between 1 and 43200 seconds")
    sample_path = output.with_suffix(".jsonl")
    if output.exists() or sample_path.exists():
        raise ValueError("Choose unused resource artifact paths for a new observation")
    root.mkdir(exist_ok=True)
    available = int(subprocess.check_output(["docker","info","--format","{{.NCPU}}"],text=True).strip())
    started = time.monotonic()
    started_utc = datetime.now(timezone.utc).isoformat()
    samples, peak_by_service, errors = [], {}, 0
    error_samples = []
    process = subprocess.Popen(["docker","stats","--all","--format","{{json .}}",*SERVICES],
                               stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
    messages = queue.Queue(maxsize=128)
    def read_stream():
        for line in process.stdout:
            try:
                messages.put(line, timeout=2)
            except queue.Full:
                return
    reader = threading.Thread(target=read_stream, daemon=True)
    reader.start()
    frame = {}
    last_persist = 0.0
    sample_stream = sample_path.open("x", encoding="utf-8")

    def persist():
        if not samples:
            return
        elapsed = [sample["elapsed_seconds"] for sample in samples]
        report = {"sample_count":len(samples), "available_cpu_count":available,
                  "peak_stack_memory_bytes":max(s["memory_bytes"] for s in samples),
                  "peak_service_memory_bytes":peak_by_service, "errors":errors,
                  "error_samples":error_samples, "sample_file":sample_path.name,
                  "started_at_utc":started_utc,
                  "first_sample_utc":samples[0]["observed_at_utc"],
                  "last_sample_utc":samples[-1]["observed_at_utc"],
                  "largest_sample_gap_seconds":max((b-a for a,b in zip(elapsed,elapsed[1:])),default=0),
                  "stack_cpu_cores":distribution([s["cpu_cores"] for s in samples]),
                  "stack_cpu_percent_of_available":distribution([100*s["cpu_cores"]/available for s in samples])}
        temporary = output.with_suffix(".tmp")
        temporary.write_text(json.dumps(report,ensure_ascii=False),encoding="utf-8")
        temporary.replace(output)

    try:
        while time.monotonic()-started < args.duration_seconds and not stop.exists():
            try:
                line = messages.get(timeout=1)
            except queue.Empty:
                if process.poll() is not None:
                    errors += 1
                    raise RuntimeError("Docker resource stream ended before observation completed")
                continue
            try:
                item = parse_stats_line(line)
                if item is None:
                    continue
                row = resource_row(item, available)
            except (ValueError, KeyError) as exc:
                errors += 1
                if len(error_samples) < 5:
                    error_samples.append({"type":type(exc).__name__,"line":repr(line)[:600]})
                continue
            peak_by_service[row["service"]] = max(peak_by_service.get(row["service"],0),row["memory_bytes"])
            frame[row["service"]] = row
            if set(frame) != set(SERVICES):
                continue
            rows = [frame[name] for name in SERVICES]
            frame.clear()
            sample = {"elapsed_seconds":time.monotonic()-started,
                      "observed_at_utc":datetime.now(timezone.utc).isoformat(),
                      "memory_bytes":sum(r["memory_bytes"] for r in rows),
                      "cpu_cores":sum(r["cpu_cores"] for r in rows),"services":rows}
            # Append each observation once. Rewriting the complete history on
            # every frame makes the observer's own I/O quadratic in duration.
            sample_stream.write(json.dumps(sample,ensure_ascii=False)+"\n")
            sample_stream.flush()
            samples.append(sample)
            if not last_persist or time.monotonic()-last_persist >= 5:
                persist()
                last_persist = time.monotonic()
    finally:
        sample_stream.close()
        persist()
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)
        print(json.dumps({"output":str(output),"sample_count":len(samples),"errors":errors}),flush=True)


if __name__ == "__main__":
    main()
