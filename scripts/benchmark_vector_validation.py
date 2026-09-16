"""Bounded synthetic benchmark of byte-identical vector validation kernels."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import random
import struct
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


def previous_bulk(vector):
    """Frozen previous native-list fast path, for valid synthetic input only."""
    from app.services.vector_store import MIN_EMBEDDING_VECTOR_NORM
    assert all(type(value) in (float, int) for value in vector)
    values = struct.unpack(f'>{len(vector)}f', struct.pack(f'>{len(vector)}f', *vector))
    assert all(math.isfinite(value) for value in values)
    result = [0.0 if value == 0.0 else value for value in values]
    assert math.sqrt(math.fsum(value * value for value in result)) > MIN_EMBEDDING_VECTOR_NORM
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'output/vector-validation-benchmark.json')
    parser.add_argument('--vectors', type=int, default=128, choices=range(1, 513))
    parser.add_argument('--repeats', type=int, default=3, choices=range(1, 6))
    args = parser.parse_args()
    if not args.output.resolve().is_relative_to((ROOT / 'output').resolve()) or args.output.exists():
        parser.error('Choose a new file in output/.')
    sys.path.insert(0, str(ROOT / 'apps/api'))
    from app.services.vector_store import canonical_embedding_vector, _canonical_embedding_vector_scalar
    rng, rows = random.Random(531), []
    for dimension in (64, 512, 1024, 2048):
        vectors = [[rng.uniform(-1, 1) for _ in range(dimension)] for _ in range(args.vectors)]
        expected = [_canonical_embedding_vector_scalar(vector, source='synthetic-reference') for vector in vectors]
        for name, operation in (('previous_bulk', previous_bulk),
                                ('native_array', lambda v: canonical_embedding_vector(v, source='synthetic'))):
            actual = [operation(vector) for vector in vectors]  # one warmup + equality check
            assert actual == expected
            latencies, cpus = [], []
            for _ in range(args.repeats):
                wall, cpu = time.perf_counter(), time.process_time()
                for vector in vectors:
                    operation(vector)
                latencies.append(time.perf_counter() - wall)
                cpus.append(time.process_time() - cpu)
            rows.append({'mode':name, 'dimension':dimension, 'batch_seconds':latencies,
                'batch_cpu_seconds':cpus, 'mean_microseconds_per_vector':sum(latencies)/len(latencies)/len(vectors)*1e6})
    result = {'synthetic':True, 'model_calls':0, 'database_writes':0, 'scalar_reference_equal':True,
        'warmup_batches':1, 'measured_batches':args.repeats, 'vectors_per_batch':args.vectors,
        'full_qa_latency_claimed':False, 'results':rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
