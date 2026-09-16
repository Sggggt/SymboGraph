"""Synthetic exact-kernel scale study; no graph/database writes or model calls."""
from __future__ import annotations

import argparse
import json
import math
import time
from types import SimpleNamespace

from _context_graph_maintenance import write_report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--sizes", default="256,512,1024,2048,4096")
    args = parser.parse_args()
    sizes = [int(value) for value in args.sizes.split(",")]
    if any(value < 1 or value > 4096 for value in sizes) or not 1 <= args.repetitions <= 10 or not 0 <= args.warmups <= 3:
        raise ValueError("Kernel benchmark scope exceeds bounded allowance")
    if not args.execute:
        print(json.dumps({"sizes":sizes,"dimensions":[128,1024],"repetitions":args.repetitions,"warmups":args.warmups,"writes":False}))
        return
    import numpy as np
    from app.services.graph_build_workspace import GraphBuildWorkspace
    from app.services.context_graph import train_rq_kmeans,encode_rq_vectors_batch,cosine_similarity
    from _graph_scalar_reference import train_codebooks, primary_path
    results = []
    for d in (128,1024):
        for n in sizes:
            chunks = [SimpleNamespace(id=f"unit-test-{i:05d}") for i in range(n)]
            matrix = np.random.default_rng(20260909).normal(size=(n,d))
            vectors = {chunk.id:row.tolist() for chunk,row in zip(chunks,matrix)}
            reference_vectors = list(vectors.values())[:min(n,64)]
            reference_runs = []
            for repetition in range(-args.warmups,args.repetitions):
                with GraphBuildWorkspace() as workspace:
                    start = time.perf_counter()
                    workspace.bind(chunks,vectors)
                    similarity_seconds = time.perf_counter()-start
                    for row in range(min(n,8)):
                        for column in range(min(n,8)):
                            if abs(float(workspace.scores[row,column])-cosine_similarity(vectors[chunks[row].id],vectors[chunks[column].id])) > 1e-12:
                                raise RuntimeError("Exact-kernel scalar tolerance failed")
                    start = time.perf_counter()
                    model = train_rq_kmeans(list(vectors.values()),levels=3,max_k=6,tau_r=.65,tau_l=.35)
                    encoded,_ = encode_rq_vectors_batch(list(vectors.items()),model)
                    rq_seconds = time.perf_counter()-start
                    assert all(len(item["prefix_memberships"])==3 for item in encoded.values())
                    result = {"n":n,"d":d,"repetition":repetition,"similarity_seconds":similarity_seconds,"rq_core_seconds":rq_seconds,
                              "matrix_bytes":workspace.scores.nbytes,"mapped_bytes":workspace.counts["mapped_bytes"],"model_call_count":0}
                    if repetition >= 0:
                        results.append(result)
                        print(json.dumps(result),flush=True)
                # Same-input scalar comparison is bounded independently from
                # the full-scale vectorized growth study; no extrapolated
                # large-input speedup is presented as an observation.
                start = time.perf_counter()
                reference = train_codebooks(reference_vectors)
                scalar_seconds = time.perf_counter()-start
                with GraphBuildWorkspace():
                    start = time.perf_counter()
                    model = train_rq_kmeans(reference_vectors,levels=3,max_k=6,tau_r=.65,tau_l=.35)
                    optimized_seconds = time.perf_counter()-start
                for actual,expected in zip(model["codebooks"],reference):
                    np.testing.assert_allclose(actual,expected,rtol=1e-12,atol=1e-12)
                assert all(primary_path(row,model["codebooks"]) == primary_path(row,reference) for row in reference_vectors)
                if repetition >= 0:
                    reference_runs.append({"n":len(reference_vectors),"d":d,"repetition":repetition,
                                           "scalar_seconds":scalar_seconds,"optimized_seconds":optimized_seconds})
            for row in results:
                if row["n"] == n and row["d"] == d:
                    row["same_input_reference"] = reference_runs[row["repetition"]]
    slopes = []
    for d in (128,1024):
        for metric in ("similarity_seconds","rq_core_seconds"):
            grouped = {n:sum(row[metric] for row in results if row["d"]==d and row["n"]==n)/args.repetitions for n in sizes}
            for left,right in zip(sizes,sizes[1:]):
                slopes.append({"d":d,"metric":metric,"left_n":left,"right_n":right,"observed_log_slope":math.log(grouped[right]/grouped[left])/math.log(right/left)})
    path = write_report("graph_kernel_scaling",{"synthetic":True,"warmups_per_scope":args.warmups,"repetitions":args.repetitions,
        "reference_scope":"same first min(n,64) synthetic vectors; training and primary-chain parity; full-scale growth uses the optimized implementation",
        "runs":results,"observed_slopes":slopes,"formal_complexity":{"similarity":"O(n^2*d+n^2*log(n))","rq_core":"O(L*I*K*n*d+L*n*d+L*n*log(n))"}})
    print(json.dumps({"report":str(path)}),flush=True)


if __name__ == "__main__":
    main()
