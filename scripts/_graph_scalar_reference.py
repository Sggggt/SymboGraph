"""Frozen scalar numerical reference, for synthetic tests/benchmarks only.

This preserves the pre-vectorization initialization, summation, tie handling,
empty-cluster and convergence rules. Production never imports this module.
"""
from __future__ import annotations

import math


def squared_distance(left, right):
    return sum((float(a)-float(b))**2 for a,b in zip(left,right))


def centroid(vectors):
    return [sum(vector[i] for vector in vectors)/len(vectors)
            for i in range(min(map(len,vectors)))]


def nearest(vector, centers):
    return min(range(len(centers)),key=lambda i:squared_distance(vector,centers[i]))


def train_codebook(vectors, *, k, iterations=8):
    from app.services.context_graph import stable_hash
    if not vectors:
        return []
    ordered = sorted(vectors,key=lambda vector:stable_hash([round(value,6) for value in vector]))
    if k <= 1:
        return [centroid(ordered)]
    step = max(1,len(ordered)//k)
    centers = [ordered[min(i*step,len(ordered)-1)] for i in range(k)]
    for _ in range(iterations):
        groups = [[] for _ in centers]
        for vector in ordered:
            groups[nearest(vector,centers)].append(vector)
        updated = [centroid(group) if group else centers[i] for i,group in enumerate(groups)]
        if all(squared_distance(a,b)<1e-12 for a,b in zip(centers,updated)):
            break
        centers = updated
    return centers


def train_codebooks(vectors, *, levels=3, max_k=6):
    residuals = [list(vector) for vector in vectors]
    codebooks = []
    for _ in range(levels):
        if not residuals:
            break
        k = min(max_k,max(1,int(math.sqrt(len(residuals)))+1),len(residuals))
        centers = train_codebook(residuals,k=k)
        codebooks.append(centers)
        residuals = [[float(a)-float(b) for a,b in zip(row,centers[nearest(row,centers)])]
                     for row in residuals]
    return codebooks


def primary_path(vector, codebooks):
    residual = list(vector)
    path = []
    for centers in codebooks:
        index = nearest(residual,centers)
        path.append(index+1)
        residual = [float(a)-float(b) for a,b in zip(residual,centers[index])]
    return path
