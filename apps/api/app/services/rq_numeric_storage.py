"""Lossless storage for full RQ residual/reconstruction audit vectors."""
from __future__ import annotations

import base64
import hashlib

import numpy as np

RQ_NUMERIC_STORAGE_PROTOCOL = "rq_diagnostics_float64_le_base64_v1"
MAX_DIMENSIONS = 65536


def pack_rq_vector(values):
    vector=np.asarray(values,dtype="<f8")
    if vector.ndim != 1 or not 0 < len(vector) <= MAX_DIMENSIONS or not np.isfinite(vector).all():
        raise ValueError("RQ diagnostic vector must be finite and within the dimension limit")
    data=vector.tobytes(order="C")
    return {"protocol":RQ_NUMERIC_STORAGE_PROTOCOL,"dimensions":len(vector),
            "sha256":hashlib.sha256(data).hexdigest(),"data":base64.b64encode(data).decode("ascii")}


def validate_rq_vector_blob(value):
    if not isinstance(value,dict) or set(value) != {"protocol","dimensions","sha256","data"}:
        raise ValueError("Invalid RQ numeric storage fields")
    dimensions=value["dimensions"]
    if value["protocol"] != RQ_NUMERIC_STORAGE_PROTOCOL or type(dimensions) is not int or not 0 < dimensions <= MAX_DIMENSIONS:
        raise ValueError("Invalid RQ numeric storage protocol or dimensions")
    encoded=value["data"]
    expected_length=4*((dimensions*8+2)//3)
    if not isinstance(encoded,str) or len(encoded) != expected_length:
        raise ValueError("Invalid RQ numeric storage length")
    try:
        data=base64.b64decode(encoded,validate=True)
    except (ValueError,TypeError) as exc:
        raise ValueError("Invalid RQ numeric storage encoding") from exc
    if len(data) != dimensions*8 or hashlib.sha256(data).hexdigest() != value["sha256"]:
        raise ValueError("RQ numeric storage integrity mismatch")
    vector=np.frombuffer(data,dtype="<f8")
    if not np.isfinite(vector).all():
        raise ValueError("RQ numeric storage contains non-finite values")
    return vector


def read_rq_vector(value):
    if isinstance(value,dict):
        return validate_rq_vector_blob(value).tolist()
    # Existing frozen graph states and scalar test fixtures retain their
    # original array representation; new builders never write this branch.
    if isinstance(value,list):
        return value
    if value is None:
        return []
    raise ValueError("Invalid RQ diagnostic vector representation")
