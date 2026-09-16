"""Fast database JSON transport without changing canonical graph encoding."""
from __future__ import annotations

import json
import math
import hashlib

import orjson


def database_json_loads(value):
    """Fast JSON transport decoding with the original extended value domain."""
    if type(value) not in (str, bytes, bytearray):
        return json.loads(value)
    try:
        decoded = orjson.loads(value)
    except orjson.JSONDecodeError:
        return json.loads(value)
    # Out-of-range integer literals may be decoded as binary64. Such a value
    # has magnitude at least 2**63 (including negative signed overflow). Only
    # that case needs the original arbitrary-precision decoder; ordinary
    # embedding floats avoid an additional scan of their serialized digits.
    pending = [decoded]
    while pending:
        item = pending.pop()
        kind = type(item)
        if kind is float:
            if abs(item) >= 2**63:
                return json.loads(value)
        elif kind is dict:
            pending.extend(item.values())
        elif kind is list:
            pending.extend(item)
    return decoded


def database_json_dumps(value):
    encoded = _database_json_dumps(value)
    if getattr(type(value), "_graph_json_immutable", False):
        value._database_transport_digest = (database_json_dumps, hashlib.sha256(encoded.encode("utf-8")).hexdigest())
    return encoded


def database_json_digest(value, serializer):
    immutable = getattr(type(value), "_graph_json_immutable", False)
    cached = getattr(value, "_database_transport_digest", None) if immutable else None
    if cached is not None and cached[0] is serializer:
        return cached[1]
    encoded = serializer(value)
    cached = getattr(value, "_database_transport_digest", None) if immutable else None
    return cached[1] if cached is not None and cached[0] is serializer else hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _database_json_dumps(value):
    pending=[value]
    while pending:
        item=pending.pop()
        kind=type(item)
        if item is None or kind is str or kind is bool:
            continue
        if kind is float:
            if not math.isfinite(item):
                raise ValueError("Database JSON rejects non-finite numbers")
        elif isinstance(item,dict):
            if any(type(key) is not str for key in item):
                return json.dumps(value,allow_nan=False)
            pending.extend(item.values())
        elif isinstance(item,(list,tuple)):
            pending.extend(item)
        elif kind is int:
            if not -(2**63) <= item < 2**64:
                return json.dumps(value,allow_nan=False)
        elif item is not None and kind is not str and kind is not bool:
            # Match the standard encoder for subclasses/unusual containers;
            # never let a native encoder silently accept datetime/bytes/etc.
            return json.dumps(value,allow_nan=False)
    try:
        return orjson.dumps(value).decode("utf-8")
    except orjson.JSONEncodeError:
        return json.dumps(value,allow_nan=False)
