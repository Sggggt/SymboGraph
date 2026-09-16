from fractions import Fraction
import math
import random
import struct

import pytest

from app.services.vector_store import canonical_embedding_vector, _canonical_embedding_vector_scalar
from app.services.context_graph import _canonical_vector_payload_digest


@pytest.mark.parametrize("vector", [
    [1., -0., 0., 1e-44, -1e-44], [1 + 2**-24, 1 + 3 * 2**-24],
    [3.4028234663852886e38, -3.4028234663852886e38], [1, -2, 3],
    [Fraction(1, 3), Fraction(-1, 7)],
    [random.Random(42 + i).uniform(-1, 1) for i in range(1024)],
])
def test_native_bulk_keeps_exact_scalar_bytes_and_hash(vector):
    import hashlib
    old = _canonical_embedding_vector_scalar(vector, source="unit-test")
    actual = canonical_embedding_vector(vector, source="unit-test")
    assert actual == old
    assert struct.pack(f">{len(actual)}f", *actual) == b"".join(struct.pack(">f", v) for v in old)
    assert all(math.copysign(1, v) == 1 for v in actual if v == 0)
    identity = ("unit-test-scope", "unit-test-model")
    payload = bytearray(b"unit-test-protocol\0")
    for value in identity:
        encoded = value.encode()
        payload.extend(len(encoded).to_bytes(8, "big"))
        payload.extend(encoded)
    raw = b"".join(struct.pack(">f", value) for value in old)
    payload.extend(len(raw).to_bytes(8, "big"))
    payload.extend(raw)
    assert _canonical_vector_payload_digest(protocol_version="unit-test-protocol", vector=vector,
        embedding_dimension=len(vector), identity_values=identity) == hashlib.sha256(payload).hexdigest()


@pytest.mark.parametrize("vector", [[], [True, 1.], ["1", 1.], [complex(1, 0)],
    [float("nan")], [float("inf")], [3.5e38], [0., -0.], [1e-50]])
def test_bulk_keeps_invalid_vector_rejections(vector):
    with pytest.raises(ValueError):
        canonical_embedding_vector(vector, source="unit-test")


@pytest.mark.parametrize('dimension', [64, 65, 256, 1024, 4096])
def test_native_array_preserves_full_binary32_domain(dimension):
    rng = random.Random(918)
    vector = [rng.uniform(-1, 1) * 10.0 ** rng.randint(-44, 38) for _ in range(dimension)]
    vector[:5] = [-0.0, 0.0, 1e-44, 1 + 2**-24, 1 + 3 * 2**-24]
    expected = _canonical_embedding_vector_scalar(vector, source='unit-test')
    actual = canonical_embedding_vector(vector, source='unit-test')
    assert struct.pack(f'>{dimension}f', *actual) == struct.pack(f'>{dimension}f', *expected)
    assert all(math.copysign(1, value) == 1 for value in actual if value == 0)


@pytest.mark.parametrize('bad', [True, '1', float('nan'), float('inf'), 3.5e38])
def test_array_path_preserves_indexed_invalid_diagnostic(bad):
    vector = [1.0] * 64
    vector[-1] = bad
    with pytest.raises(ValueError) as expected:
        _canonical_embedding_vector_scalar(vector, source='unit-test')
    with pytest.raises(ValueError) as actual:
        canonical_embedding_vector(vector, source='unit-test')
    assert str(actual.value) == str(expected.value)


def test_native_norm_boundary_uses_original_exact_sum(monkeypatch):
    from app.services import vector_store
    import numpy as np
    monkeypatch.setattr(vector_store, 'MIN_EMBEDDING_VECTOR_NORM', 1.0)
    original, calls = math.fsum, []
    def record_sum(values):
        calls.append(True)
        return original(values)
    monkeypatch.setattr(vector_store.math, 'fsum', record_sum)
    with pytest.raises(ValueError, match='norm greater than'):
        canonical_embedding_vector([.125] * 64, source='unit-test')
    assert calls == [True]
    above = [.125] * 63 + [float(np.nextafter(np.float32(.125), np.float32(1.0)))]
    below = [.125] * 63 + [float(np.nextafter(np.float32(.125), np.float32(0.0)))]
    assert canonical_embedding_vector(above, source='unit-test') == _canonical_embedding_vector_scalar(above, source='unit-test')
    with pytest.raises(ValueError, match='norm greater than'):
        canonical_embedding_vector(below, source='unit-test')
