import json
import math
import random
import struct

import pytest
from app.core.json_codec import database_json_loads


@pytest.mark.parametrize('value', [
    {'integer': 2**100, 'negative': -(2**100)}, {'text': '\ud800'},
    {'text': '中文𝄞', 'items': [None, True, False, '123456789012345678901234']},
    {'floats': [-0.0, 5e-324, 1.7976931348623157e308, 0.10000000000000002]},
])
def test_database_decoder_retains_standard_json_values(value):
    encoded = json.dumps(value)
    assert json.dumps(database_json_loads(encoded), sort_keys=True) == json.dumps(json.loads(encoded), sort_keys=True)


def test_finite_binary64_differential_decode_is_bit_exact():
    rng = random.Random(20260912)
    values = []
    while len(values) < 10000:
        value = struct.unpack('d', rng.getrandbits(64).to_bytes(8, 'little'))[0]
        if math.isfinite(value):
            values.append(value)
    encoded = json.dumps(values)
    old = json.loads(encoded)
    new = database_json_loads(encoded)
    assert struct.pack(f'{len(old)}d', *old) == struct.pack(f'{len(new)}d', *new)


def test_embedding_floats_use_the_native_path(monkeypatch):
    values = [random.Random(i).uniform(-1, 1) for i in range(1024)]
    encoded = json.dumps({'vectors': [values], 'count': 1024})
    def forbid_standard(*args, **kwargs):
        raise AssertionError('ordinary floats must use the native decoder')
    monkeypatch.setattr('app.core.json_codec.json.loads', forbid_standard)
    assert database_json_loads(encoded)['vectors'][0] == values


@pytest.mark.parametrize('value', [-(2**63)-1, 2**64, 2**100, -(2**100)])
def test_integer_overflow_remains_arbitrary_precision(value):
    assert database_json_loads(str(value)) == value


@pytest.mark.parametrize('encoded', ['{', '[1,]', '{"x":NaN}', '[1e400]', '"\\ud800"'])
def test_extended_or_invalid_inputs_keep_original_decoder_behavior(encoded):
    try:
        original = json.loads(encoded)
    except json.JSONDecodeError:
        with pytest.raises(json.JSONDecodeError):
            database_json_loads(encoded)
    else:
        assert json.dumps(database_json_loads(encoded)) == json.dumps(original)
