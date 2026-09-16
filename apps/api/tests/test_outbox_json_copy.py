import json
import math
import random
from datetime import datetime
from types import SimpleNamespace

import pytest

from app.services import qdrant_outbox as outbox


def scalar_reference(value,path='$'):
    if value is None or isinstance(value,(str,bool)):return value
    if type(value) is int:return value
    if type(value) is float:
        if not math.isfinite(value):raise outbox.QdrantOutboxError(f'Qdrant outbox value at {path} must be finite JSON')
        return 0.0 if value==0.0 else value
    if isinstance(value,list):return [scalar_reference(item,f'{path}[{i}]') for i,item in enumerate(value)]
    if isinstance(value,dict):
        result={}
        for key,item in value.items():
            if not isinstance(key,str):raise outbox.QdrantOutboxError(f'Qdrant outbox object at {path} contains a non-string key')
            result[key]=scalar_reference(item,f'{path}.{key}')
        return result
    raise outbox.QdrantOutboxError(f'Qdrant outbox value at {path} is outside the frozen strict-JSON schema')


def test_copy_matches_frozen_reference_bytes_and_does_not_alias_containers():
    randomizer=random.Random(713)
    scalars=[None,True,False,0,1,-1,2**90,0.0,-0.0,.1,-2.5,'','公开测试','tab\tline\n']
    def generate(depth):
        if depth==0 or randomizer.randrange(3)==0:return randomizer.choice(scalars)
        children=[generate(depth-1) for _ in range(randomizer.randrange(5))]
        return children if randomizer.randrange(2) else {f'key_{i}':value for i,value in enumerate(children)}
    for _ in range(100):
        value=generate(4)
        expected=scalar_reference(value)
        encoded=json.dumps(expected,ensure_ascii=False,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
        assert outbox._outbox_v2_canonical_bytes(value)==encoded
    value={'rows':[[0.0,-0.0],[{'key':[1,2]}]]}
    copied=outbox._strict_json_copy(value)
    copied['rows'][1][0]['key'].append(3)
    assert value['rows'][1][0]['key']==[1,2]
    assert math.copysign(1,copied['rows'][0][1])==1


@pytest.mark.parametrize('bad',[float('nan'),float('inf'),float('-inf'),b'bytes',{1:'key'},(1,2),{1,2},datetime(2020,1,1)])
def test_invalid_nested_value_keeps_the_same_error_and_path(bad):
    value={'target':[[0.25,bad]]}
    with pytest.raises(outbox.QdrantOutboxError) as old:scalar_reference(value)
    with pytest.raises(outbox.QdrantOutboxError) as new:outbox._strict_json_copy(value)
    assert str(old.value)==str(new.value)


def test_decoder_reuses_raw_bytes_but_keeps_independent_decoded_comparison(monkeypatch):
    from test_qdrant_outbox import _strict_v2_point
    identifier='unit-test-byte-reuse'
    collection,point=_strict_v2_point('unit-test-point',[.1,.9],knowledge_base_id='unit-test-kb',owner_intent_id=identifier)
    prepared=outbox._prepare_qdrant_upsert_envelope(intent_id=identifier,knowledge_base_id='unit-test-kb',job_id=None,
        collection_name=collection,target_points=[point],before_points=[],strict_schema=True)
    contract=outbox._outbox_protocol_contract(outbox.QDRANT_OUTBOX_PROTOCOL_VERSION)
    payload={'protocol_version':outbox.QDRANT_OUTBOX_PROTOCOL_VERSION,'intent_id':identifier,'collection_name':collection,
        'target_points':list(prepared.target_points),'before_points':[],'target_payload_hash':prepared.target_payload_hash,
        'before_image_hash':prepared.before_image_hash,**contract}
    row=SimpleNamespace(id=identifier,knowledge_base_id='unit-test-kb',target_ids_json=['unit-test-point'],payload_json=payload)
    original=outbox._outbox_v2_canonical_bytes;calls=[]
    canonical_targets=outbox._canonicalize_v2_target_points
    canonical_before=outbox._canonicalize_before_points
    inputs=[]
    def targets_input(value,**kwargs):inputs.append(value);return canonical_targets(value,**kwargs)
    def before_input(value,**kwargs):inputs.append(value);return canonical_before(value,**kwargs)
    def counted(value):calls.append(id(value));return original(value)
    monkeypatch.setattr(outbox,'_outbox_v2_canonical_bytes',counted)
    monkeypatch.setattr(outbox,'_canonicalize_v2_target_points',targets_input)
    monkeypatch.setattr(outbox,'_canonicalize_before_points',before_input)
    targets,before=outbox._validated_reconcile_payload(row)
    assert targets==list(prepared.target_points) and before==[]
    assert len(calls)==4
    # The decoder first snapshots the envelope; track those inputs, not the
    # caller-owned containers intentionally separated by the strict copy.
    assert len(inputs)==2 and all(calls.count(id(value))==1 for value in inputs)
