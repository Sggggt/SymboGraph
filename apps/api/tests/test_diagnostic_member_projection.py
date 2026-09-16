from copy import deepcopy
import json
from pathlib import Path

import pytest
from sqlalchemy import event,select

from app.models import ChunkRelationGraphState,RQPrefix,RQPrefixMembership


def _reader(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[3]/'scripts'))
    from diagnose_context_graph import read_rq_membership_quality_rows
    return read_rq_membership_quality_rows


def _state(db,kb):
    return db.scalar(select(ChunkRelationGraphState.id).where(ChunkRelationGraphState.knowledge_base_id==kb,
        ChunkRelationGraphState.state=='active').order_by(ChunkRelationGraphState.created_at.desc()).limit(1))


def _old_rows(db,state):
    members=list(db.scalars(select(RQPrefixMembership).join(RQPrefix).where(RQPrefix.graph_state_id==state)
        .order_by(RQPrefixMembership.id)))
    fields=('id','rq_prefix_id','chunk_id','membership_score','membership_role',
            'membership_entropy','residual_norm','rank','rq_path')
    return [{**{name:getattr(row,name) for name in fields},'rq_path':row.rq_path or [],
        'role_evaluation':(row.diagnostics_json or {}).get('membership_role_evaluation') or {}} for row in members]


def test_projection_preserves_every_quality_value_without_loading_unused_json(
    db_session,populated_context_graph,monkeypatch,
):
    reader=_reader(monkeypatch)
    state=_state(db_session,populated_context_graph['knowledge_base'].id)
    member=db_session.scalar(select(RQPrefixMembership).join(RQPrefix).where(RQPrefix.graph_state_id==state).limit(1))
    payload=deepcopy(member.diagnostics_json)
    payload['unused_vector_block']='unit-test-unused-'*10000
    member.diagnostics_json=payload
    db_session.flush()
    expected=_old_rows(db_session,state)
    statements=[]
    def capture(connection,cursor,statement,parameters,context,executemany):
        statements.append(statement)
    event.listen(db_session.get_bind(),'before_cursor_execute',capture)
    try:
        actual=reader(db_session,state)
    finally:
        event.remove(db_session.get_bind(),'before_cursor_execute',capture)
    assert json.dumps(actual,sort_keys=True)==json.dumps(expected,sort_keys=True)
    assert len(statements)==1 and 'JSON_EXTRACT' in statements[0]
    assert 'unit-test-unused' not in json.dumps(actual)
    assert len(actual)==len(expected)>0
    assert [row['id'] for row in actual]==sorted(row['id'] for row in actual)
    assert reader(db_session,'unit-test-unrelated-state')==[] and reader(db_session,None)==[]


@pytest.mark.parametrize('diagnostics',[None,{}, {'membership_role_evaluation':'unit-test-invalid-value'}])
def test_missing_or_invalid_values_are_preserved_like_existing_reader(
    db_session,populated_context_graph,monkeypatch,diagnostics,
):
    reader=_reader(monkeypatch)
    state=_state(db_session,populated_context_graph['knowledge_base'].id)
    member=db_session.scalar(select(RQPrefixMembership).join(RQPrefix).where(RQPrefix.graph_state_id==state).limit(1))
    member.diagnostics_json=diagnostics
    db_session.flush()
    assert reader(db_session,state)==_old_rows(db_session,state)
