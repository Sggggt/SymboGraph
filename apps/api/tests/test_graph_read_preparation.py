from copy import deepcopy
from types import SimpleNamespace

import pytest

from app.services import context_graph as graph


def test_multi_layer_read_keeps_payloads_and_prepares_stats_once(db_session,populated_context_graph,monkeypatch):
    kb=populated_context_graph['knowledge_base'].id
    layers=('chunk-structure','chunk-relation','mid-concepts','coarse-concepts')
    fixed_stats=graph.context_graph_stats(db_session,kb)
    calls=[]
    def stats(db,knowledge_base_id,**kwargs):
        assert db is db_session and knowledge_base_id==kb
        assert kwargs['chunks'] and len(kwargs['bound_layers'])==3
        calls.append(1)
        return deepcopy(fixed_stats)
    monkeypatch.setattr(graph,'context_graph_stats',stats)
    expected={layer:graph.graph_layer_payload(db_session,kb,layer,limit=5) for layer in layers}
    assert len(calls)==4
    calls.clear()
    actual=graph.graph_layers_payload(db_session,kb,layers,limit=5)
    assert len(calls)==1
    assert actual['layers']==expected and actual['stats']==fixed_stats


def test_postgres_multi_layer_read_requires_snapshot_isolation_before_querying():
    class Database:
        def get_bind(self):return SimpleNamespace(dialect=SimpleNamespace(name='postgresql'))
        def connection(self):return SimpleNamespace(get_isolation_level=lambda:'READ COMMITTED')
        def scalars(self,*args):raise AssertionError('No graph rows may be read before isolation validation')
    with pytest.raises(ValueError,match='require_repeatable_read'):
        graph.graph_layers_payload(Database(),'unit-test-kb',('chunk-structure','chunk-relation'))


@pytest.mark.parametrize('layers',[(),('chunk-structure','chunk-structure'),('unknown',)])
def test_layer_selection_is_closed(layers):
    with pytest.raises(ValueError,match='graph_layer_selection_invalid'):
        graph.graph_layers_payload(None,'unit-test-kb',layers)
