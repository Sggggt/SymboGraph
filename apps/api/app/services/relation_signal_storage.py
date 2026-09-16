"""Lossless graph-local storage for repeatedly referenced node signal cards."""
from __future__ import annotations

import base64
import hashlib
import json
import zlib

PROTOCOL = "relation_node_signal_pool_v1"
STATE_KEY = "node_signal_pool"
MAX_BYTES = 64*1024**2
SIGNALS = {
    "source_out_signal_card":("source","out_evidence_mass"),
    "target_in_acceptance_signal_card":("target","in_acceptance_capacity"),
    "source_node_quality_card":("source","node_quality"),
    "target_node_quality_card":("target","node_quality"),
}


def compact_signal_features(features, pool):
    def compact(card, source, target):
        result=dict(card)
        for field,(side,kind) in SIGNALS.items():
            value=result.get(field)
            if not value:
                continue
            if not isinstance(value,dict):
                raise ValueError("Relation node signal must be an object")
            endpoint=source if side == "source" else target
            if not endpoint:
                raise ValueError("Relation node signal lacks an endpoint")
            chunk_id=str(endpoint)
            slots=pool.setdefault(chunk_id,{})
            previous=slots.get(kind)
            if previous is not None and previous != value:
                raise ValueError("One graph operating point has inconsistent node signal facts")
            slots[kind]=value
            result[field]={"protocol":PROTOCOL,"chunk_id":chunk_id,"signal_kind":kind}
        return result
    source=features.get("directed_source_chunk_id")
    target=features.get("directed_target_chunk_id")
    if not source or not target:
        return dict(features)
    result=compact(features,source,target)
    result["directional_contributions"]=[compact(item,item.get("source_chunk_id"),item.get("target_chunk_id"))
        for item in features.get("directional_contributions",[])]
    result["node_signal_storage_protocol"]=PROTOCOL
    return result


def pack_signal_pool(pool):
    raw=json.dumps(pool,ensure_ascii=False,sort_keys=True,separators=(",",":"),allow_nan=False).encode("utf-8")
    if len(raw)>MAX_BYTES:
        raise MemoryError("Relation node signal pool exceeds its storage budget")
    return {"protocol":PROTOCOL,"raw_bytes":len(raw),"sha256":hashlib.sha256(raw).hexdigest(),
            "data":base64.b64encode(zlib.compress(raw,1)).decode("ascii")}


def unpack_signal_pool(blob):
    if not isinstance(blob,dict) or set(blob) != {"protocol","raw_bytes","sha256","data"}:
        raise ValueError("Invalid relation node signal storage fields")
    if blob["protocol"] != PROTOCOL or type(blob["raw_bytes"]) is not int or not 0<=blob["raw_bytes"]<=MAX_BYTES:
        raise ValueError("Invalid relation node signal storage protocol or size")
    if not isinstance(blob["data"],str) or len(blob["data"])>2*MAX_BYTES:
        raise ValueError("Invalid relation node signal storage encoding size")
    try:
        compressed=base64.b64decode(blob["data"],validate=True)
        decoder=zlib.decompressobj()
        raw=decoder.decompress(compressed,blob["raw_bytes"]+1)
    except (ValueError,zlib.error) as exc:
        raise ValueError("Invalid relation node signal storage encoding") from exc
    if not decoder.eof or decoder.unused_data or len(raw)!=blob["raw_bytes"] or hashlib.sha256(raw).hexdigest()!=blob["sha256"]:
        raise ValueError("Relation node signal storage integrity mismatch")
    def reject_constant(_value):
        raise ValueError("Relation node signals contain non-finite numbers")
    result=json.loads(raw,parse_constant=reject_constant)
    if not isinstance(result,dict):
        raise ValueError("Relation node signal pool must be an object")
    for chunk_id,slots in result.items():
        if not chunk_id or not isinstance(slots,dict) or not set(slots)<=set(kind for _side,kind in SIGNALS.values()) or not all(isinstance(card,dict) for card in slots.values()):
            raise ValueError("Invalid relation node signal pool entries")
    return result


def validate_signal_refs(features,pool,endpoints):
    if features.get("node_signal_storage_protocol") != PROTOCOL:
        return
    for card in [features,*features.get("directional_contributions",[])]:
        for field,(side,kind) in SIGNALS.items():
            ref=card.get(field)
            if not ref: continue
            if not isinstance(ref,dict) or set(ref)!={"protocol","chunk_id","signal_kind"} or ref["protocol"]!=PROTOCOL or ref["signal_kind"]!=kind:
                raise ValueError("Invalid relation node signal reference")
            if ref["chunk_id"] not in endpoints or kind not in pool.get(ref["chunk_id"],{}):
                raise ValueError("Relation node signal reference is missing or outside its endpoints")
            expected=card.get(f"directed_{side}_chunk_id",card.get(f"{side}_chunk_id"))
            if ref["chunk_id"]!=expected:
                raise ValueError("Relation node signal reference has the wrong direction")


def expand_features_from_pool(features,pool,endpoints):
    validate_signal_refs(features,pool,endpoints)
    def expand(card):
        result=dict(card)
        for field in SIGNALS:
            ref=result.get(field)
            if ref: result[field]=pool[ref["chunk_id"]][ref["signal_kind"]]
        return result
    result=expand(features)
    result["directional_contributions"]=[expand(item) for item in features.get("directional_contributions",[])]
    return result


def expand_signal_features(edge,features):
    if features.get("node_signal_storage_protocol") != PROTOCOL:
        return features
    from sqlalchemy.orm import object_session
    from app.models import ChunkRelationGraphState
    db=object_session(edge)
    if db is None:
        raise ValueError("Stored relation signal trace requires its database scope")
    state=db.get(ChunkRelationGraphState,edge.graph_state_id)
    if state is None:
        raise ValueError("Relation signal trace state is missing")
    blob=(state.diagnostics_json or {}).get(STATE_KEY)
    if not isinstance(blob,dict):
        raise ValueError("Relation signal trace pool is missing")
    identity=(tuple(sorted(blob)),*(blob.get(key) for key in ("protocol","raw_bytes","sha256","data")))
    cached=getattr(state,"_signal_read_cache",None)
    if cached is None or cached[0]!=identity:
        cached=(identity,unpack_signal_pool(blob))
        state._signal_read_cache=cached
    pool=cached[1]
    return expand_features_from_pool(features,pool,{edge.source_chunk_id,edge.target_chunk_id})
