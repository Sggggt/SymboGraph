"""Bounded navigation inside an already-proven source document version."""
from __future__ import annotations

from copy import deepcopy
import re
import unicodedata
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models import Chunk, ChunkStructureNode, ChunkStructureMapping, ChunkStructureEdge, ContextPackage, ContextPackageSourceExpansion, RetrievalTrace
from app.services.agent_reflection import ReflectionContractError, reflection_hash


EXPANSION_PROTOCOL = "reflection_source_structure_expansion_v1"
MAX_DOCUMENT_CHUNKS = 4096
MAX_STRUCTURE_DEPTH = 32
FOCUSED_NAVIGATION_PROTOCOL = "source_structure_facet_navigation_v3"
PREVIOUS_FOCUSED_NAVIGATION_PROTOCOL = "source_structure_facet_navigation_v2"


def restoration_section_locators(focus: list[str]) -> list[dict[str, Any]]:
    """Parse explicit addresses only; prose is never executable authority."""
    number = r"\d{1,3}(?:\.\d{1,3}){0,5}"
    pattern = re.compile(rf"(?:§\s*({number})(?!\d|\.\d)|\b(?:section|chapter)\s+({number})(?!\d|\.\d)\b|第\s*({number})\s*[章节])"
        r"(\s*(?:及以后|及之后|及后续|and\s+later|onwards?))?", re.IGNORECASE)
    locators = []
    for text in focus:
        for match in pattern.finditer(unicodedata.normalize("NFKC", text)):
            raw = next(value for value in match.groups()[:3] if value is not None)
            item = {"section": [int(part) for part in raw.split(".")], "at_or_after": bool(match.group(4))}
            if item not in locators:
                locators.append(item)
                if len(locators) > 8:
                    raise ReflectionContractError("structure_section_locator_capacity_exceeded")
    return locators


def _validated_restoration_focus(value: list[str] | None) -> list[str]:
    if value is None:
        return []
    if (not isinstance(value, list) or len(value) > 8 or any(not isinstance(item, str) or not item.strip()
        or len(item) > 180 or "\x00" in item for item in value) or len(value) != len(set(value))):
        raise ReflectionContractError("structure_restoration_focus_invalid")
    return list(value)


def _heading_section_number(title: str) -> list[int] | None:
    title = unicodedata.normalize("NFKC", title).strip().lstrip("# ")
    if re.search(r"\.{4,}|…{2,}", title):
        return None  # Contents entries are addresses, not the target section.
    match = re.match(r"(?:section\s+|chapter\s+)?(\d{1,3}(?:\.\d{1,3}){0,5})(?=[\s.:、：-])", title, re.IGNORECASE)
    return [int(part) for part in match.group(1).split(".")] if match else None


def expansion_identity(row: ContextPackageSourceExpansion) -> dict[str, Any]:
    return {key: getattr(row, key) for key in (
        "knowledge_base_id", "target_context_package_id", "source_context_package_id", "source_retrieval_trace_id",
        "anchor_chunk_id", "chunk_id", "protocol_version", "anchor_item_hash", "target_item_hash", "witness_json")}


def _row_fact(row) -> dict[str, Any]:
    return {column.name: deepcopy(getattr(row, column.name)) for column in row.__table__.columns if column.name != "created_at"}


def structure_witness(db: Session, anchor: Chunk, target: Chunk, *, for_update: bool = False) -> dict[str, Any]:
    if (anchor.knowledge_base_id != target.knowledge_base_id or anchor.document_version_id != target.document_version_id
        or anchor.document_id != target.document_id or anchor.state != "active" or target.state != "active"):
        raise ReflectionContractError("structure_expansion_source_scope_invalid")

    def mapped_chain(chunk):
        statement = select(ChunkStructureMapping, ChunkStructureNode).join(
            ChunkStructureNode, ChunkStructureNode.id == ChunkStructureMapping.structure_node_id).where(
            ChunkStructureMapping.chunk_id == chunk.id, ChunkStructureMapping.document_version_id == chunk.document_version_id,
            ChunkStructureNode.document_version_id == chunk.document_version_id,
            ChunkStructureNode.knowledge_base_id == chunk.knowledge_base_id,
        ).order_by(ChunkStructureNode.depth.desc(), ChunkStructureMapping.mapping_weight.desc(), ChunkStructureNode.id)
        if for_update:
            statement = statement.with_for_update().execution_options(populate_existing=True)
        pairs = list(db.execute(statement).all())
        if not pairs:
            raise ReflectionContractError("structure_expansion_mapping_missing")
        mapping, node = pairs[0]
        chain = []
        while node is not None:
            if (len(chain) >= MAX_STRUCTURE_DEPTH or any(old.id == node.id for old in chain)
                or node.knowledge_base_id != chunk.knowledge_base_id or node.document_version_id != chunk.document_version_id
                or node.document_id != chunk.document_id):
                raise ReflectionContractError("structure_expansion_ancestry_invalid")
            chain.append(node)
            if not node.parent_id:
                break
            statement = select(ChunkStructureNode).where(ChunkStructureNode.id == node.parent_id)
            if for_update:
                statement = statement.with_for_update().execution_options(populate_existing=True)
            parent = db.scalar(statement)
            if parent is None:
                raise ReflectionContractError("structure_expansion_parent_missing")
            node = parent
        return mapping, chain

    anchor_mapping, left = mapped_chain(anchor)
    target_mapping, right = mapped_chain(target)
    right_ids = {node.id: index for index, node in enumerate(right)}
    common = next(((index, right_ids[node.id]) for index, node in enumerate(left) if node.id in right_ids), None)
    if common is None:
        raise ReflectionContractError("structure_expansion_disconnected_source")
    li, ri = common
    nodes = [*left[:li + 1], *reversed(right[:ri])]
    edges = []
    for first, second in zip(nodes, nodes[1:]):
        parent, child = (second, first) if first.parent_id == second.id else (first, second)
        if child.parent_id != parent.id:
            raise ReflectionContractError("structure_expansion_parent_path_invalid")
        statement = select(ChunkStructureEdge).where(
            ChunkStructureEdge.knowledge_base_id == anchor.knowledge_base_id,
            ChunkStructureEdge.document_version_id == anchor.document_version_id,
            ChunkStructureEdge.source_node_id == parent.id, ChunkStructureEdge.target_node_id == child.id,
            ChunkStructureEdge.edge_type == "parent_child").order_by(ChunkStructureEdge.id)
        if for_update:
            statement = statement.with_for_update().execution_options(populate_existing=True)
        edge = db.scalar(statement)
        if edge is None:
            raise ReflectionContractError("structure_expansion_edge_missing")
        edges.append(_row_fact(edge))
    return {"protocol_version": EXPANSION_PROTOCOL, "document_version_id": anchor.document_version_id,
        "anchor_mapping": _row_fact(anchor_mapping), "target_mapping": _row_fact(target_mapping),
        "nodes": [_row_fact(node) for node in nodes], "edges": edges}


def _words(value: str) -> set[str]:
    value = unicodedata.normalize("NFKC", value).casefold()
    value = value.replace("摘要", " summary ").replace("概述", " summary ").replace("abstract", "summary")
    return set(re.findall(r"[a-z0-9]+|[\u4e00-\u9fff]+", value))


def _alias_forms(value: str) -> list[set[str]]:
    words = _words(value)
    latin = {word for word in words if re.fullmatch(r"[a-z0-9]+", word)}
    other = words - latin
    return [words, latin, other] if latin and other else [words]


def select_source_expansions(db: Session, *, source_package: ContextPackage, anchor_ids: list[str],
    query_facets: dict[str, Any], per_anchor_budget: int, restoration_focus: list[str] | None = None,
    _replay_rank_protocol: str | None = None) -> list[dict[str, Any]]:
    focus = _validated_restoration_focus(restoration_focus)
    locators = restoration_section_locators(focus)
    rank_protocol = _replay_rank_protocol or (FOCUSED_NAVIGATION_PROTOCOL if focus else "source_structure_facet_navigation_v1")
    if (rank_protocol not in {"source_structure_facet_navigation_v1", PREVIOUS_FOCUSED_NAVIGATION_PROTOCOL, FOCUSED_NAVIGATION_PROTOCOL}
        or (rank_protocol == "source_structure_facet_navigation_v1") != (not focus)):
        raise ReflectionContractError("structure_expansion_navigation_protocol_invalid")
    continuation_directions = []
    if rank_protocol == FOCUSED_NAVIGATION_PROTOCOL:
        focus_text = " ".join(focus).casefold()
        if any(word in focus_text for word in ("截断", "后续", "下一", "之后", "truncat", "continuation", "following text", "next chunk", "next passage")):
            continuation_directions.append("next")
        if any(word in focus_text for word in ("前文", "上一", "previous chunk", "previous passage")):
            continuation_directions.append("previous")
    if type(per_anchor_budget) is not int or per_anchor_budget < 0 or len(anchor_ids) > 256:
        raise ReflectionContractError("structure_expansion_budget_invalid")
    anchor_ids = list(dict.fromkeys(anchor_ids))
    if per_anchor_budget == 0:
        return []
    items = {item["chunk_id"]: item for item in source_package.package_json["chunks"]}
    if not set(anchor_ids).issubset(items):
        raise ReflectionContractError("structure_expansion_anchor_outside_package")
    groups = [group for group in query_facets.get("facet_groups", []) if isinstance(group, dict)]
    terms = [[_words(str(term)) for term in [group.get("facet", ""), *group.get("aliases", [])] if str(term).strip()] for group in groups]
    if rank_protocol == FOCUSED_NAVIGATION_PROTOCOL:
        terms = [[_words(str(group.get("facet", ""))), *(variant for alias in group.get("aliases", [])
            for variant in _alias_forms(str(alias)))] for group in groups]
    requested_summary = any("summary" in term for group in terms for term in group)
    requested_detail = any("detailed" in term or "详细" in term for group in terms for term in group)
    if focus:
        # A mention of the summary inside a section-specific request is not
        # another request to restore the summary. Separate locations use
        # separate missing-facet entries in the existing closed decision.
        requested_summary = any("summary" in _words(item) and not restoration_section_locators([item]) for item in focus)
    topic_words = set().union(*(term for group in terms for term in group)) - {"and", "the", "of", "in", "for", "a", "an"}
    domain_groups = {index for index, group in enumerate(groups) if group.get("role") == "domain"}
    selected = []
    selected_ids = set(items)
    summary_by_document = {}
    summary_windows = {}
    pending_summary_continuation = set()
    for anchor_position, anchor_id in enumerate(anchor_ids):
        anchor = db.get(Chunk, anchor_id)
        if anchor is None or anchor.document_version_id != items[anchor_id]["document_version_id"]:
            raise ReflectionContractError("structure_expansion_anchor_changed")
        candidates = list(db.scalars(select(Chunk).where(Chunk.knowledge_base_id == source_package.knowledge_base_id,
            Chunk.document_version_id == anchor.document_version_id, Chunk.state == "active").order_by(Chunk.chunk_index, Chunk.id).limit(MAX_DOCUMENT_CHUNKS + 1)))
        if len(candidates) > MAX_DOCUMENT_CHUNKS:
            raise ReflectionContractError("structure_expansion_document_capacity_exceeded")
        summary_overlap = {}
        section_matches = {}
        non_summary_mapped = set()
        outline_nodes = {}
        if requested_summary or locators:
            rows = list(db.execute(select(ChunkStructureMapping, ChunkStructureNode).join(
                ChunkStructureNode, ChunkStructureNode.id == ChunkStructureMapping.structure_node_id).where(
                ChunkStructureMapping.document_version_id == anchor.document_version_id,
                ChunkStructureNode.document_version_id == anchor.document_version_id,
                ChunkStructureNode.node_type.in_(["section", "heading"])).limit(MAX_DOCUMENT_CHUNKS * 16 + 1)))
            if len(rows) > MAX_DOCUMENT_CHUNKS * 16:
                raise ReflectionContractError("structure_expansion_outline_capacity_exceeded")
            for mapping, node in rows:
                outline_nodes[node.id] = node
                cid, overlap, title = mapping.chunk_id, mapping.overlap_chars, node.title
                if requested_summary and _words(title or "") in ({"summary"}, {"executive", "summary"}):
                    summary_overlap[cid] = max(summary_overlap.get(cid, 0), int(overlap or 0))
                elif int(overlap or 0) > 0:
                    non_summary_mapped.add(cid)
                section = _heading_section_number(title or "") if locators else None
                if section is not None and int(overlap or 0) > 0 and any(tuple(section) >= tuple(item["section"]) if item["at_or_after"]
                    else section[:len(item["section"])] == item["section"] for item in locators):
                    match = {"section": section, "title_overlap": len(_words(title or "") & topic_words),
                        "mapping": _row_fact(mapping), "node": _row_fact(node)}
                    key = lambda item: (-item["title_overlap"], -int(item["mapping"].get("overlap_chars") or 0), item["node"]["id"], item["mapping"]["id"])
                    if cid not in section_matches or key(match) < key(section_matches[cid]):
                        section_matches[cid] = match
        ranked = []
        for candidate in candidates:
            if candidate.id in selected_ids:
                continue
            words = _words(candidate.text)
            matched = {index for index, group in enumerate(terms) if any(term and term.issubset(words) for term in group)}
            summary = bool(summary_overlap.get(candidate.id, 0))
            if summary:
                matched.add("summary_section")
            adjacent = candidate.id in {anchor.previous_chunk_id, anchor.next_chunk_id}
            ranked.append((candidate, matched, summary, adjacent))
        covered = set()
        summary_index = None
        for slot in range(per_anchor_budget):
            summary_continuation = False
            summary_anchor = None
            summary_window = None
            continuation_direction = None
            remaining = [entry for entry in ranked if entry[0].id not in selected_ids]
            if not remaining:
                break
            summary_candidates = [entry for entry in remaining if entry[2]]
            detail_candidates = [entry for entry in remaining if not entry[2] and entry[1].intersection(domain_groups)]
            continuation_candidates = [entry for entry in remaining if entry[1].intersection(domain_groups)
                and (not entry[2] or entry[0].id in non_summary_mapped)]
            section_candidates = [entry for entry in remaining if entry[0].id in section_matches]
            neighbor_choice = next(((entry, direction) for direction in continuation_directions for entry in remaining
                if entry[0].id == getattr(anchor, f"{direction}_chunk_id")), None)
            remaining_slots = (len(anchor_ids) - anchor_position - 1) * per_anchor_budget + per_anchor_budget - slot
            if neighbor_choice is not None:
                (candidate, matched, summary, adjacent), continuation_direction = neighbor_choice
            elif (slot == 0 or rank_protocol == FOCUSED_NAVIGATION_PROTOCOL) and summary_candidates and (not focus or anchor.document_version_id not in summary_by_document):
                candidate, matched, summary, adjacent = min(summary_candidates, key=lambda entry: (
                    -summary_overlap[entry[0].id], -len(entry[1]), entry[0].chunk_index, entry[0].id))
                summary_index = candidate.chunk_index
                if focus:
                    summary_by_document[anchor.document_version_id] = (candidate.id, summary_index)
                    if rank_protocol == FOCUSED_NAVIGATION_PROTOCOL:
                        summary_nodes = [node for node in outline_nodes.values() if _words(node.title or "") in ({"summary"}, {"executive", "summary"})
                            and node.char_start <= candidate.char_end and node.char_end >= candidate.char_start]
                        if summary_nodes:
                            start_node = max(summary_nodes, key=lambda node: (node.char_start, node.id))
                            boundaries = [node for node in outline_nodes.values() if node.char_start > start_node.char_start
                                and (_words(node.title or "") in ({"contents"}, {"table", "of", "contents"}, {"目录"})
                                     or _heading_section_number(node.title or "") is not None)]
                            end_node = min(boundaries, key=lambda node: (node.char_start, node.id)) if boundaries else None
                            summary_windows[anchor.document_version_id] = {"start": start_node.char_start,
                                "end": end_node.char_start if end_node else max(c.char_end for c in candidates)}
                    if requested_detail:
                        pending_summary_continuation.add(anchor.document_version_id)
            elif (focus and locators and anchor.document_version_id in pending_summary_continuation
                and remaining_slots > 1 and continuation_candidates):
                # A summary can span several chunks even when a parser marks
                # an intervening sentence as a heading. Preserve the existing
                # nearest-topic continuation, but reserve a slot for the
                # explicitly requested body section across the bounded anchors.
                summary_anchor, summary_index = summary_by_document[anchor.document_version_id]
                if rank_protocol == FOCUSED_NAVIGATION_PROTOCOL:
                    summary_window = summary_windows.get(anchor.document_version_id)
                    forward = [entry for entry in continuation_candidates if entry[0].chunk_index > summary_index
                        and (summary_window is None or entry[0].char_start < summary_window["end"])]
                    if forward:
                        def specificity(entry):
                            words = _words(entry[0].text)
                            return sum(len(term) for index in domain_groups for term in terms[index] if term and term.issubset(words))
                        candidate, matched, summary, adjacent = min(forward, key=lambda entry: (
                            -specificity(entry), entry[0].chunk_index - summary_index, entry[0].chunk_index, entry[0].id))
                    else:
                        if not section_candidates:
                            break
                        candidate, matched, summary, adjacent = min(section_candidates, key=lambda entry: (
                            -section_matches[entry[0].id]["title_overlap"], -len(entry[1]), entry[0].chunk_index, entry[0].id))
                        summary_anchor = None
                        summary_window = None
                else:
                    candidate, matched, summary, adjacent = min(continuation_candidates, key=lambda entry: (
                        abs(entry[0].chunk_index - summary_index), -len(entry[1]), entry[0].chunk_index, entry[0].id))
                summary_continuation = summary_anchor is not None
                pending_summary_continuation.remove(anchor.document_version_id)
            elif locators and section_candidates:
                candidate, matched, summary, adjacent = min(section_candidates, key=lambda entry: (
                    -section_matches[entry[0].id]["title_overlap"], -len(entry[1]), entry[0].chunk_index, entry[0].id))
            elif locators:
                break
            elif requested_detail and summary_index is not None and detail_candidates:
                candidate, matched, summary, adjacent = min(detail_candidates, key=lambda entry: (
                    abs(entry[0].chunk_index - summary_index), -len(entry[1]), entry[0].chunk_index, entry[0].id))
            else:
                candidate, matched, summary, adjacent = min(remaining, key=lambda entry: (
                    -len(entry[1] - covered), -int(entry[2]), -len(entry[1]), -int(entry[3]), entry[0].chunk_index, entry[0].id))
            if not matched and not adjacent and candidate.id not in section_matches:
                break
            selected_ids.add(candidate.id)
            covered.update(matched)
            selected.append({"anchor_chunk_id": anchor.id, "chunk_id": candidate.id,
                "structure": structure_witness(db, anchor, candidate),
                "navigation": {"query_facet_hash": reflection_hash(query_facets), "query_facets": deepcopy(query_facets),
                    "anchor_ids": anchor_ids, "selection_index": slot, "candidate_count": len(candidates),
                    "per_anchor_budget": per_anchor_budget, "matched_group_indexes": sorted(x for x in matched if type(x) is int),
                    "summary_section": summary, "adjacent": adjacent,
                    "rank_protocol": rank_protocol,
                    **({"restoration_focus": focus, "section_locators": locators,
                        "section_match": section_matches.get(candidate.id),
                        "summary_continuation": summary_continuation,
                        "summary_anchor_chunk_id": summary_anchor} if focus else {}),
                    **({"continuation_direction": continuation_direction,
                        "continuation_anchor_chunk_id": anchor.id if continuation_direction else None,
                        "summary_window": summary_window}
                        if rank_protocol == FOCUSED_NAVIGATION_PROTOCOL else {})}})
    return selected


def audit_source_expansions(db: Session, *, package: ContextPackage, for_update: bool, ancestors: tuple[str, ...]):
    from app.services.citation_provenance import audit_citation_provenance
    from app.services.context_graph import context_package_to_contexts
    from app.services.reflection_sources import source_citation
    statement = select(ContextPackageSourceExpansion).where(ContextPackageSourceExpansion.target_context_package_id == package.id)
    if for_update:
        statement = statement.with_for_update().execution_options(populate_existing=True)
    rows = list(db.scalars(statement))
    declared = (package.diagnostics_json or {}).get("source_expansion") or {}
    reasons = []
    declared_ids = declared.get("expanded_chunk_ids") or []
    if len(declared_ids) != len(set(declared_ids)) or set(declared_ids) != {row.chunk_id for row in rows}:
        reasons.append("context_expansion_row_scope_mismatch")
    if rows and (declared.get("protocol_version") != EXPANSION_PROTOCOL
        or declared.get("graph_hits_added") != 0 or declared.get("gray_zone_model_call_count") != 0
        or not set(declared_ids).issubset(package.restored_chunk_ids_json or [])):
        reasons.append("context_expansion_protocol_mismatch")
    supports = {}
    replayed_selections = {}
    for row in rows:
        statement = select(ContextPackage).where(ContextPackage.id == row.source_context_package_id)
        if for_update:
            statement = statement.with_for_update().execution_options(populate_existing=True)
        origin = db.scalar(statement)
        source = next((item for item in (origin.package_json if origin else {}).get("chunks", []) if item["chunk_id"] == row.anchor_chunk_id), None)
        target = next((item for item in package.package_json["chunks"] if item["chunk_id"] == row.chunk_id), None)
        valid = bool(origin and source and target and origin.id != package.id
            and origin.knowledge_base_id == package.knowledge_base_id == row.knowledge_base_id
            and origin.retrieval_trace_id == row.source_retrieval_trace_id and row.protocol_version == EXPANSION_PROTOCOL
            and reflection_hash(source) == row.anchor_item_hash and reflection_hash(target) == row.target_item_hash
            and reflection_hash(expansion_identity(row)) == row.witness_hash
            and row.witness_json.get("anchor_chunk_id") == row.anchor_chunk_id
            and row.witness_json.get("chunk_id") == row.chunk_id)
        origin_hash = None
        if valid:
            try:
                def load_chunk(chunk_id):
                    statement = select(Chunk).where(Chunk.id == chunk_id)
                    if for_update:
                        statement = statement.with_for_update().execution_options(populate_existing=True)
                    return db.scalar(statement)
                anchor = load_chunk(row.anchor_chunk_id)
                chunk = load_chunk(row.chunk_id)
                valid = anchor is not None and chunk is not None and structure_witness(db, anchor, chunk, for_update=for_update) == row.witness_json.get("structure")
                if valid:
                    navigation = row.witness_json.get("navigation") or {}
                    selection_input = {key: navigation.get(key) for key in ("anchor_ids", "query_facets", "per_anchor_budget")}
                    if navigation.get("rank_protocol") in {FOCUSED_NAVIGATION_PROTOCOL, PREVIOUS_FOCUSED_NAVIGATION_PROTOCOL}:
                        selection_input["restoration_focus"] = navigation.get("restoration_focus")
                        selection_input["_replay_rank_protocol"] = navigation["rank_protocol"]
                    elif navigation.get("rank_protocol") != "source_structure_facet_navigation_v1":
                        raise ReflectionContractError("structure_expansion_navigation_protocol_invalid")
                    selection_key = (origin.id, reflection_hash(selection_input))
                    if selection_key not in replayed_selections:
                        if (not isinstance(selection_input["anchor_ids"], list) or not isinstance(selection_input["query_facets"], dict)
                            or selection_input["anchor_ids"] != list(dict.fromkeys(selection_input["anchor_ids"]))):
                            raise ReflectionContractError("structure_expansion_navigation_invalid")
                        selected = select_source_expansions(db, source_package=origin, **selection_input)
                        replayed_selections[selection_key] = {item["chunk_id"]: item for item in selected}
                    valid = replayed_selections[selection_key].get(row.chunk_id) == row.witness_json
                if valid:
                    proof = audit_citation_provenance(db, knowledge_base_id=package.knowledge_base_id, package=origin,
                        citations=[source_citation(source, origin)], contexts=context_package_to_contexts(origin),
                        for_update=for_update, _retention_ancestors=(*ancestors, package.id))
                    valid = proof["all_valid"]
                    origin_hash = proof["provenance_session_hash"]
            except (ReflectionContractError, KeyError, TypeError, ValueError):
                valid = False
        supports[row.chunk_id] = {"valid": bool(valid), "witness_hash": row.witness_hash,
            "anchor_chunk_id": row.anchor_chunk_id, "source_context_package_id": row.source_context_package_id,
            "anchor_provenance_hash": origin_hash}
    return supports, reasons


def _has_native_hit_authority(db: Session, *, package: ContextPackage, item: dict[str, Any]) -> bool:
    if item.get("role") != "hit":
        return False
    trace = db.get(RetrievalTrace, package.retrieval_trace_id)
    if (trace is None or trace.knowledge_base_id != package.knowledge_base_id
        or item["chunk_id"] not in (package.hit_chunk_ids_json or [])
        or item["chunk_id"] not in (trace.result_chunk_ids_json or [])):
        raise ReflectionContractError("structure_expansion_native_hit_authority_missing")
    return True


def persist_source_expansions(db: Session, *, source: ContextPackage, target: ContextPackage, witnesses: list[dict]) -> None:
    originals = {item["chunk_id"]: item for item in source.package_json["chunks"]}
    targets = {item["chunk_id"]: item for item in target.package_json["chunks"]}
    for witness in witnesses:
        anchor_id, chunk_id = witness["anchor_chunk_id"], witness["chunk_id"]
        if anchor_id not in originals or chunk_id not in targets:
            raise ReflectionContractError("structure_expansion_required_span_did_not_fit")
        # Navigation may prioritize a previously budget-skipped trace hit.
        # Its original hit proof remains authoritative; expansion rows belong
        # only to restored sources, even when selection used the same navigator.
        if _has_native_hit_authority(db, package=target, item=targets[chunk_id]):
            continue
        row = ContextPackageSourceExpansion(knowledge_base_id=target.knowledge_base_id,
            target_context_package_id=target.id, source_context_package_id=source.id,
            source_retrieval_trace_id=source.retrieval_trace_id, anchor_chunk_id=anchor_id, chunk_id=chunk_id,
            protocol_version=EXPANSION_PROTOCOL, anchor_item_hash=reflection_hash(originals[anchor_id]),
            target_item_hash=reflection_hash(targets[chunk_id]), witness_json=deepcopy(witness))
        row.witness_hash = reflection_hash(expansion_identity(row))
        db.add(row)
    db.flush()
    expanded = list(db.scalars(select(ContextPackageSourceExpansion.chunk_id).where(ContextPackageSourceExpansion.target_context_package_id == target.id)))
    if expanded:
        target.diagnostics_json = {**target.diagnostics_json, "source_expansion": {
            "protocol_version": EXPANSION_PROTOCOL, "expanded_chunk_ids": sorted(expanded),
            "graph_hits_added": 0, "gray_zone_model_call_count": 0}}
        db.flush()
    elif "source_expansion" in (target.diagnostics_json or {}):
        target.diagnostics_json = {key: value for key, value in target.diagnostics_json.items()
            if key != "source_expansion"}
        db.flush()


def copy_source_expansions(db: Session, *, source: ContextPackage, target: ContextPackage) -> None:
    targets = {item["chunk_id"]: item for item in target.package_json["chunks"]}
    for original in db.scalars(select(ContextPackageSourceExpansion).where(ContextPackageSourceExpansion.target_context_package_id == source.id)):
        if original.chunk_id not in targets:
            continue
        if _has_native_hit_authority(db, package=target, item=targets[original.chunk_id]):
            continue
        values = {column.name: deepcopy(getattr(original, column.name)) for column in ContextPackageSourceExpansion.__table__.columns if column.name not in {"id", "created_at"}}
        values.update(target_context_package_id=target.id, target_item_hash=reflection_hash(targets[original.chunk_id]))
        row = ContextPackageSourceExpansion(**values)
        row.witness_hash = reflection_hash(expansion_identity(row))
        db.add(row)
    persist_source_expansions(db, source=source, target=target, witnesses=[])
