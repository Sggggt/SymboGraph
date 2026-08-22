from __future__ import annotations

from collections import Counter, defaultdict

from sqlalchemy import func, insert, select


def _reference_structure_positive_context(
    db,
    chunks,
    probes,
    *,
    chunk_business_keys: dict[str, str],
):
    """Frozen replay of the pre-bounded implementation for semantic equality."""

    from app.models import (
        ChunkSpan,
        ChunkStructureEdge,
        ChunkStructureMapping,
        ChunkStructureNode,
    )
    from app.services import auto_tpe

    chunk_by_id = {str(chunk.id): chunk for chunk in chunks}
    chunk_ids = set(chunk_by_id)
    targets = {
        str(probe.id): {
            "previous_next": set(),
            "same_section": set(),
            "same_page": set(),
            "special_object_closure": set(),
        }
        for probe in probes
    }
    by_version_section = defaultdict(set)
    by_version = defaultdict(list)
    for chunk in chunks:
        version_key = str(chunk.document_version_id)
        by_version[version_key].append(chunk)
        section = str(chunk.section_path or "").strip()
        if section:
            by_version_section[(version_key, section)].add(str(chunk.id))
    for probe in probes:
        probe_id = str(probe.id)
        for neighbor_id in (
            str(probe.previous_chunk_id or ""),
            str(probe.next_chunk_id or ""),
        ):
            if neighbor_id in chunk_ids and neighbor_id != probe_id:
                targets[probe_id]["previous_next"].add(neighbor_id)
        section = str(probe.section_path or "").strip()
        if section:
            targets[probe_id]["same_section"].update(
                by_version_section.get(
                    (str(probe.document_version_id), section), set()
                )
                - {probe_id}
            )
        if probe.page_start is not None and probe.page_end is not None:
            probe_start = int(probe.page_start)
            probe_end = int(probe.page_end)
            for candidate in by_version[str(probe.document_version_id)]:
                if (
                    str(candidate.id) == probe_id
                    or candidate.page_start is None
                    or candidate.page_end is None
                ):
                    continue
                if max(probe_start, int(candidate.page_start)) <= min(
                    probe_end, int(candidate.page_end)
                ):
                    targets[probe_id]["same_page"].add(str(candidate.id))

    mappings = list(
        db.scalars(
            select(ChunkStructureMapping).where(
                ChunkStructureMapping.chunk_id.in_(sorted(chunk_ids))
            )
        ).all()
    )
    node_ids = {str(row.structure_node_id) for row in mappings}
    nodes = list(
        db.scalars(
            select(ChunkStructureNode).where(
                ChunkStructureNode.id.in_(sorted(node_ids))
            )
        ).all()
    )
    node_by_id = {str(node.id): node for node in nodes}
    chunk_ids_by_node = defaultdict(set)
    node_ids_by_chunk = defaultdict(set)
    for mapping in mappings:
        chunk_id = str(mapping.chunk_id)
        node_id = str(mapping.structure_node_id)
        if chunk_id in chunk_ids and node_id in node_by_id:
            chunk_ids_by_node[node_id].add(chunk_id)
            node_ids_by_chunk[chunk_id].add(node_id)

    special_types = {"table", "formula", "caption", "code_block"}
    special_node_ids = {
        node_id
        for node_id, node in node_by_id.items()
        if str(node.node_type) in special_types
    }
    closure_neighbors = defaultdict(set)
    structure_edges = list(
        db.scalars(
            select(ChunkStructureEdge).where(
                ChunkStructureEdge.source_node_id.in_(sorted(node_ids)),
                ChunkStructureEdge.target_node_id.in_(sorted(node_ids)),
                ChunkStructureEdge.edge_type.in_(
                    ("prev_next", "table_formula_context", "same_page_region")
                ),
            )
        ).all()
    )
    for edge in structure_edges:
        source_id = str(edge.source_node_id)
        target_id = str(edge.target_node_id)
        if source_id in special_node_ids or target_id in special_node_ids:
            closure_neighbors[source_id].add(target_id)
            closure_neighbors[target_id].add(source_id)
    for probe in probes:
        probe_id = str(probe.id)
        for node_id in node_ids_by_chunk.get(probe_id, set()):
            if node_id in special_node_ids:
                targets[probe_id]["special_object_closure"].update(
                    chunk_ids_by_node.get(node_id, set()) - {probe_id}
                )
            for neighbor_node_id in closure_neighbors.get(node_id, set()):
                targets[probe_id]["special_object_closure"].update(
                    chunk_ids_by_node.get(neighbor_node_id, set()) - {probe_id}
                )

    truncated_by_category = Counter()
    for category_card in targets.values():
        for category, target_ids in category_card.items():
            ordered_target_ids = sorted(
                target_ids,
                key=lambda target_id: chunk_business_keys[target_id],
            )
            truncated_by_category[category] += max(
                0,
                len(ordered_target_ids)
                - auto_tpe.TPE_STRUCTURE_POSITIVE_PER_CATEGORY_LIMIT,
            )
            category_card[category] = set(
                ordered_target_ids[
                    : auto_tpe.TPE_STRUCTURE_POSITIVE_PER_CATEGORY_LIMIT
                ]
            )

    spans = list(
        db.scalars(
            select(ChunkSpan).where(ChunkSpan.chunk_id.in_(sorted(chunk_ids)))
        ).all()
    )
    raw_span_traceable_ids = set()
    for row in spans:
        chunk = chunk_by_id.get(str(row.chunk_id))
        if (
            chunk is not None
            and str(row.document_version_id) == str(chunk.document_version_id)
            and int(row.char_start) <= int(chunk.char_start)
            and int(row.char_end) >= int(chunk.char_end)
            and int(row.char_end) > int(row.char_start)
        ):
            raw_span_traceable_ids.add(str(chunk.id))
    structure_traceable_ids = raw_span_traceable_ids.intersection(
        node_ids_by_chunk
    )
    categories = (
        "previous_next",
        "same_section",
        "same_page",
        "special_object_closure",
    )
    category_counts = {
        category: sum(len(card[category]) for card in targets.values())
        for category in categories
    }
    diagnostics = {
        "protocol_version": auto_tpe.TPE_STRUCTURE_RECOVERY_PROTOCOL_VERSION,
        "category_positive_counts": category_counts,
        "positive_count": sum(category_counts.values()),
        "mapped_chunk_count": len(node_ids_by_chunk),
        "raw_span_traceable_chunk_count": len(raw_span_traceable_ids),
        "structure_traceable_chunk_count": len(structure_traceable_ids),
        "special_node_count": len(special_node_ids),
        "special_closure_edge_count": len(structure_edges),
        "positive_limit_per_probe_category": (
            auto_tpe.TPE_STRUCTURE_POSITIVE_PER_CATEGORY_LIMIT
        ),
        "truncated_positive_counts": dict(sorted(truncated_by_category.items())),
        "model_call_count": 0,
    }
    return targets, structure_traceable_ids, diagnostics


def test_structure_positive_bounded_replays_reference_without_orm_identity_load(
    db_session,
    sample_knowledge_base,
):
    from app.models import (
        Chunk,
        ChunkSpan,
        ChunkStructureEdge,
        ChunkStructureMapping,
        ChunkStructureNode,
        Document,
        DocumentVersion,
    )
    from app.services import auto_tpe

    document = Document(
        knowledge_base_id=sample_knowledge_base.id,
        title="Structure positive reference",
        source_path="structure-positive-reference.md",
        source_type="markdown",
        checksum="b" * 64,
        is_active=True,
    )
    db_session.add(document)
    db_session.flush()
    version = DocumentVersion(
        document_id=document.id,
        version=1,
        checksum="b" * 64,
        storage_path="structure-positive-reference.md",
        is_active=True,
    )
    db_session.add(version)
    db_session.flush()
    chunks = [
        Chunk(
            knowledge_base_id=sample_knowledge_base.id,
            document_id=document.id,
            document_version_id=version.id,
            chunk_version=1,
            chunk_index=index,
            token_start=index * 10,
            token_end=index * 10 + 5,
            char_start=index * 100,
            char_end=index * 100 + 50,
            text=f"reference chunk {index}",
            text_hash=f"{index + 1:064x}",
            section_path=("shared" if index < 2 else f"section-{index}"),
            page_start=(1 if index < 3 else 2),
            page_end=(1 if index < 3 else 2),
            rq_path=[],
            metadata_json={},
            state="active",
        )
        for index in range(4)
    ]
    db_session.add_all(chunks)
    db_session.flush()
    chunks[0].next_chunk_id = chunks[1].id
    chunks[1].previous_chunk_id = chunks[0].id
    nodes = {
        "table": ChunkStructureNode(
            knowledge_base_id=sample_knowledge_base.id,
            document_id=document.id,
            document_version_id=version.id,
            node_type="table",
            depth=1,
            title="table",
            path="reference > table",
        ),
        "formula": ChunkStructureNode(
            knowledge_base_id=sample_knowledge_base.id,
            document_id=document.id,
            document_version_id=version.id,
            node_type="formula",
            depth=1,
            title="formula",
            path="reference > formula",
        ),
        "paragraph": ChunkStructureNode(
            knowledge_base_id=sample_knowledge_base.id,
            document_id=document.id,
            document_version_id=version.id,
            node_type="paragraph",
            depth=1,
            title="paragraph",
            path="reference > paragraph",
        ),
        "other": ChunkStructureNode(
            knowledge_base_id=sample_knowledge_base.id,
            document_id=document.id,
            document_version_id=version.id,
            node_type="paragraph",
            depth=1,
            title="other",
            path="reference > other",
        ),
    }
    db_session.add_all(nodes.values())
    db_session.flush()
    db_session.add_all(
        [
            ChunkStructureMapping(
                chunk_id=chunks[0].id,
                structure_node_id=nodes["table"].id,
                document_version_id=version.id,
            ),
            ChunkStructureMapping(
                chunk_id=chunks[0].id,
                structure_node_id=nodes["paragraph"].id,
                document_version_id=version.id,
            ),
            ChunkStructureMapping(
                chunk_id=chunks[1].id,
                structure_node_id=nodes["table"].id,
                document_version_id=version.id,
            ),
            ChunkStructureMapping(
                chunk_id=chunks[2].id,
                structure_node_id=nodes["formula"].id,
                document_version_id=version.id,
            ),
            ChunkStructureMapping(
                chunk_id=chunks[3].id,
                structure_node_id=nodes["other"].id,
                document_version_id=version.id,
            ),
        ]
    )
    db_session.add_all(
        [
            ChunkSpan(
                chunk_id=chunk.id,
                document_version_id=version.id,
                char_start=chunk.char_start,
                char_end=chunk.char_end,
                token_start=chunk.token_start,
                token_end=chunk.token_end,
            )
            for chunk in chunks
        ]
    )
    db_session.add_all(
        [
            ChunkStructureEdge(
                knowledge_base_id=sample_knowledge_base.id,
                document_version_id=version.id,
                source_node_id=nodes["paragraph"].id,
                target_node_id=nodes["formula"].id,
                edge_type="table_formula_context",
            ),
            ChunkStructureEdge(
                knowledge_base_id=sample_knowledge_base.id,
                document_version_id=version.id,
                source_node_id=nodes["table"].id,
                target_node_id=nodes["other"].id,
                edge_type="same_page_region",
            ),
            ChunkStructureEdge(
                knowledge_base_id=sample_knowledge_base.id,
                document_version_id=version.id,
                source_node_id=nodes["paragraph"].id,
                target_node_id=nodes["other"].id,
                edge_type="prev_next",
            ),
            ChunkStructureEdge(
                knowledge_base_id=sample_knowledge_base.id,
                document_version_id=version.id,
                source_node_id=nodes["table"].id,
                target_node_id=nodes["formula"].id,
                edge_type="hierarchy",
            ),
        ]
    )
    db_session.commit()
    knowledge_base_id = sample_knowledge_base.id
    chunks = list(
        db_session.scalars(
            select(Chunk)
            .where(
                Chunk.knowledge_base_id == knowledge_base_id,
                Chunk.state == "active",
            )
            .order_by(Chunk.chunk_index.asc())
        ).all()
    )
    probe_ids = {str(chunk.id) for chunk in chunks[::2]}
    chunk_business_keys = {
        str(chunk.id): f"fixture-chunk-{index:04d}"
        for index, chunk in enumerate(chunks)
    }
    reference = _reference_structure_positive_context(
        db_session,
        chunks,
        [chunk for chunk in chunks if str(chunk.id) in probe_ids],
        chunk_business_keys=chunk_business_keys,
    )
    db_session.expunge_all()
    chunks = list(
        db_session.scalars(
            select(Chunk)
            .where(
                Chunk.knowledge_base_id == knowledge_base_id,
                Chunk.state == "active",
            )
            .order_by(Chunk.chunk_index.asc())
        ).all()
    )
    probes = [chunk for chunk in chunks if str(chunk.id) in probe_ids]

    actual_targets, actual_traceable, actual_diagnostics = (
        auto_tpe._structure_positive_context(
            db_session,
            chunks,
            probes,
            chunk_business_keys=chunk_business_keys,
        )
    )

    reference_targets, reference_traceable, reference_diagnostics = reference
    assert actual_targets == reference_targets
    assert actual_traceable == reference_traceable
    for key, expected in reference_diagnostics.items():
        assert actual_diagnostics[key] == expected
    assert actual_diagnostics["bounded_structure_query_rows"]
    assert not any(
        isinstance(value, (ChunkStructureMapping, ChunkStructureNode))
        for value in db_session.identity_map.values()
    )


def test_structure_positive_large_repeated_labels_has_bounded_rows_and_exact_limit(
    db_session,
    sample_knowledge_base,
):
    from app.models import (
        Chunk,
        ChunkStructureMapping,
        ChunkStructureNode,
        Document,
        DocumentVersion,
    )
    from app.services import auto_tpe

    document = Document(
        knowledge_base_id=sample_knowledge_base.id,
        title="Structure positive scale",
        source_path="structure-positive-scale.md",
        source_type="markdown",
        checksum="a" * 64,
        is_active=True,
    )
    db_session.add(document)
    db_session.flush()
    version = DocumentVersion(
        document_id=document.id,
        version=1,
        checksum="a" * 64,
        storage_path="structure-positive-scale.md",
        is_active=True,
    )
    db_session.add(version)
    db_session.flush()

    target_count = auto_tpe.TPE_STRUCTURE_POSITIVE_PER_CATEGORY_LIMIT + 17
    chunk_ids = [f"scale-chunk-{index:024d}" for index in range(target_count + 1)]
    db_session.execute(
        insert(Chunk),
        [
            {
                "id": chunk_id,
                "knowledge_base_id": sample_knowledge_base.id,
                "document_id": document.id,
                "document_version_id": version.id,
                "chunk_version": 1,
                "chunk_index": index,
                "token_start": index,
                "token_end": index + 1,
                "char_start": index * 2,
                "char_end": index * 2 + 1,
                "text": f"chunk {index}",
                "text_hash": f"{index:064x}",
                "section_path": f"unique-section-{index}",
                "page_start": None,
                "page_end": None,
                "rq_path": [],
                "metadata_json": {},
                "state": "active",
            }
            for index, chunk_id in enumerate(chunk_ids)
        ],
    )
    special_node_id = "scale-special-table-node"
    repeated_node_count = 2_000
    repeated_node_ids = [
        f"scale-repeat-node-{index:016d}" for index in range(repeated_node_count)
    ]
    node_rows = [
        {
            "id": special_node_id,
            "knowledge_base_id": sample_knowledge_base.id,
            "document_id": document.id,
            "document_version_id": version.id,
            "node_type": "table",
            "depth": 1,
            "title": "special",
            "bbox_json": {},
            "layout_json": {},
            "path": "Repeated section",
        }
    ]
    node_rows.extend(
        {
            "id": node_id,
            "knowledge_base_id": sample_knowledge_base.id,
            "document_id": document.id,
            "document_version_id": version.id,
            "node_type": "paragraph",
            "depth": 2,
            "title": "repeated",
            "bbox_json": {},
            "layout_json": {},
            "path": "Repeated section > repeated paragraph",
        }
        for node_id in repeated_node_ids
    )
    db_session.execute(insert(ChunkStructureNode), node_rows)
    mapping_rows = [
        {
            "chunk_id": chunk_id,
            "structure_node_id": special_node_id,
            "document_version_id": version.id,
        }
        for chunk_id in chunk_ids
    ]
    mapping_rows.extend(
        {
            "chunk_id": chunk_ids[1],
            "structure_node_id": node_id,
            "document_version_id": version.id,
        }
        for node_id in repeated_node_ids
    )
    db_session.execute(insert(ChunkStructureMapping), mapping_rows)
    db_session.commit()
    db_session.expunge_all()

    chunks = list(
        db_session.scalars(
            select(Chunk)
            .where(Chunk.id.in_(chunk_ids))
            .order_by(Chunk.chunk_index.asc())
        ).all()
    )
    probe = chunks[0]
    chunk_business_keys = {
        str(chunk.id): f"scale-business-{index:08d}"
        for index, chunk in enumerate(chunks)
    }
    targets, traceable_ids, diagnostics = auto_tpe._structure_positive_context(
        db_session,
        chunks,
        [probe],
        chunk_business_keys=chunk_business_keys,
    )

    expected_ids = set(chunk_ids[1 : 1 + auto_tpe.TPE_STRUCTURE_POSITIVE_PER_CATEGORY_LIMIT])
    assert targets[str(probe.id)]["special_object_closure"] == expected_ids
    assert diagnostics["truncated_positive_counts"] == {
        "previous_next": 0,
        "same_page": 0,
        "same_section": 0,
        "special_object_closure": 17,
    }
    assert diagnostics["mapped_chunk_count"] == target_count + 1
    assert diagnostics["special_node_count"] == 1
    assert diagnostics["positive_limit_per_probe_category"] == 512
    assert not traceable_ids
    query_rows = diagnostics["bounded_structure_query_rows"]
    assert query_rows == {
        "probe_mapping_rows": 1,
        "probe_node_type_rows": 1,
        "incident_special_edge_rows": 0,
        "relevant_node_mapping_rows": target_count + 1,
        "raw_span_traceable_rows": 0,
        "structure_traceable_rows": 0,
    }
    total_mapping_count = int(
        db_session.scalar(select(func.count(ChunkStructureMapping.id))) or 0
    )
    assert total_mapping_count == target_count + 1 + repeated_node_count
    assert sum(query_rows.values()) < total_mapping_count
    assert not any(
        isinstance(value, (ChunkStructureMapping, ChunkStructureNode))
        for value in db_session.identity_map.values()
    )
