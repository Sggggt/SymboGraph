"""Complete, request-local incident-edge reads for a frozen relation graph."""
from __future__ import annotations

from sqlalchemy import or_, select
from sqlalchemy.orm import defer

from app.models import ChunkRelationEdge
from app.services.qa_performance import qa_stage
from app.services.storage import raise_if_source_io_cancelled


class CompleteChunkAdjacency:
    protocol_version = "complete_incident_edges_v1"

    def __init__(self, db, *, graph_state_id, allowed_types):
        self.db = db
        self.graph_state_id = graph_state_id
        self.allowed_types = tuple(sorted(set(allowed_types)))
        self._complete = {}
        self.query_count = 0
        self.rows_read = 0

    def preload(self, node_ids):
        pending = sorted(set(node_ids) - self._complete.keys())
        for offset in range(0, len(pending), 128):
            raise_if_source_io_cancelled()
            batch = pending[offset:offset + 128]
            rows_by_node = {node_id: [] for node_id in batch}
            if self.graph_state_id and self.allowed_types:
                statement = select(ChunkRelationEdge).options(
                    defer(ChunkRelationEdge.raw_strength_summary_json, raiseload=True),
                    defer(ChunkRelationEdge.normalization_stats_json, raiseload=True),
                ).where(ChunkRelationEdge.graph_state_id == self.graph_state_id,
                    ChunkRelationEdge.edge_type.in_(self.allowed_types), or_(
                        ChunkRelationEdge.source_chunk_id.in_(batch),
                        ChunkRelationEdge.target_chunk_id.in_(batch)))
                with qa_stage("graph_edge_read") as timing:
                    count = 0
                    for edge in self.db.scalars(statement).yield_per(128):
                        if count % 128 == 0:
                            raise_if_source_io_cancelled()
                        count += 1
                        # Preserve the former bidirectional/self-loop behavior.
                        for node_id in (edge.source_chunk_id, edge.target_chunk_id):
                            if node_id in rows_by_node:
                                rows_by_node[node_id].append(edge)
                    if timing is not None:
                        timing.annotate(item_count=count)
                self.query_count += 1
                self.rows_read += count
            raise_if_source_io_cancelled()
            self._complete.update({key: tuple(value) for key, value in rows_by_node.items()})

    def get(self, node_id, default=()):
        raise_if_source_io_cancelled()
        if node_id not in self._complete:
            self.preload([node_id])
        return self._complete.get(node_id, default)

    def loaded_edges(self):
        """Return each request-loaded edge once, without claiming full graph scan."""

        by_id = {}
        for rows in self._complete.values():
            for edge in rows:
                by_id[str(edge.id)] = edge
        return tuple(by_id[key] for key in sorted(by_id))
