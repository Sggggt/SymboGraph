"""Prepare fixed evaluation and first-route vectors in one bounded call."""
from __future__ import annotations

from typing import Literal

from pydantic import Field, model_validator

from app.retrieval_control_contracts import ControlContract
from app.services.context_graph import (
    QueryEmbeddingRequestMemo, query_embedding_request_memo_key,
    semantic_entry_query_for_search,
)
from app.services.embeddings import validate_embedding_vectors
from app.services.retrieval_models import routing_facets


class QueryVectorBatchAudit(ControlContract):
    protocol_version: Literal["retrieval_query_vector_batch_v1"] = "retrieval_query_vector_batch_v1"
    input_count: int = Field(ge=3, le=10)
    unique_text_count: int = Field(ge=1, le=10)
    measurement_indices: tuple[int, ...] = Field(min_length=2, max_length=9)
    routing_index: int = Field(ge=0, le=9)
    provider_service_call_count: Literal[1] = 1
    request_memo_seeded: Literal[True] = True

    @model_validator(mode="after")
    def validate_projection(self):
        indices = (*self.measurement_indices, self.routing_index)
        if (len(indices) != self.input_count or self.unique_text_count > self.input_count
                or set(indices) != set(range(self.unique_text_count))):
            raise ValueError("retrieval_query_vector_projection_invalid")
        return self


async def prepare_query_vectors(*, task, strategy, initial_facets, target, provider):
    facets = routing_facets(task, strategy, initial_facets)
    semantic = semantic_entry_query_for_search(task.question, facets)
    texts = ["; ".join(item.text for item in task.requirements),
             *[item.text for item in task.requirements], semantic["query"]]
    unique = list(dict.fromkeys(texts))
    positions = {text: index for index, text in enumerate(unique)}
    indices = tuple(positions[text] for text in texts)
    returned = await provider.embed_texts(unique, text_type="query")
    validate_embedding_vectors(returned, expected_count=len(unique),
                               expected_dimensions=target.schema.embedding_dimension)
    # Publish only after the whole provider result passes validation. Existing
    # provider batching/concurrency and caller timeout/cancellation still apply.
    memo = QueryEmbeddingRequestMemo()
    memo.put(query_embedding_request_memo_key(task.knowledge_base_id, semantic, target),
             returned[indices[-1]])
    vectors = [list(returned[index]) for index in indices[:-1]]
    audit = QueryVectorBatchAudit(input_count=len(texts), unique_text_count=len(unique),
        measurement_indices=indices[:-1], routing_index=indices[-1])
    return vectors, memo, audit
