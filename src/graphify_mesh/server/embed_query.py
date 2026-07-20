"""Query-time embedding wrapper (WS5): turns a raw query string into a
vector for `retrieval.vector_candidates`, reusing the exact `/api/embed`
contract and base_url/model config `graphify_mesh.sync.embedding` /
`graphify_mesh.sync.config` already established for the sync pipeline (C9) — the
query MUST be embedded with the same model/endpoint that produced the
published vectors, or cosine similarity is meaningless.

Never raises: any transport/shape failure degrades to `None` (the caller,
`retrieval.vector_candidates`, surfaces `"embeddings_unavailable"` in the
response's `degraded` field), mirroring the pipeline's own "never crash on a
down local service" convention (C9) — just applied at query time instead of
build time.
"""

from __future__ import annotations

import logging
import os

from graphify_mesh.sync.config import EMBED_DEFAULT_BASE_URL, EMBED_DEFAULT_MODEL
from graphify_mesh.sync.embedding import embed_batch

log = logging.getLogger("graphify_mesh.server.embed_query")

# Kept short: a query-time embed call blocks one MCP `search`/`context_pack`
# invocation, unlike the sync pipeline's bulk embed calls which can afford a
# longer per-batch timeout.
QUERY_EMBED_TIMEOUT = 10.0


def make_embed_query_fn(
    base_url: str | None = None, model: str | None = None, timeout: float = QUERY_EMBED_TIMEOUT
):
    """Returns a `str -> list[float] | None` callable matching
    `retrieval.EmbedQueryFn`, bound to the resolved base_url/model (env
    override via the SAME `GRAPHIFY_MESH_OLLAMA_EMBED_*` variables the sync
    pipeline reads, so a query never drifts onto a different embedding
    space than the one the published vectors were built with)."""
    resolved_base_url = base_url or os.environ.get(
        "GRAPHIFY_MESH_OLLAMA_EMBED_BASE_URL", EMBED_DEFAULT_BASE_URL
    )
    resolved_model = model or os.environ.get(
        "GRAPHIFY_MESH_OLLAMA_EMBED_MODEL", EMBED_DEFAULT_MODEL
    )

    def embed_query(query: str) -> list[float] | None:
        if not query or not query.strip():
            return None
        try:
            vectors = embed_batch(resolved_base_url, resolved_model, [query], timeout=timeout)
        except RuntimeError as exc:
            log.warning(
                "graphify-mesh: query embedding unavailable, "
                "degrading to lexical+structural only: %s",
                exc,
            )
            return None
        if not vectors:
            return None
        return vectors[0]

    return embed_query
