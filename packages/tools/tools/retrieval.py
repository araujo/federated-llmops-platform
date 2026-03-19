"""Retrieval tool - semantic search over document chunks."""

from typing import Any

import asyncpg

from retrieval import search_chunks_with_similarity


async def retrieval_tool(
    query_embedding: list[float],
    pool: asyncpg.Pool,
    top_k: int = 5,
) -> list[dict[str, Any]]:
    """Search document chunks by semantic similarity.

    Args:
        query_embedding: Query vector from embedding model.
        pool: asyncpg connection pool.
        top_k: Number of chunks to return.

    Returns:
        List of dicts with content, document_id, similarity.
    """
    chunks = await search_chunks_with_similarity(
        pool, query_embedding, top_k=top_k
    )
    return [
        {
            "content": c["content"],
            "document_id": c["document_id"],
            "similarity": c["similarity"],
        }
        for c in chunks
    ]
