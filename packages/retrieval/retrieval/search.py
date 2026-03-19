"""Vector similarity search over chunks using pgvector."""

import asyncpg


def _safe_metadata(m) -> dict:
    if m is None:
        return {}
    if isinstance(m, dict):
        return m
    return {}


async def search_chunks(
    pool: asyncpg.Pool,
    query_embedding: list[float],
    top_k: int = 5,
) -> list[dict]:
    """Retrieve top-k chunks by cosine similarity.

    Args:
        pool: asyncpg connection pool.
        query_embedding: Query vector (768 dims for nomic-embed-text).
        top_k: Number of chunks to return.

    Returns:
        List of dicts with keys: id, document_id, content, chunk_index, metadata.
    """
    vec = "[" + ",".join(str(x) for x in query_embedding) + "]"
    rows = await pool.fetch(
        """
        SELECT id, document_id, content, chunk_index, metadata
        FROM chunks
        ORDER BY embedding <=> $1::vector
        LIMIT $2
        """,
        vec,
        top_k,
    )
    return [
        {
            "id": str(r["id"]),
            "document_id": str(r["document_id"]),
            "content": r["content"],
            "chunk_index": r["chunk_index"],
            "metadata": _safe_metadata(r["metadata"]),
        }
        for r in rows
    ]


async def search_chunks_with_similarity(
    pool: asyncpg.Pool,
    query_embedding: list[float],
    top_k: int = 5,
) -> list[dict]:
    """Retrieve top-k chunks by cosine similarity with similarity scores.

    Args:
        pool: asyncpg connection pool.
        query_embedding: Query vector (768 dims for nomic-embed-text).
        top_k: Number of chunks to return.

    Returns:
        List of dicts with keys: id, document_id, content, chunk_index, metadata, similarity.
        similarity is 1 - cosine_distance (higher = more similar).
    """
    vec = "[" + ",".join(str(x) for x in query_embedding) + "]"
    rows = await pool.fetch(
        """
        SELECT id, document_id, content, chunk_index, metadata,
               (1 - (embedding <=> $1::vector))::float AS similarity
        FROM chunks
        ORDER BY embedding <=> $1::vector
        LIMIT $2
        """,
        vec,
        top_k,
    )
    return [
        {
            "id": str(r["id"]),
            "document_id": str(r["document_id"]),
            "content": r["content"],
            "chunk_index": r["chunk_index"],
            "metadata": _safe_metadata(r["metadata"]),
            "similarity": round(float(r["similarity"]), 6),
        }
        for r in rows
    ]
