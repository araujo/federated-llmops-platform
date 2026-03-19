"""Document ingestion: chunk, embed, store in Postgres and MinIO."""

import uuid
from io import BytesIO

import asyncpg
from langchain_openai import OpenAIEmbeddings
from langfuse import observe
from minio import Minio

from api.dependencies import Settings

# Chunk size and overlap (chars)
CHUNK_SIZE = 800
CHUNK_OVERLAP = 100

ALLOWED_EXTENSIONS = {".txt", ".md"}


def chunk_text(text: str) -> list[str]:
    """Split text into overlapping chunks."""
    chunks: list[str] = []
    start = 0
    text = text.strip()
    if not text:
        return []

    while start < len(text):
        end = start + CHUNK_SIZE
        chunk = text[start:end]
        # Prefer breaking at paragraph or sentence boundary
        if end < len(text):
            last_newline = chunk.rfind("\n")
            last_period = chunk.rfind(". ")
            break_at = max(last_newline, last_period)
            if break_at > CHUNK_SIZE // 2:
                chunk = chunk[: break_at + 1]
                end = start + len(chunk)
        chunks.append(chunk.strip())
        start = end - CHUNK_OVERLAP
        if start >= len(text):
            break

    return [c for c in chunks if c]


@observe()
async def ingest_document(
    *,
    pool: asyncpg.Pool,
    embeddings: OpenAIEmbeddings,
    minio_client: Minio,
    settings: Settings,
    filename: str,
    content_type: str,
    content: bytes,
) -> uuid.UUID:
    """Upload to MinIO, chunk, embed, store in documents and chunks.

    Returns:
        document_id (UUID)
    """
    text = content.decode("utf-8", errors="replace")
    chunks_text = chunk_text(text)
    if not chunks_text:
        raise ValueError("Document has no content after chunking")

    # Generate MinIO key
    doc_id = uuid.uuid4()
    minio_key = f"{doc_id}/{filename}"

    # Upload to MinIO
    minio_client.put_object(
        settings.minio_bucket,
        minio_key,
        BytesIO(content),
        len(content),
        content_type=content_type,
    )

    # Embed chunks
    embeddings_list = await embeddings.aembed_documents(chunks_text)

    # Insert document
    async with pool.acquire() as conn:
        await conn.execute(
            """
            INSERT INTO documents (id, filename, content_type, minio_key)
            VALUES ($1, $2, $3, $4)
            """,
            doc_id,
            filename,
            content_type,
            minio_key,
        )

        # Insert chunks
        for i, (chunk_content, emb) in enumerate(zip(chunks_text, embeddings_list)):
            vec = "[" + ",".join(str(x) for x in emb) + "]"
            await conn.execute(
                """
                INSERT INTO chunks (document_id, content, embedding, chunk_index, metadata)
                VALUES ($1, $2, $3::vector, $4, $5)
                """,
                doc_id,
                chunk_content,
                vec,
                i,
                "{}",
            )

    return doc_id
