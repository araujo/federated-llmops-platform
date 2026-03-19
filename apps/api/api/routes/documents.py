"""Document upload and management."""

import uuid

import asyncpg
from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile
from langfuse import observe
from minio import Minio
from pydantic import BaseModel

from api.dependencies import Settings, get_embeddings, get_pool, get_settings, verify_api_key
from api.ingestion import ALLOWED_EXTENSIONS, ingest_document
from retrieval import search_chunks_with_similarity

router = APIRouter(prefix="/documents", tags=["documents"])


class DocumentResponse(BaseModel):
    """Response after document upload."""

    document_id: str
    filename: str
    chunks_count: int


class DocumentItem(BaseModel):
    """Document list item."""

    document_id: str
    filename: str
    content_type: str
    chunks_count: int
    created_at: str


class SearchResultItem(BaseModel):
    """Search result item."""

    content: str
    document_id: str
    similarity: float


def _get_minio_client(settings: Settings) -> Minio:
    """Create MinIO client."""
    return Minio(
        endpoint=f"{settings.minio_host}:{settings.minio_port}",
        access_key=settings.minio_user,
        secret_key=settings.minio_password,
        secure=False,
    )


@router.post("/upload", response_model=DocumentResponse)
@observe()
async def upload_document(
    file: UploadFile = File(...),
    settings: Settings = Depends(get_settings),
    pool: asyncpg.Pool = Depends(get_pool),
    _: None = Depends(verify_api_key),
) -> DocumentResponse:
    """Upload a .txt or .md document. Ingests to MinIO, chunks, embeds, stores."""
    if not file.filename:
        raise HTTPException(400, "Missing filename")

    ext = "." + file.filename.rsplit(".", 1)[-1].lower() if "." in file.filename else ""
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            400,
            f"Only .txt and .md files are allowed. Got: {ext or 'no extension'}",
        )

    content_type = file.content_type or "text/plain"
    if content_type not in ("text/plain", "text/markdown"):
        content_type = "text/plain"

    content = await file.read()
    if not content:
        raise HTTPException(400, "Empty file")

    minio_client = _get_minio_client(settings)
    embeddings = get_embeddings(settings)

    try:
        doc_id = await ingest_document(
            pool=pool,
            embeddings=embeddings,
            minio_client=minio_client,
            settings=settings,
            filename=file.filename,
            content_type=content_type,
            content=content,
        )
    except ValueError as e:
        raise HTTPException(400, str(e))

    # Count chunks
    async with pool.acquire() as conn:
        count = await conn.fetchval(
            "SELECT COUNT(*) FROM chunks WHERE document_id = $1",
            doc_id,
        )

    return DocumentResponse(
        document_id=str(doc_id),
        filename=file.filename,
        chunks_count=count or 0,
    )


@router.get("/search", response_model=list[SearchResultItem])
@observe()
async def search_documents(
    q: str = Query(..., min_length=1, description="Search query"),
    top_k: int = Query(5, ge=1, le=50, description="Number of results"),
    settings: Settings = Depends(get_settings),
    pool: asyncpg.Pool = Depends(get_pool),
    _: None = Depends(verify_api_key),
) -> list[SearchResultItem]:
    """Search documents by semantic similarity over stored embeddings."""
    embeddings = get_embeddings(settings)
    query_embedding = await embeddings.aembed_query(q)
    chunks = await search_chunks_with_similarity(pool, query_embedding, top_k=top_k)
    return [
        SearchResultItem(
            content=c["content"],
            document_id=str(c["document_id"]),
            similarity=c["similarity"],
        )
        for c in chunks
    ]


@router.get("", response_model=list[DocumentItem])
async def list_documents(
    pool: asyncpg.Pool = Depends(get_pool),
    _: None = Depends(verify_api_key),
) -> list[DocumentItem]:
    """List all ingested documents."""
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT d.id, d.filename, d.content_type, d.created_at,
                   (SELECT COUNT(*) FROM chunks c WHERE c.document_id = d.id) AS chunks_count
            FROM documents d
            ORDER BY d.created_at DESC
            """
        )
    return [
        DocumentItem(
            document_id=str(r["id"]),
            filename=r["filename"],
            content_type=r["content_type"],
            chunks_count=r["chunks_count"],
            created_at=r["created_at"].isoformat() if r["created_at"] else "",
        )
        for r in rows
    ]


@router.delete("/{document_id}")
async def delete_document(
    document_id: str,
    settings: Settings = Depends(get_settings),
    pool: asyncpg.Pool = Depends(get_pool),
    _: None = Depends(verify_api_key),
) -> dict:
    """Delete a document and its chunks. Removes from Postgres and MinIO."""
    try:
        doc_uuid = uuid.UUID(document_id)
    except ValueError:
        raise HTTPException(400, "Invalid document_id")

    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT id, minio_key FROM documents WHERE id = $1",
            doc_uuid,
        )
        if not row:
            raise HTTPException(404, "Document not found")

        await conn.execute("DELETE FROM documents WHERE id = $1", doc_uuid)

    minio_client = _get_minio_client(settings)
    try:
        minio_client.remove_object(settings.minio_bucket, row["minio_key"])
    except Exception:
        pass  # Best-effort; document already removed from DB

    return {"status": "deleted", "document_id": document_id}
