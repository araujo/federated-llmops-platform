"""Health check endpoint."""

import asyncpg
import httpx
from fastapi import APIRouter, Depends
from pydantic import BaseModel

from api.dependencies import Settings, get_settings

router = APIRouter(tags=["health"])


class HealthStatus(BaseModel):
    """Health check response."""

    status: str
    postgres: str
    minio: str
    litellm: str
    langfuse: str


async def _check_postgres(settings: Settings) -> bool:
    """Check Postgres connectivity."""
    try:
        conn = await asyncpg.connect(
            host=settings.postgres_host,
            port=settings.postgres_port,
            user=settings.postgres_user,
            password=settings.postgres_password,
            database=settings.postgres_db,
            timeout=3,
        )
        await conn.close()
        return True
    except Exception:
        return False


async def _check_minio(settings: Settings) -> bool:
    """Check MinIO connectivity."""
    try:
        async with httpx.AsyncClient(timeout=3.0) as client:
            resp = await client.get(
                f"http://{settings.minio_host}:{settings.minio_port}/minio/health/live"
            )
            return resp.status_code == 200
    except Exception:
        return False


async def _check_litellm(settings: Settings) -> bool:
    """Check LiteLLM health endpoint."""
    try:
        base = settings.litellm_base_url.replace("/v1", "")
        async with httpx.AsyncClient(timeout=3.0) as client:
            resp = await client.get(f"{base}/health")
            return resp.status_code == 200
    except Exception:
        return False


async def _check_langfuse(settings: Settings) -> bool:
    """Check Langfuse health endpoint."""
    try:
        async with httpx.AsyncClient(timeout=3.0) as client:
            resp = await client.get(f"{settings.langfuse_host}/api/public/health")
            return resp.status_code == 200
    except Exception:
        return False


@router.get("/health", response_model=HealthStatus)
async def health(settings: Settings = Depends(get_settings)) -> HealthStatus:
    """Check health of critical dependencies."""
    postgres_ok = await _check_postgres(settings)
    minio_ok = await _check_minio(settings)
    litellm_ok = await _check_litellm(settings)
    langfuse_ok = await _check_langfuse(settings)

    all_ok = postgres_ok and minio_ok and litellm_ok and langfuse_ok
    status = "ok" if all_ok else "degraded"

    return HealthStatus(
        status=status,
        postgres="ok" if postgres_ok else "unavailable",
        minio="ok" if minio_ok else "unavailable",
        litellm="ok" if litellm_ok else "unavailable",
        langfuse="ok" if langfuse_ok else "unavailable",
    )
