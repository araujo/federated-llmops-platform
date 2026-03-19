"""FastAPI dependencies: config, Langfuse callback, embeddings, DB pool."""

from functools import lru_cache

import asyncpg
from fastapi import Depends, Header, HTTPException
from langchain_openai import OpenAIEmbeddings
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings from environment."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # API auth - when set, protected endpoints require X-API-Key header
    api_key: str = ""

    # Rate limit - max requests per minute per client (0 = disabled)
    rate_limit_per_minute: int = 0

    # LiteLLM - all model calls go through here (never Ollama directly)
    litellm_base_url: str = "http://litellm:4000/v1"
    litellm_api_key: str = "sk-1234"
    litellm_model: str = "ollama/llama3.2"
    litellm_embedding_model: str = "ollama/nomic-embed-text"

    # Langfuse
    langfuse_public_key: str = ""
    langfuse_secret_key: str = ""
    langfuse_host: str = "http://langfuse-web:3000"

    # Health check targets (from POSTGRES_*, MINIO_*)
    postgres_host: str = "postgres"
    postgres_port: int = 5432
    postgres_user: str = "llmops"
    postgres_password: str = "llmops"
    postgres_db: str = "llmops"
    minio_host: str = "minio"
    minio_port: int = 9000
    minio_user: str = "minio"
    minio_password: str = "miniosecret"
    minio_bucket: str = "documents"


@lru_cache
def get_settings() -> Settings:
    return Settings()


def get_embeddings(settings: Settings) -> OpenAIEmbeddings:
    """Create embeddings client via LiteLLM."""
    return OpenAIEmbeddings(
        model=settings.litellm_embedding_model,
        openai_api_base=settings.litellm_base_url,
        openai_api_key=settings.litellm_api_key,
    )


async def verify_api_key(
    x_api_key: str | None = Header(None, alias="X-API-Key"),
    settings: Settings = Depends(get_settings),
) -> None:
    """Require valid API key when API_KEY env is set. Public endpoints skip this."""
    if not settings.api_key:
        return  # Auth disabled
    if not x_api_key or x_api_key != settings.api_key:
        raise HTTPException(401, "Invalid or missing API key")


# Postgres pool - created at startup, closed at shutdown
_pool: asyncpg.Pool | None = None


async def get_pool() -> asyncpg.Pool:
    """Get Postgres connection pool. Raises if not initialized."""
    if _pool is None:
        raise RuntimeError("Postgres pool not initialized")
    return _pool


async def init_pool(settings: Settings) -> None:
    """Create Postgres connection pool."""
    global _pool
    _pool = await asyncpg.create_pool(
        host=settings.postgres_host,
        port=settings.postgres_port,
        user=settings.postgres_user,
        password=settings.postgres_password,
        database=settings.postgres_db,
        min_size=1,
        max_size=5,
    )


async def close_pool() -> None:
    """Close Postgres connection pool."""
    global _pool
    if _pool:
        await _pool.close()
        _pool = None
