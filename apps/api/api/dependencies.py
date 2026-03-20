"""FastAPI dependencies: config, Langfuse callback, embeddings, DB pool."""

import logging
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

    # MongoDB - prompt registry (use memory:// or empty for in-memory in tests)
    mongodb_uri: str = "mongodb://root:mongosecret@mongodb:27017?authSource=admin"
    mongodb_database: str = "llmops"


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


# MongoDB client for prompt registry
_mongo_client = None
_mongo_database = ""


def get_mongo_db():
    """Get MongoDB database for prompts. Raises if not initialized."""
    if _mongo_client is None:
        raise RuntimeError("MongoDB client not initialized")
    return _mongo_client[_mongo_database]


def init_mongo(settings: Settings) -> bool:
    """Create MongoDB client for prompt registry.

    Returns True if MongoDB is used, False if in-memory fallback.
    Startup does not break if Mongo is unreachable; falls back to in-memory.
    """
    global _mongo_client, _mongo_database
    uri = (settings.mongodb_uri or "").strip()
    if not uri or uri == "memory://":
        _mongo_client = None
        _mongo_database = ""
        return False
    try:
        from pymongo import MongoClient

        client = MongoClient(
            uri,
            serverSelectionTimeoutMS=5000,
        )
        client.admin.command("ping")
        _mongo_client = client
        _mongo_database = settings.mongodb_database
        return True
    except Exception as e:
        logging.getLogger(__name__).warning(
            "MongoDB unreachable, using in-memory prompt registry: %s", e
        )
        _mongo_client = None
        _mongo_database = ""
        return False


def close_mongo() -> None:
    """Close MongoDB client."""
    global _mongo_client
    if _mongo_client:
        _mongo_client.close()
        _mongo_client = None
