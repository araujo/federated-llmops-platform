"""Chat orchestration - request handling with decision layer and execution."""

from collections.abc import AsyncIterator
from typing import Any

import asyncpg
from langfuse.langchain import CallbackHandler

from orchestration.decision import decide_strategy
from orchestration.execution import (
    execute_direct,
    execute_direct_stream,
    execute_rag,
    execute_rag_stream,
)
from retrieval import search_chunks_with_similarity


async def _fetch_chunks_for_decision(
    pool: asyncpg.Pool,
    message: str,
    *,
    base_url: str,
    api_key: str,
    embedding_model: str,
    top_k: int = 5,
) -> list[dict]:
    """Fetch chunks with similarity for strategy decision. Uses search_chunks_with_similarity."""
    from langchain_openai import OpenAIEmbeddings

    embeddings = OpenAIEmbeddings(
        model=embedding_model,
        openai_api_base=base_url,
        openai_api_key=api_key,
    )
    query_embedding = await embeddings.aembed_query(message)
    return await search_chunks_with_similarity(pool, query_embedding, top_k=top_k)


# --- Public API (unchanged signatures for API routes) ---


async def chat_direct(
    message: str,
    *,
    base_url: str,
    api_key: str,
    model: str,
    temperature: float = 0.7,
    langfuse_handler: CallbackHandler | None = None,
) -> tuple[str, dict[str, Any]]:
    """Direct chat - no RAG. Delegates to execution layer. Returns (response, metadata)."""
    return await execute_direct(
        message,
        base_url=base_url,
        api_key=api_key,
        model=model,
        temperature=temperature,
        langfuse_handler=langfuse_handler,
    )


async def chat_rag(
    message: str,
    *,
    pool: asyncpg.Pool,
    base_url: str,
    api_key: str,
    embedding_model: str,
    chat_model: str,
    temperature: float = 0.7,
    top_k: int = 5,
    prompt_name: str = "rag_chat",
    prompt_version: str | None = None,
    langfuse_handler: CallbackHandler | None = None,
) -> tuple[str, dict[str, Any]]:
    """RAG chat - always RAG. Delegates to execution layer."""
    return await execute_rag(
        message,
        pool=pool,
        base_url=base_url,
        api_key=api_key,
        embedding_model=embedding_model,
        chat_model=chat_model,
        top_k=top_k,
        prompt_name=prompt_name,
        prompt_version=prompt_version,
        langfuse_handler=langfuse_handler,
    )


async def chat_smart(
    message: str,
    *,
    pool: asyncpg.Pool,
    base_url: str,
    api_key: str,
    embedding_model: str,
    chat_model: str,
    top_k: int = 5,
    min_similarity: float = 0.0,
    langfuse_handler: CallbackHandler | None = None,
) -> tuple[str, dict[str, Any]]:
    """Smart chat - decision layer chooses RAG or direct based on context.

    Flow: fetch chunks -> decide_strategy -> execute.
    Falls back to direct when retrieval has no relevant context.
    """
    chunks = await _fetch_chunks_for_decision(
        pool,
        message,
        base_url=base_url,
        api_key=api_key,
        embedding_model=embedding_model,
        top_k=top_k,
    )
    strategy = decide_strategy("smart", chunks=chunks, min_similarity=min_similarity)

    if strategy == "rag":
        return await execute_rag(
            message,
            pool=pool,
            base_url=base_url,
            api_key=api_key,
            embedding_model=embedding_model,
            chat_model=chat_model,
            top_k=top_k,
            context_chunks=chunks,
            langfuse_handler=langfuse_handler,
        )
    # direct
    return await execute_direct(
        message,
        base_url=base_url,
        api_key=api_key,
        model=chat_model,
        temperature=0.7,
        langfuse_handler=langfuse_handler,
    )


async def chat_direct_stream(
    message: str,
    *,
    base_url: str,
    api_key: str,
    model: str,
    langfuse_handler: CallbackHandler | None = None,
) -> AsyncIterator[str]:
    """Direct chat streaming. Delegates to execution layer."""
    async for chunk in execute_direct_stream(
        message,
        base_url=base_url,
        api_key=api_key,
        model=model,
        langfuse_handler=langfuse_handler,
    ):
        yield chunk


async def chat_rag_stream(
    message: str,
    *,
    pool: asyncpg.Pool,
    base_url: str,
    api_key: str,
    embedding_model: str,
    chat_model: str,
    top_k: int = 5,
    langfuse_handler: CallbackHandler | None = None,
) -> AsyncIterator[str]:
    """RAG chat streaming. Delegates to execution layer."""
    async for chunk in execute_rag_stream(
        message,
        pool=pool,
        base_url=base_url,
        api_key=api_key,
        embedding_model=embedding_model,
        chat_model=chat_model,
        top_k=top_k,
        langfuse_handler=langfuse_handler,
    ):
        yield chunk


async def chat_smart_stream(
    message: str,
    *,
    pool: asyncpg.Pool,
    base_url: str,
    api_key: str,
    embedding_model: str,
    chat_model: str,
    top_k: int = 5,
    min_similarity: float = 0.0,
    langfuse_handler: CallbackHandler | None = None,
) -> AsyncIterator[str]:
    """Smart chat streaming - decision layer chooses strategy, then streams."""
    chunks = await _fetch_chunks_for_decision(
        pool,
        message,
        base_url=base_url,
        api_key=api_key,
        embedding_model=embedding_model,
        top_k=top_k,
    )
    strategy = decide_strategy("smart", chunks=chunks, min_similarity=min_similarity)

    if strategy == "rag":
        async for chunk in execute_rag_stream(
            message,
            pool=pool,
            base_url=base_url,
            api_key=api_key,
            embedding_model=embedding_model,
            chat_model=chat_model,
            top_k=top_k,
            context_chunks=chunks,
            langfuse_handler=langfuse_handler,
        ):
            yield chunk
    else:
        async for chunk in execute_direct_stream(
            message,
            base_url=base_url,
            api_key=api_key,
            model=chat_model,
            langfuse_handler=langfuse_handler,
        ):
            yield chunk
