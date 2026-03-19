"""Chat endpoints - delegate to orchestration layer."""

import time

import asyncpg
from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
from langfuse import observe
from langfuse.langchain import CallbackHandler
from pydantic import BaseModel

from api.dependencies import Settings, get_pool, get_settings, verify_api_key
from api.tracing import attach_trace_metadata, build_chat_metadata
from orchestration.chat import (
    chat_direct,
    chat_direct_stream,
    chat_rag as run_chat_rag,
    chat_rag_stream as run_chat_rag_stream,
    chat_smart as run_chat_smart,
    chat_smart_stream as run_chat_smart_stream,
)

router = APIRouter(tags=["chat"])


class ChatRequest(BaseModel):
    """Chat request body."""

    message: str
    prompt_version: str | None = None  # e.g. v1, v2; None = latest


class ChatResponse(BaseModel):
    """Chat response body."""

    response: str


def _get_langfuse_handler(settings: Settings) -> CallbackHandler | None:
    """Create Langfuse callback handler if credentials are set."""
    if settings.langfuse_public_key and settings.langfuse_secret_key:
        return CallbackHandler(
            public_key=settings.langfuse_public_key,
            secret_key=settings.langfuse_secret_key,
            host=settings.langfuse_host,
        )
    return None


@router.post("/chat", response_model=ChatResponse)
@observe()
async def chat(
    body: ChatRequest,
    settings: Settings = Depends(get_settings),
    _: None = Depends(verify_api_key),
) -> ChatResponse:
    """Minimal chat - direct LLM via LiteLLM."""
    handler = _get_langfuse_handler(settings)
    start = time.perf_counter()
    response, metadata = await chat_direct(
        body.message,
        base_url=settings.litellm_base_url,
        api_key=settings.litellm_api_key,
        model=settings.litellm_model,
        langfuse_handler=handler,
    )
    latency_ms = (time.perf_counter() - start) * 1000
    trace_meta = build_chat_metadata(
        route_name="POST /chat",
        strategy_used="direct",
        latency_ms=latency_ms,
        prompt_name=metadata.get("prompt_name"),
        prompt_version=metadata.get("prompt_version"),
        model=metadata.get("model"),
        temperature=metadata.get("temperature"),
        prompt_tokens=metadata.get("prompt_tokens"),
        completion_tokens=metadata.get("completion_tokens"),
        total_tokens=metadata.get("total_tokens"),
    )
    with attach_trace_metadata(trace_meta):
        pass
    return ChatResponse(response=response)


@router.post("/chat/stream")
@observe()
async def chat_stream(
    body: ChatRequest,
    settings: Settings = Depends(get_settings),
    _: None = Depends(verify_api_key),
) -> StreamingResponse:
    """Streaming chat - tokens as they are generated."""
    handler = _get_langfuse_handler(settings)

    async def generate():
        start = time.perf_counter()
        try:
            async for chunk in chat_direct_stream(
                body.message,
                base_url=settings.litellm_base_url,
                api_key=settings.litellm_api_key,
                model=settings.litellm_model,
                langfuse_handler=handler,
            ):
                yield chunk
        finally:
            latency_ms = (time.perf_counter() - start) * 1000
            with attach_trace_metadata(
                build_chat_metadata(
                    route_name="POST /chat/stream",
                    strategy_used="direct",
                    latency_ms=latency_ms,
                    model=settings.litellm_model,
                )
            ):
                pass

    return StreamingResponse(
        generate(),
        media_type="text/plain; charset=utf-8",
    )


@router.post("/chat/rag", response_model=ChatResponse)
@observe()
async def chat_rag(
    body: ChatRequest,
    settings: Settings = Depends(get_settings),
    pool: asyncpg.Pool = Depends(get_pool),
    _: None = Depends(verify_api_key),
) -> ChatResponse:
    """RAG chat - retrieves context from chunks, then generates with LLM."""
    handler = _get_langfuse_handler(settings)
    start = time.perf_counter()
    response, metadata = await run_chat_rag(
        body.message,
        pool=pool,
        base_url=settings.litellm_base_url,
        api_key=settings.litellm_api_key,
        embedding_model=settings.litellm_embedding_model,
        chat_model=settings.litellm_model,
        top_k=5,
        prompt_version=body.prompt_version,
        langfuse_handler=handler,
    )
    latency_ms = (time.perf_counter() - start) * 1000
    trace_meta = build_chat_metadata(
        route_name="POST /chat/rag",
        strategy_used="rag",
        latency_ms=latency_ms,
        prompt_name=metadata.get("prompt_name"),
        prompt_version=metadata.get("prompt_version"),
        model=metadata.get("model"),
        temperature=metadata.get("temperature"),
        prompt_tokens=metadata.get("prompt_tokens"),
        completion_tokens=metadata.get("completion_tokens"),
        total_tokens=metadata.get("total_tokens"),
    )
    with attach_trace_metadata(trace_meta):
        pass
    return ChatResponse(response=response)


@router.post("/chat/rag/stream")
@observe()
async def chat_rag_stream(
    body: ChatRequest,
    settings: Settings = Depends(get_settings),
    pool: asyncpg.Pool = Depends(get_pool),
    _: None = Depends(verify_api_key),
) -> StreamingResponse:
    """Streaming RAG chat."""
    handler = _get_langfuse_handler(settings)

    async def generate():
        start = time.perf_counter()
        try:
            async for chunk in run_chat_rag_stream(
                body.message,
                pool=pool,
                base_url=settings.litellm_base_url,
                api_key=settings.litellm_api_key,
                embedding_model=settings.litellm_embedding_model,
                chat_model=settings.litellm_model,
                top_k=5,
                langfuse_handler=handler,
            ):
                yield chunk
        finally:
            latency_ms = (time.perf_counter() - start) * 1000
            with attach_trace_metadata(
                build_chat_metadata(
                    route_name="POST /chat/rag/stream",
                    strategy_used="rag",
                    latency_ms=latency_ms,
                )
            ):
                pass

    return StreamingResponse(
        generate(),
        media_type="text/plain; charset=utf-8",
    )


@router.post("/chat/smart", response_model=ChatResponse)
@observe()
async def chat_smart(
    body: ChatRequest,
    settings: Settings = Depends(get_settings),
    pool: asyncpg.Pool = Depends(get_pool),
    _: None = Depends(verify_api_key),
) -> ChatResponse:
    """Smart chat - RAG when context exists, else direct LLM."""
    handler = _get_langfuse_handler(settings)
    start = time.perf_counter()
    response, metadata = await run_chat_smart(
        body.message,
        pool=pool,
        base_url=settings.litellm_base_url,
        api_key=settings.litellm_api_key,
        embedding_model=settings.litellm_embedding_model,
        chat_model=settings.litellm_model,
        top_k=5,
        langfuse_handler=handler,
    )
    latency_ms = (time.perf_counter() - start) * 1000
    strategy = "rag" if metadata.get("prompt_name") else "direct"
    trace_meta = build_chat_metadata(
        route_name="POST /chat/smart",
        strategy_used=strategy,
        latency_ms=latency_ms,
        prompt_name=metadata.get("prompt_name"),
        prompt_version=metadata.get("prompt_version"),
        model=metadata.get("model"),
        temperature=metadata.get("temperature"),
        prompt_tokens=metadata.get("prompt_tokens"),
        completion_tokens=metadata.get("completion_tokens"),
        total_tokens=metadata.get("total_tokens"),
    )
    with attach_trace_metadata(trace_meta):
        pass
    return ChatResponse(response=response)


@router.post("/chat/smart/stream")
@observe()
async def chat_smart_stream(
    body: ChatRequest,
    settings: Settings = Depends(get_settings),
    pool: asyncpg.Pool = Depends(get_pool),
    _: None = Depends(verify_api_key),
) -> StreamingResponse:
    """Streaming smart chat."""
    handler = _get_langfuse_handler(settings)

    async def generate():
        start = time.perf_counter()
        try:
            async for chunk in run_chat_smart_stream(
                body.message,
                pool=pool,
                base_url=settings.litellm_base_url,
                api_key=settings.litellm_api_key,
                embedding_model=settings.litellm_embedding_model,
                chat_model=settings.litellm_model,
                top_k=5,
                langfuse_handler=handler,
            ):
                yield chunk
        finally:
            latency_ms = (time.perf_counter() - start) * 1000
            with attach_trace_metadata(
                build_chat_metadata(
                    route_name="POST /chat/smart/stream",
                    strategy_used="smart",
                    latency_ms=latency_ms,
                )
            ):
                pass

    return StreamingResponse(
        generate(),
        media_type="text/plain; charset=utf-8",
    )