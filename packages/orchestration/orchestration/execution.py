"""Execution layer - pure execution, no decision logic."""

from collections.abc import AsyncIterator
from typing import Any

import asyncpg
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langfuse.langchain import CallbackHandler

from prompts import load_prompt
from retrieval import search_chunks


def _get_llm(
    base_url: str,
    api_key: str,
    model: str,
    streaming: bool = False,
    temperature: float = 0.7,
) -> ChatOpenAI:
    return ChatOpenAI(
        model=model,
        base_url=base_url,
        api_key=api_key,
        temperature=temperature,
        streaming=streaming,
    )


def _get_embeddings(
    base_url: str, api_key: str, model: str
) -> OpenAIEmbeddings:
    return OpenAIEmbeddings(
        model=model,
        openai_api_base=base_url,
        openai_api_key=api_key,
    )


def _extract_usage(response: Any) -> dict[str, int | None]:
    """Extract token usage from LLM response if available."""
    usage: dict[str, int | None] = {}
    um = getattr(response, "usage_metadata", None)
    if um:
        usage["prompt_tokens"] = getattr(um, "input_tokens", None)
        usage["completion_tokens"] = getattr(um, "output_tokens", None)
        usage["total_tokens"] = getattr(um, "total_tokens", None)
    if not usage:
        rm = getattr(response, "response_metadata", None) or {}
        if isinstance(rm, dict):
            u = rm.get("usage", rm.get("token_usage"))
            if isinstance(u, dict):
                usage["prompt_tokens"] = u.get("prompt_tokens", u.get("input_tokens"))
                usage["completion_tokens"] = u.get("completion_tokens", u.get("output_tokens"))
                usage["total_tokens"] = u.get("total_tokens")
    return usage


async def execute_direct(
    message: str,
    *,
    base_url: str,
    api_key: str,
    model: str,
    temperature: float = 0.7,
    langfuse_handler: CallbackHandler | None = None,
) -> tuple[str, dict[str, Any]]:
    """Execute direct LLM chat - no retrieval. Returns (content, metadata)."""
    llm = _get_llm(base_url, api_key, model, streaming=False, temperature=temperature)
    callbacks = [langfuse_handler] if langfuse_handler else []
    response = await llm.ainvoke(
        message,
        config={"callbacks": callbacks},
    )
    metadata: dict[str, Any] = {
        "prompt_name": None,
        "prompt_version": None,
        "model": model,
        "temperature": temperature,
    }
    usage = _extract_usage(response)
    if usage.get("prompt_tokens") is not None:
        metadata["prompt_tokens"] = usage["prompt_tokens"]
    if usage.get("completion_tokens") is not None:
        metadata["completion_tokens"] = usage["completion_tokens"]
    if usage.get("total_tokens") is not None:
        metadata["total_tokens"] = usage["total_tokens"]
    return response.content, metadata


async def execute_rag(
    message: str,
    *,
    pool: asyncpg.Pool,
    base_url: str,
    api_key: str,
    embedding_model: str,
    chat_model: str,
    temperature: float = 0.7,
    top_k: int = 5,
    context_chunks: list[dict] | None = None,
    prompt_name: str = "rag_chat",
    prompt_version: str | None = None,
    prompt_alias: str | None = None,
    langfuse_handler: CallbackHandler | None = None,
) -> tuple[str, dict[str, Any]]:
    """Execute RAG: use provided chunks or fetch, then generate.

    Args:
        context_chunks: Optional pre-fetched chunks. If None, fetches.
        prompt_name: Prompt template name. Default rag_chat.
        prompt_version: Prompt version (e.g. v1, v2). None = latest.
        prompt_alias: Optional alias (e.g. production). Takes precedence over prompt_version.
    """
    if prompt_alias is not None:
        prompt = load_prompt(prompt_name, alias=prompt_alias)
    else:
        version = prompt_version if prompt_version else "latest"
        prompt = load_prompt(prompt_name, version)
    llm = _get_llm(
        base_url, api_key,
        prompt.metadata.model or chat_model,
        streaming=False,
        temperature=prompt.metadata.temperature,
    )
    callbacks = [langfuse_handler] if langfuse_handler else []

    if context_chunks is not None:
        chunks = context_chunks
    else:
        embeddings = _get_embeddings(base_url, api_key, embedding_model)
        query_embedding = await embeddings.aembed_query(message)
        chunks = await search_chunks(pool, query_embedding, top_k=top_k)

    context = (
        "\n\n".join(c["content"] for c in chunks)
        if chunks
        else "No relevant context found."
    )
    system_message = prompt.content.format(context=context)
    messages = [
        SystemMessage(content=system_message),
        HumanMessage(content=message),
    ]

    response = await llm.ainvoke(messages, config={"callbacks": callbacks})
    metadata: dict[str, Any] = {
        "prompt_name": prompt.name,
        "prompt_version": prompt.version,
        "model": prompt.metadata.model or chat_model,
        "temperature": prompt.metadata.temperature,
    }
    if prompt_alias is not None:
        metadata["prompt_alias"] = prompt_alias
    elif prompt.metadata.alias:
        metadata["prompt_alias"] = prompt.metadata.alias
    usage = _extract_usage(response)
    if usage.get("prompt_tokens") is not None:
        metadata["prompt_tokens"] = usage["prompt_tokens"]
    if usage.get("completion_tokens") is not None:
        metadata["completion_tokens"] = usage["completion_tokens"]
    if usage.get("total_tokens") is not None:
        metadata["total_tokens"] = usage["total_tokens"]
    return response.content, metadata


async def execute_direct_stream(
    message: str,
    *,
    base_url: str,
    api_key: str,
    model: str,
    langfuse_handler: CallbackHandler | None = None,
) -> AsyncIterator[str]:
    """Execute direct LLM streaming."""
    llm = _get_llm(base_url, api_key, model, streaming=True)
    callbacks = [langfuse_handler] if langfuse_handler else []
    async for chunk in llm.astream(message, config={"callbacks": callbacks}):
        if chunk.content:
            yield chunk.content


async def execute_rag_stream(
    message: str,
    *,
    pool: asyncpg.Pool,
    base_url: str,
    api_key: str,
    embedding_model: str,
    chat_model: str,
    top_k: int = 5,
    context_chunks: list[dict] | None = None,
    prompt_name: str = "rag_chat",
    prompt_version: str | None = None,
    prompt_alias: str | None = None,
    langfuse_handler: CallbackHandler | None = None,
) -> AsyncIterator[str]:
    """Execute RAG streaming. Uses context_chunks if provided."""
    if prompt_alias is not None:
        prompt = load_prompt(prompt_name, alias=prompt_alias)
    else:
        version = prompt_version if prompt_version else "latest"
        prompt = load_prompt(prompt_name, version)
    llm = _get_llm(
        base_url, api_key,
        prompt.metadata.model or chat_model,
        streaming=True,
        temperature=prompt.metadata.temperature,
    )
    callbacks = [langfuse_handler] if langfuse_handler else []

    if context_chunks is not None:
        chunks = context_chunks
    else:
        embeddings = _get_embeddings(base_url, api_key, embedding_model)
        query_embedding = await embeddings.aembed_query(message)
        chunks = await search_chunks(pool, query_embedding, top_k=top_k)

    context = (
        "\n\n".join(c["content"] for c in chunks)
        if chunks
        else "No relevant context found."
    )
    system_message = prompt.content.format(context=context)
    messages = [
        SystemMessage(content=system_message),
        HumanMessage(content=message),
    ]

    async for chunk in llm.astream(messages, config={"callbacks": callbacks}):
        if chunk.content:
            yield chunk.content
