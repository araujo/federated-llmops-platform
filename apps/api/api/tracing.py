"""Tracing helper - consistent metadata for Langfuse traces."""

from contextlib import contextmanager
from typing import Any

from langfuse import propagate_attributes


def build_chat_metadata(
    route_name: str,
    strategy_used: str,
    latency_ms: float,
    *,
    prompt_name: str | None = None,
    prompt_version: str | None = None,
    prompt_alias: str | None = None,
    model: str | None = None,
    temperature: float | None = None,
    prompt_tokens: int | None = None,
    completion_tokens: int | None = None,
    total_tokens: int | None = None,
) -> dict[str, Any]:
    """Build metadata dict for chat traces.

    All values are converted to strings for Langfuse metadata.
    """
    meta: dict[str, Any] = {
        "route_name": route_name,
        "strategy_used": strategy_used,
        "latency_ms": str(round(latency_ms, 2)),
    }
    if prompt_name is not None:
        meta["prompt_name"] = str(prompt_name)
    if prompt_version is not None:
        meta["prompt_version"] = str(prompt_version)
    if prompt_alias is not None:
        meta["prompt_alias"] = str(prompt_alias)
    if model is not None:
        meta["model"] = str(model)
    if temperature is not None:
        meta["temperature"] = str(temperature)
    if prompt_tokens is not None:
        meta["prompt_tokens"] = str(prompt_tokens)
    if completion_tokens is not None:
        meta["completion_tokens"] = str(completion_tokens)
    if total_tokens is not None:
        meta["total_tokens"] = str(total_tokens)
    return meta


def _to_str_dict(d: dict[str, Any]) -> dict[str, str]:
    """Convert metadata values to strings for propagate_attributes."""
    return {
        k: str(v) if v is not None else ""
        for k, v in d.items()
    }


@contextmanager
def attach_trace_metadata(metadata: dict[str, Any]):
    """Attach metadata to the current Langfuse trace via propagate_attributes."""
    with propagate_attributes(metadata=_to_str_dict(metadata)):
        yield
