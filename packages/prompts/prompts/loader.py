"""Structured prompt loader - loads from registry (MongoDB or other backends)."""

from prompts.models import Prompt, PromptMetadata
from prompts.registry import PromptRegistry

__all__ = [
    "Prompt",
    "PromptMetadata",
    "load_prompt",
    "list_prompt_names",
    "list_prompt_versions",
    "init_registry",
]

_registry: PromptRegistry | None = None


def init_registry(registry: PromptRegistry) -> None:
    """Initialize the prompt registry. Call at application startup."""
    global _registry
    _registry = registry


def _get_registry() -> PromptRegistry:
    if _registry is None:
        raise RuntimeError(
            "Prompt registry not initialized. "
            "Call init_registry(PromptRegistry(repository)) at startup."
        )
    return _registry


def list_prompt_names() -> list[str]:
    """List available prompt names."""
    return _get_registry().list_prompt_names()


def list_prompt_versions(name: str) -> list[str]:
    """List available versions for a prompt."""
    return _get_registry().list_prompt_versions(name)


def load_prompt(
    name: str,
    version: str = "latest",
    alias: str | None = None,
    filename: str = "system.txt",  # kept for backward compat; ignored
) -> Prompt:
    """Load a prompt template with metadata.

    Args:
        name: Prompt name (e.g. 'rag_chat')
        version: Version (e.g. 'v1' or 'latest'). Ignored if alias is set.
        alias: Optional alias (e.g. 'production'). Takes precedence over version.
        filename: Ignored when using registry (kept for backward compatibility).

    Returns:
        Prompt with name, version, content, and metadata.

    Raises:
        FileNotFoundError: If prompt does not exist.
    """
    return _get_registry().load_prompt(name=name, version=version, alias=alias)
