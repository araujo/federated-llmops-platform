"""Repository abstraction for prompt storage."""

import re
from abc import ABC, abstractmethod
from typing import Any


def _parse_version(version_str: str) -> tuple[int, ...]:
    """Parse version string (e.g. 'v1', 'v2') into comparable tuple."""
    match = re.match(r"v?(\d+(?:\.\d+)*)", version_str, re.IGNORECASE)
    if not match:
        return (0,)
    return tuple(int(x) for x in match.group(1).split("."))


class PromptRepository(ABC):
    """Abstract repository for prompt storage. Enables multiple backends."""

    @abstractmethod
    def find_by_name_version(self, name: str, version: str) -> dict[str, Any] | None:
        """Find prompt by name and version. Returns None if not found."""

    @abstractmethod
    def find_by_name_alias(self, name: str, alias: str) -> dict[str, Any] | None:
        """Find prompt by name and alias. Returns None if not found."""

    @abstractmethod
    def find_latest_by_name(self, name: str) -> dict[str, Any] | None:
        """Find latest prompt by name (highest version). Returns None if not found."""

    @abstractmethod
    def list_names(self) -> list[str]:
        """List distinct prompt names."""

    @abstractmethod
    def list_versions(self, name: str) -> list[str]:
        """List versions for a prompt name, sorted by version descending."""


# --- In-memory repository for testing ---

_DEFAULT_SEED = [
    {
        "name": "rag_chat",
        "version": "v1",
        "alias": None,
        "description": "RAG chat system prompt - answers using provided context only",
        "content": "You are a helpful assistant. Answer the user's question using only the provided context. If the context does not contain relevant information, say so. Do not make up information.\n\nContext:\n{context}\n",
        "model": "ollama/llama3.2",
        "temperature": 0.7,
        "max_tokens": 4096,
        "variables": ["context"],
    },
    {
        "name": "rag_chat",
        "version": "v2",
        "alias": "latest",
        "description": "RAG chat v2 - concise, direct answers",
        "content": "You are a concise assistant. Use the context below to answer. If the context lacks relevant information, respond briefly that you don't have enough information. Be direct and brief.\n\nContext:\n{context}\n",
        "model": "ollama/llama3.2",
        "temperature": 0.7,
        "max_tokens": 4096,
        "variables": ["context"],
    },
]


class InMemoryPromptRepository(PromptRepository):
    """In-memory repository for testing. Seeded with rag_chat v1/v2 by default."""

    def __init__(self, seed: list[dict[str, Any]] | None = None) -> None:
        self._docs: list[dict[str, Any]] = list(seed or _DEFAULT_SEED)

    def find_by_name_version(self, name: str, version: str) -> dict[str, Any] | None:
        for d in self._docs:
            if d.get("name") == name and d.get("version") == version:
                return dict(d)
        return None

    def find_by_name_alias(self, name: str, alias: str) -> dict[str, Any] | None:
        for d in self._docs:
            if d.get("name") == name and d.get("alias") == alias:
                return dict(d)
        return None

    def find_latest_by_name(self, name: str) -> dict[str, Any] | None:
        candidates = [d for d in self._docs if d.get("name") == name]
        if not candidates:
            return None
        alias_latest = next((d for d in candidates if d.get("alias") == "latest"), None)
        if alias_latest:
            return dict(alias_latest)
        return dict(max(candidates, key=lambda d: _parse_version(d.get("version", ""))))

    def list_names(self) -> list[str]:
        return sorted(set(d.get("name", "") for d in self._docs if d.get("name")))

    def list_versions(self, name: str) -> list[str]:
        versions = [d["version"] for d in self._docs if d.get("name") == name]
        return sorted(versions, key=_parse_version, reverse=True)
