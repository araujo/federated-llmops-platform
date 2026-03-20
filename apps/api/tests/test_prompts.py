"""Unit tests for prompt registry, loader, and repository. No MongoDB required."""

import pytest

from prompts import init_registry, load_prompt, list_prompt_names, list_prompt_versions
from prompts.models import Prompt
from prompts.registry import PromptRegistry
from prompts.repository import InMemoryPromptRepository


# --- Fixtures ---


@pytest.fixture
def repo() -> InMemoryPromptRepository:
    """In-memory repository with default rag_chat v1/v2 seed."""
    return InMemoryPromptRepository()


@pytest.fixture
def registry(repo: InMemoryPromptRepository) -> PromptRegistry:
    """PromptRegistry backed by in-memory repository."""
    return PromptRegistry(repo)


@pytest.fixture(autouse=True)
def init_registry_for_tests(registry: PromptRegistry) -> None:
    """Initialize global registry before each test."""
    init_registry(registry)


# --- InMemoryPromptRepository tests ---


class TestInMemoryPromptRepository:
    """Tests for InMemoryPromptRepository."""

    def test_find_by_name_version(self, repo: InMemoryPromptRepository) -> None:
        """Find prompt by name and version."""
        doc = repo.find_by_name_version("rag_chat", "v1")
        assert doc is not None
        assert doc["name"] == "rag_chat"
        assert doc["version"] == "v1"
        assert "{context}" in doc["content"]

    def test_find_by_name_version_not_found(self, repo: InMemoryPromptRepository) -> None:
        """Returns None for unknown name/version."""
        assert repo.find_by_name_version("unknown", "v1") is None
        assert repo.find_by_name_version("rag_chat", "v99") is None

    def test_find_by_name_alias(self, repo: InMemoryPromptRepository) -> None:
        """Find prompt by name and alias."""
        doc = repo.find_by_name_alias("rag_chat", "latest")
        assert doc is not None
        assert doc["version"] == "v2"

    def test_find_by_name_alias_not_found(self, repo: InMemoryPromptRepository) -> None:
        """Returns None for unknown alias."""
        assert repo.find_by_name_alias("rag_chat", "production") is None
        assert repo.find_by_name_alias("unknown", "latest") is None

    def test_find_latest_by_name(self, repo: InMemoryPromptRepository) -> None:
        """Find latest prompt by name (alias or highest version)."""
        doc = repo.find_latest_by_name("rag_chat")
        assert doc is not None
        assert doc["version"] == "v2"

    def test_find_latest_by_name_not_found(self, repo: InMemoryPromptRepository) -> None:
        """Returns None for unknown name."""
        assert repo.find_latest_by_name("unknown") is None

    def test_list_names(self, repo: InMemoryPromptRepository) -> None:
        """List distinct prompt names."""
        names = repo.list_names()
        assert names == ["rag_chat"]

    def test_list_versions(self, repo: InMemoryPromptRepository) -> None:
        """List versions for a prompt, sorted descending."""
        versions = repo.list_versions("rag_chat")
        assert versions == ["v2", "v1"]

    def test_list_versions_empty_for_unknown(self, repo: InMemoryPromptRepository) -> None:
        """Returns empty list for unknown name."""
        assert repo.list_versions("unknown") == []

    def test_custom_seed(self) -> None:
        """Repository accepts custom seed."""
        seed = [
            {"name": "custom", "version": "v1", "content": "Hello", "alias": "prod"},
        ]
        repo = InMemoryPromptRepository(seed=seed)
        doc = repo.find_by_name_version("custom", "v1")
        assert doc is not None
        assert doc["content"] == "Hello"
        doc = repo.find_by_name_alias("custom", "prod")
        assert doc is not None


# --- PromptRegistry tests ---


class TestPromptRegistry:
    """Tests for PromptRegistry."""

    def test_load_prompt_by_version(self, registry: PromptRegistry) -> None:
        """Load prompt by exact version."""
        prompt = registry.load_prompt("rag_chat", version="v1")
        assert isinstance(prompt, Prompt)
        assert prompt.name == "rag_chat"
        assert prompt.version == "v1"
        assert prompt.metadata.description
        assert "context" in prompt.metadata.variables

    def test_load_prompt_by_alias(self, registry: PromptRegistry) -> None:
        """Load prompt by alias."""
        prompt = registry.load_prompt("rag_chat", alias="latest")
        assert prompt.version == "v2"

    def test_load_prompt_latest_resolution(self, registry: PromptRegistry) -> None:
        """version='latest' resolves to highest version."""
        prompt = registry.load_prompt("rag_chat", version="latest")
        assert prompt.version == "v2"

    def test_load_prompt_alias_takes_precedence(self, registry: PromptRegistry) -> None:
        """Alias takes precedence over version when both provided."""
        prompt = registry.load_prompt("rag_chat", version="v1", alias="latest")
        assert prompt.version == "v2"

    def test_load_prompt_missing_version_raises(self, registry: PromptRegistry) -> None:
        """FileNotFoundError for unknown version."""
        with pytest.raises(FileNotFoundError, match="version 'v99' not found"):
            registry.load_prompt("rag_chat", version="v99")

    def test_load_prompt_missing_alias_raises(self, registry: PromptRegistry) -> None:
        """FileNotFoundError for unknown alias."""
        with pytest.raises(FileNotFoundError, match="alias 'production' not found"):
            registry.load_prompt("rag_chat", alias="production")

    def test_load_prompt_missing_name_raises(self, registry: PromptRegistry) -> None:
        """FileNotFoundError for unknown prompt name."""
        with pytest.raises(FileNotFoundError, match="version 'latest' not found"):
            registry.load_prompt("unknown", version="latest")

    def test_list_prompt_names(self, registry: PromptRegistry) -> None:
        """List prompt names."""
        names = registry.list_prompt_names()
        assert "rag_chat" in names

    def test_list_prompt_versions(self, registry: PromptRegistry) -> None:
        """List versions for a prompt."""
        versions = registry.list_prompt_versions("rag_chat")
        assert "v1" in versions
        assert "v2" in versions
        assert versions.index("v2") < versions.index("v1")


# --- Loader (init_registry + load_prompt) tests ---


class TestLoader:
    """Tests for prompts.loader via init_registry."""

    def test_load_prompt_after_init(self, registry: PromptRegistry) -> None:
        """load_prompt works after init_registry."""
        init_registry(registry)
        prompt = load_prompt("rag_chat", version="v1")
        assert prompt.name == "rag_chat"

    def test_load_prompt_via_loader(self) -> None:
        """load_prompt returns Prompt from initialized registry."""
        prompt = load_prompt("rag_chat", "v1")
        assert prompt.name == "rag_chat"
        assert prompt.version == "v1"

    def test_list_prompt_names_via_loader(self) -> None:
        """list_prompt_names returns names from registry."""
        names = list_prompt_names()
        assert isinstance(names, list)
        assert "rag_chat" in names

    def test_list_prompt_versions_via_loader(self) -> None:
        """list_prompt_versions returns versions from registry."""
        versions = list_prompt_versions("rag_chat")
        assert "v1" in versions
        assert "v2" in versions
