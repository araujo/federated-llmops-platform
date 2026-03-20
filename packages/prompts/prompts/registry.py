"""Prompt registry - loads prompts via repository abstraction."""

from prompts.models import Prompt, PromptMetadata
from prompts.repository import PromptRepository


class PromptRegistry:
    """Registry that loads prompts from a repository backend."""

    def __init__(self, repository: PromptRepository) -> None:
        self._repo = repository

    def load_prompt(
        self,
        name: str,
        version: str | None = "latest",
        alias: str | None = None,
    ) -> Prompt:
        """Load a prompt by name with version or alias.

        Args:
            name: Prompt name (e.g. 'rag_chat')
            version: Version (e.g. 'v1' or 'latest'). Ignored if alias is set.
            alias: Optional alias (e.g. 'production'). Takes precedence over version.

        Returns:
            Prompt with name, version, content, and metadata.

        Raises:
            FileNotFoundError: If prompt does not exist (kept for API compatibility).
        """
        doc: dict | None = None
        resolved_version = version if version is not None else "latest"
        if alias is not None:
            doc = self._repo.find_by_name_alias(name, alias)
        elif resolved_version == "latest":
            doc = self._repo.find_latest_by_name(name)
        else:
            doc = self._repo.find_by_name_version(name, resolved_version)

        if doc is None:
            v_or_a = f"alias '{alias}'" if alias else f"version '{resolved_version}'"
            raise FileNotFoundError(f"Prompt '{name}' {v_or_a} not found")

        return _doc_to_prompt(doc)

    def list_prompt_names(self) -> list[str]:
        """List available prompt names."""
        return self._repo.list_names()

    def list_prompt_versions(self, name: str) -> list[str]:
        """List available versions for a prompt."""
        return self._repo.list_versions(name)


def _doc_to_prompt(doc: dict) -> Prompt:
    """Convert repository document to Prompt model."""
    variables = doc.get("variables")
    tags = doc.get("tags")
    metadata = PromptMetadata(
        name=doc.get("name", ""),
        version=doc.get("version", ""),
        description=doc.get("description", ""),
        model=doc.get("model", ""),
        temperature=float(doc.get("temperature", 0.7)),
        max_tokens=int(doc.get("max_tokens", 4096)),
        variables=variables if isinstance(variables, list) else [],
        tags=tags if isinstance(tags, list) else [],
        status=str(doc.get("status", "active")),
        alias=doc.get("alias"),
        created_at=doc.get("created_at"),
        updated_at=doc.get("updated_at"),
        metadata=doc.get("metadata") if isinstance(doc.get("metadata"), dict) else {},
    )
    return Prompt(
        name=doc.get("name", ""),
        version=doc.get("version", ""),
        content=doc.get("content", ""),
        metadata=metadata,
    )
