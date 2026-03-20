"""MongoDB prompt registry - data model and schema."""

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class PromptDocument(BaseModel):
    """MongoDB document schema for the prompts collection.

    Collection: prompts

    Uniqueness: (name, version) must be unique.
    Query patterns:
      - name + version (exact)
      - name + alias (e.g. "latest", "production", "staging")
      - name + "latest" (resolve to highest version)
    """

    id: str | None = Field(None, alias="_id")
    name: str = Field(..., description="Prompt name, e.g. 'rag_chat'")
    version: str = Field(..., description="Version, e.g. 'v1', 'v2'")
    alias: str | None = Field(
        None,
        description="Optional alias: 'latest', 'production', 'staging'",
    )
    description: str = Field("", description="Human-readable description")
    content: str = Field(..., description="Prompt template content (e.g. system.txt)")
    model: str = Field("", description="Default model for this prompt")
    temperature: float = Field(0.7, description="Default temperature")
    max_tokens: int = Field(4096, description="Default max tokens")
    variables: list[str] = Field(
        default_factory=list,
        description="Optional template variables (e.g. ['context'])",
    )
    tags: list[str] = Field(
        default_factory=list,
        description="Optional tags for filtering",
    )
    status: str = Field(
        "active",
        description="draft | active | archived",
    )
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    created_by: str | None = Field(None, description="Optional creator identifier")
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Optional extra metadata",
    )

    model_config = {"populate_by_name": True}


# --- Index definitions for MongoDB ---

PROMPTS_COLLECTION = "prompts"

# Unique compound index: (name, version)
INDEX_NAME_VERSION_UNIQUE = {
    "name": "idx_name_version_unique",
    "keys": [("name", 1), ("version", 1)],
    "unique": True,
}

# Query by alias: (name, alias)
INDEX_NAME_ALIAS = {
    "name": "idx_name_alias",
    "keys": [("name", 1), ("alias", 1)],
}

# Query by name for "latest" resolution
INDEX_NAME = {
    "name": "idx_name",
    "keys": [("name", 1)],
}

# Filter by status
INDEX_STATUS = {
    "name": "idx_status",
    "keys": [("status", 1)],
}

# Tags for metadata-driven retrieval
INDEX_TAGS = {
    "name": "idx_tags",
    "keys": [("tags", 1)],
}

PROMPT_INDEXES = [
    INDEX_NAME_VERSION_UNIQUE,
    INDEX_NAME_ALIAS,
    INDEX_NAME,
    INDEX_STATUS,
    INDEX_TAGS,
]
