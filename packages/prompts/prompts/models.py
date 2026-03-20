"""Prompt models - domain objects for the prompt registry."""

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class PromptMetadata(BaseModel):
    """Metadata for a prompt (model, temperature, etc.)."""

    name: str
    version: str
    description: str = ""
    model: str = ""
    temperature: float = 0.7
    max_tokens: int = 4096
    # Richer metadata from MongoDB
    variables: list[str] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)
    status: str = "active"
    alias: str | None = None
    created_at: datetime | None = None
    updated_at: datetime | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)



class Prompt(BaseModel):
    """Structured prompt with content and metadata.

    Returned by load_prompt() - compatible with existing consumers.
    """

    name: str
    version: str
    content: str
    metadata: PromptMetadata
