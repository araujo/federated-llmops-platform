"""Structured prompt loader with version resolution and metadata."""

import re
from pathlib import Path

import yaml
from pydantic import BaseModel


class PromptMetadata(BaseModel):
    """Prompt metadata from meta.yaml."""

    name: str
    version: str
    description: str = ""
    model: str = ""
    temperature: float = 0.7
    max_tokens: int = 4096


class Prompt(BaseModel):
    """Structured prompt with content and metadata."""

    name: str
    version: str
    content: str
    metadata: PromptMetadata


_TEMPLATES_ROOT = Path(__file__).resolve().parent / "templates"


def _parse_version(version_str: str) -> tuple[int, ...]:
    """Parse version string (e.g. 'v1', 'v2', 'v1.2') into comparable tuple."""
    match = re.match(r"v?(\d+(?:\.\d+)*)", version_str, re.IGNORECASE)
    if not match:
        return (0,)
    return tuple(int(x) for x in match.group(1).split("."))


def _resolve_latest(name: str) -> str:
    """Resolve 'latest' to the highest available version for the given prompt name."""
    prompt_dir = _TEMPLATES_ROOT / name
    if not prompt_dir.exists() or not prompt_dir.is_dir():
        raise FileNotFoundError(f"Prompt '{name}' not found")
    versions = [d.name for d in prompt_dir.iterdir() if d.is_dir()]
    if not versions:
        raise FileNotFoundError(f"No versions found for prompt '{name}'")
    sorted_versions = sorted(versions, key=_parse_version, reverse=True)
    return sorted_versions[0]


def list_prompt_names() -> list[str]:
    """List available prompt names."""
    if not _TEMPLATES_ROOT.exists():
        return []
    return [
        d.name
        for d in _TEMPLATES_ROOT.iterdir()
        if d.is_dir() and not d.name.startswith(".")
    ]


def list_prompt_versions(name: str) -> list[str]:
    """List available versions for a prompt."""
    prompt_dir = _TEMPLATES_ROOT / name
    if not prompt_dir.exists() or not prompt_dir.is_dir():
        raise FileNotFoundError(f"Prompt '{name}' not found")
    return sorted(
        [d.name for d in prompt_dir.iterdir() if d.is_dir()],
        key=_parse_version,
        reverse=True,
    )


def load_prompt(
    name: str,
    version: str = "latest",
    filename: str = "system.txt",
) -> Prompt:
    """Load a prompt template with metadata.

    Args:
        name: Prompt name (e.g. 'rag_chat')
        version: Version (e.g. 'v1' or 'latest')
        filename: Content file name (default: system.txt)

    Returns:
        Prompt with name, version, content, and metadata.

    Raises:
        FileNotFoundError: If prompt or version does not exist.
    """
    if version == "latest":
        version = _resolve_latest(name)

    version_dir = _TEMPLATES_ROOT / name / version
    if not version_dir.exists():
        raise FileNotFoundError(f"Prompt '{name}' version '{version}' not found")

    content_path = version_dir / filename
    if not content_path.exists():
        raise FileNotFoundError(f"Prompt file '{filename}' not found in {name}/{version}")

    meta_path = version_dir / "meta.yaml"
    if meta_path.exists():
        meta_data = yaml.safe_load(meta_path.read_text(encoding="utf-8")) or {}
        metadata = PromptMetadata(
            name=meta_data.get("name", name),
            version=meta_data.get("version", version),
            description=meta_data.get("description", ""),
            model=meta_data.get("model", ""),
            temperature=float(meta_data.get("temperature", 0.7)),
            max_tokens=int(meta_data.get("max_tokens", 4096)),
        )
    else:
        metadata = PromptMetadata(name=name, version=version)

    content = content_path.read_text(encoding="utf-8")
    return Prompt(name=name, version=version, content=content, metadata=metadata)
