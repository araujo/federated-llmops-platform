"""Prompt management API endpoints."""

from fastapi import APIRouter, HTTPException, Query

from prompts import load_prompt, list_prompt_names, list_prompt_versions

router = APIRouter(prefix="/prompts", tags=["prompts"])


@router.get("")
async def list_prompts() -> dict:
    """List available prompt names."""
    names = list_prompt_names()
    return {"prompts": names}


@router.get("/{name}")
async def get_prompt_versions(name: str) -> dict:
    """List available versions and metadata for a prompt."""
    try:
        versions = list_prompt_versions(name)
    except FileNotFoundError:
        raise HTTPException(404, f"Prompt '{name}' not found")

    result = {"name": name, "versions": []}
    for v in versions:
        try:
            prompt = load_prompt(name, v)
            result["versions"].append(
                {
                    "version": prompt.version,
                    "description": prompt.metadata.description,
                    "model": prompt.metadata.model,
                    "temperature": prompt.metadata.temperature,
                    "max_tokens": prompt.metadata.max_tokens,
                }
            )
        except FileNotFoundError:
            result["versions"].append(
                {"version": v, "description": "", "model": "", "temperature": 0.7, "max_tokens": 4096}
            )

    return result


@router.get("/{name}/{version}")
async def get_prompt(
    name: str,
    version: str,
    include_content: bool = Query(False, description="Include prompt content"),
) -> dict:
    """Get prompt metadata and optionally content."""
    try:
        prompt = load_prompt(name, version)
    except FileNotFoundError as e:
        raise HTTPException(404, str(e))

    result = {
        "name": prompt.name,
        "version": prompt.version,
        "metadata": {
            "description": prompt.metadata.description,
            "model": prompt.metadata.model,
            "temperature": prompt.metadata.temperature,
            "max_tokens": prompt.metadata.max_tokens,
        },
    }
    if include_content:
        result["content"] = prompt.content
    return result
