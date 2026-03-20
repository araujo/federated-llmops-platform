"""Prompt management API endpoints - read from MongoDB via registry."""

from fastapi import APIRouter, HTTPException, Query

from prompts import load_prompt, list_prompt_names, list_prompt_versions
from prompts.models import Prompt

router = APIRouter(prefix="/prompts", tags=["prompts"])


def _metadata_dict(p: Prompt, include_rich: bool = True) -> dict:
    """Build metadata dict for API response."""
    base = {
        "description": p.metadata.description,
        "model": p.metadata.model,
        "temperature": p.metadata.temperature,
        "max_tokens": p.metadata.max_tokens,
    }
    if include_rich:
        if p.metadata.variables:
            base["variables"] = p.metadata.variables
        if p.metadata.tags:
            base["tags"] = p.metadata.tags
        base["status"] = p.metadata.status
        if p.metadata.alias:
            base["alias"] = p.metadata.alias
        if p.metadata.created_at:
            base["created_at"] = p.metadata.created_at.isoformat()
        if p.metadata.updated_at:
            base["updated_at"] = p.metadata.updated_at.isoformat()
        if p.metadata.metadata:
            base["metadata"] = p.metadata.metadata
    return base


def _version_summary(p: Prompt) -> dict:
    """Build version summary for list response."""
    d = {
        "version": p.version,
        "description": p.metadata.description,
        "model": p.metadata.model,
        "temperature": p.metadata.temperature,
        "max_tokens": p.metadata.max_tokens,
    }
    if p.metadata.variables:
        d["variables"] = p.metadata.variables
    if p.metadata.alias:
        d["alias"] = p.metadata.alias
    if p.metadata.status:
        d["status"] = p.metadata.status
    return d


@router.get("")
async def list_prompts() -> dict:
    """List available prompt names."""
    names = list_prompt_names()
    return {"prompts": names}


@router.get("/{name}")
async def get_prompt_versions(
    name: str,
    alias: str | None = Query(None, description="Load prompt by alias (e.g. latest, production)"),
    include_content: bool = Query(True, description="Include content when loading by alias"),
) -> dict:
    """List versions or load by alias. Use ?alias=latest to load by alias."""
    if alias is not None:
        try:
            prompt = load_prompt(name, alias=alias)
        except FileNotFoundError:
            raise HTTPException(404, f"Prompt '{name}' alias '{alias}' not found")
        result = {
            "name": prompt.name,
            "version": prompt.version,
            "metadata": _metadata_dict(prompt),
        }
        if include_content:
            result["content"] = prompt.content
        return result

    versions = list_prompt_versions(name)
    if not versions:
        raise HTTPException(404, f"Prompt '{name}' not found")

    result = {"name": name, "versions": []}
    for v in versions:
        try:
            prompt = load_prompt(name, v)
            result["versions"].append(_version_summary(prompt))
        except FileNotFoundError:
            result["versions"].append(
                {"version": v, "description": "", "model": "", "temperature": 0.7, "max_tokens": 4096}
            )

    return result


@router.get("/{name}/{version}")
async def get_prompt(
    name: str,
    version: str,
    alias: str | None = Query(None, description="Load by alias instead of version"),
    include_content: bool = Query(False, description="Include prompt content"),
) -> dict:
    """Get prompt metadata and optionally content. version='latest' resolves to highest."""
    try:
        if alias is not None:
            prompt = load_prompt(name, alias=alias)
        else:
            prompt = load_prompt(name, version=version)
    except FileNotFoundError as e:
        raise HTTPException(404, str(e))

    result = {
        "name": prompt.name,
        "version": prompt.version,
        "metadata": _metadata_dict(prompt),
    }
    if include_content:
        result["content"] = prompt.content
    return result
