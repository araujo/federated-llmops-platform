"""MongoDB implementation of PromptRepository."""

from typing import Any

from pymongo.database import Database

from prompts.repository import PromptRepository, _parse_version
from prompts.schema import PROMPTS_COLLECTION


def _doc_to_dict(doc: dict[str, Any] | None) -> dict[str, Any] | None:
    """Convert MongoDB doc (with _id) to plain dict, stringifying ObjectId."""
    if doc is None:
        return None
    out = dict(doc)
    if "_id" in out:
        out["_id"] = str(out["_id"])
    return out


class MongoPromptRepository(PromptRepository):
    """MongoDB implementation of PromptRepository."""

    def __init__(self, db: Database, collection: str = PROMPTS_COLLECTION) -> None:
        self._coll = db[collection]

    def find_by_name_version(self, name: str, version: str) -> dict[str, Any] | None:
        doc = self._coll.find_one({"name": name, "version": version})
        return _doc_to_dict(doc) if doc else None

    def find_by_name_alias(self, name: str, alias: str) -> dict[str, Any] | None:
        doc = self._coll.find_one({"name": name, "alias": alias})
        return _doc_to_dict(doc) if doc else None

    def find_latest_by_name(self, name: str) -> dict[str, Any] | None:
        # Prefer alias="latest" if present
        doc = self._coll.find_one({"name": name, "alias": "latest"})
        if doc:
            return _doc_to_dict(doc)
        # Else sort by version descending (semantic version ordering)
        docs = list(self._coll.find({"name": name}))
        if not docs:
            return None
        best = max(docs, key=lambda d: _parse_version(d.get("version", "")))
        return _doc_to_dict(best)

    def list_names(self) -> list[str]:
        return list(self._coll.distinct("name"))

    def list_versions(self, name: str) -> list[str]:
        versions = list(self._coll.distinct("version", {"name": name}))
        return sorted(versions, key=_parse_version, reverse=True)
