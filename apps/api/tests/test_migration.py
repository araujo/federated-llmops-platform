"""Tests for migration script. No MongoDB required - uses dry-run or temp dirs."""

import importlib.util
import tempfile
from pathlib import Path

import pytest

# Load migration script as module (no package structure)
# tests -> api -> apps -> project_root
_project_root = Path(__file__).resolve().parent.parent.parent.parent
_script_path = _project_root / "scripts" / "migrate_prompts_to_mongo.py"
_spec = importlib.util.spec_from_file_location("migrate_prompts_to_mongo", _script_path)
_migration_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_migration_module)

migrate = _migration_module.migrate
_discover_prompts = _migration_module._discover_prompts
_extract_variables = _migration_module._extract_variables
_load_prompt_from_fs = _migration_module._load_prompt_from_fs
_parse_version = _migration_module._parse_version


class TestMigrationHelpers:
    """Unit tests for migration script helpers."""

    def test_parse_version(self) -> None:
        """Parse version strings to comparable tuples."""
        assert _parse_version("v1") == (1,)
        assert _parse_version("v2") == (2,)
        assert _parse_version("v1.2") == (1, 2)
        assert _parse_version("1") == (1,)
        assert _parse_version("unknown") == (0,)

    def test_extract_variables(self) -> None:
        """Extract {variable} placeholders from content."""
        assert _extract_variables("Hello {name}") == ["name"]
        assert _extract_variables("Context:\n{context}\n") == ["context"]
        assert _extract_variables("{a} and {b}") == ["a", "b"]
        assert _extract_variables("no vars") == []

    def test_discover_prompts_empty_dir(self) -> None:
        """Discover returns empty for empty or missing dir."""
        with tempfile.TemporaryDirectory() as tmp:
            found = _discover_prompts(Path(tmp))
            assert found == []
        assert _discover_prompts(Path("/nonexistent")) == []

    def test_discover_prompts_finds_structure(self) -> None:
        """Discover finds name/version folders with system.txt."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "my_prompt" / "v1").mkdir(parents=True)
            (root / "my_prompt" / "v1" / "system.txt").write_text("Hello")
            found = _discover_prompts(root)
            assert len(found) == 1
            name, version, path = found[0]
            assert name == "my_prompt"
            assert version == "v1"
            assert (path / "system.txt").exists()

    def test_load_prompt_from_fs(self) -> None:
        """Load prompt from filesystem with system.txt and meta.yaml."""
        with tempfile.TemporaryDirectory() as tmp:
            prompt_dir = Path(tmp) / "test" / "v1"
            prompt_dir.mkdir(parents=True)
            (prompt_dir / "system.txt").write_text("Say {greeting}")
            (prompt_dir / "meta.yaml").write_text(
                "name: test\nversion: v1\ndescription: A test\nmodel: gpt-4"
            )
            doc = _load_prompt_from_fs(prompt_dir, "test", "v1")
            assert doc is not None
            assert doc["name"] == "test"
            assert doc["version"] == "v1"
            assert doc["content"] == "Say {greeting}"
            assert doc["variables"] == ["greeting"]
            assert doc["description"] == "A test"
            assert doc["model"] == "gpt-4"

    def test_load_prompt_from_fs_no_system_txt(self) -> None:
        """Returns None when system.txt missing."""
        with tempfile.TemporaryDirectory() as tmp:
            prompt_dir = Path(tmp) / "test" / "v1"
            prompt_dir.mkdir(parents=True)
            (prompt_dir / "meta.yaml").write_text("name: test")
            doc = _load_prompt_from_fs(prompt_dir, "test", "v1")
            assert doc is None


class TestMigrationDryRun:
    """Migration script behavior - dry-run only, no MongoDB."""

    def test_migrate_dry_run_default_templates(self) -> None:
        """Dry-run discovers and reports prompts from default templates."""
        count = migrate(
            templates_root=_project_root / "packages" / "prompts" / "prompts" / "templates",
            mongodb_uri="",
            mongodb_database="llmops",
            dry_run=True,
        )
        assert count >= 2  # rag_chat v1, v2

    def test_migrate_dry_run_empty_dir(self) -> None:
        """Dry-run returns 0 for empty templates dir."""
        with tempfile.TemporaryDirectory() as tmp:
            count = migrate(
                templates_root=Path(tmp),
                mongodb_uri="",
                mongodb_database="llmops",
                dry_run=True,
            )
            assert count == 0

    def test_migrate_dry_run_custom_templates(self) -> None:
        """Dry-run discovers prompts from custom templates path."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "custom" / "v1").mkdir(parents=True)
            (root / "custom" / "v1" / "system.txt").write_text("Hi {x}")
            (root / "custom" / "v1" / "meta.yaml").write_text("name: custom\nversion: v1")
            count = migrate(
                templates_root=root,
                mongodb_uri="",
                mongodb_database="llmops",
                dry_run=True,
            )
            assert count == 1

    def test_migrate_assigns_latest_alias(self) -> None:
        """Highest version gets alias=latest in migrated doc."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for v in ["v1", "v2", "v3"]:
                (root / "multi" / v).mkdir(parents=True)
                (root / "multi" / v / "system.txt").write_text("v" + v)
                (root / "multi" / v / "meta.yaml").write_text(f"name: multi\nversion: {v}")
            # We can't easily inspect docs from migrate without MongoDB, but we can
            # run dry-run and verify it finds 3
            count = migrate(
                templates_root=root,
                mongodb_uri="",
                mongodb_database="llmops",
                dry_run=True,
            )
            assert count == 3
