"""Pytest fixtures."""

import pytest
from fastapi.testclient import TestClient

from main import app


@pytest.fixture
def client() -> TestClient:
    """FastAPI test client. Uses context manager so lifespan runs (pool init)."""
    with TestClient(app) as c:
        yield c
