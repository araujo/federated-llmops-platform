"""API endpoint tests."""

import pytest
from fastapi.testclient import TestClient

from main import app

@pytest.fixture
def mock_embeddings(monkeypatch):
    async def fake_embed_query(self, text):
        return [0.1] * 768  # vetor fake

    monkeypatch.setattr(
        "langchain_openai.embeddings.base.OpenAIEmbeddings.aembed_query",
        fake_embed_query,
    )

def test_root(client: TestClient) -> None:
    """Root returns service info."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["service"] == "llmops-api"
    assert data["status"] == "running"


def test_health(client: TestClient) -> None:
    """Health returns status and component checks."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "postgres" in data
    assert "mongodb" in data
    assert "minio" in data
    assert "litellm" in data
    assert "langfuse" in data


def test_metrics(client: TestClient) -> None:
    """Metrics returns Prometheus format."""
    response = client.get("/metrics")
    assert response.status_code == 200
    assert "llmops" in response.text or "python_" in response.text


def test_x_request_id(client: TestClient) -> None:
    """Response includes X-Request-ID header."""
    response = client.get("/health")
    assert "x-request-id" in [h.lower() for h in response.headers]


def test_documents_list(client: TestClient) -> None:
    """Documents list returns array."""
    response = client.get("/documents")
    assert response.status_code == 200
    assert isinstance(response.json(), list)


def test_documents_delete_not_found(client: TestClient) -> None:
    """Delete non-existent document returns 404."""
    response = client.delete("/documents/00000000-0000-0000-0000-000000000000")
    assert response.status_code == 404


def test_documents_delete_invalid_id(client: TestClient) -> None:
    """Delete with invalid UUID returns 400."""
    response = client.delete("/documents/not-a-uuid")
    assert response.status_code == 400


# --- Prompt API tests (use in-memory registry via MONGODB_URI=memory://, no real MongoDB) ---


def test_prompts_list(client: TestClient) -> None:
    """GET /prompts returns available prompt names."""
    response = client.get("/prompts")
    assert response.status_code == 200
    data = response.json()
    assert "prompts" in data
    assert isinstance(data["prompts"], list)
    assert "rag_chat" in data["prompts"]


def test_prompts_get_versions(client: TestClient) -> None:
    """GET /prompts/{name} returns versions and metadata."""
    response = client.get("/prompts/rag_chat")
    assert response.status_code == 200
    data = response.json()
    assert data["name"] == "rag_chat"
    assert "versions" in data
    assert len(data["versions"]) >= 1
    v = data["versions"][0]
    assert "version" in v
    assert "description" in v
    assert "model" in v
    assert "temperature" in v
    assert "max_tokens" in v


def test_prompts_get_specific(client: TestClient) -> None:
    """GET /prompts/{name}/{version} returns metadata and optionally content."""
    response = client.get("/prompts/rag_chat/v1")
    assert response.status_code == 200
    data = response.json()
    assert data["name"] == "rag_chat"
    assert data["version"] == "v1"
    assert "metadata" in data
    assert "content" not in data

    response = client.get("/prompts/rag_chat/v1?include_content=true")
    assert response.status_code == 200
    data = response.json()
    assert "content" in data
    assert "{context}" in data["content"]


def test_prompts_get_by_alias(client: TestClient) -> None:
    """GET /prompts/{name}?alias=latest returns prompt for that alias."""
    response = client.get("/prompts/rag_chat?alias=latest")
    assert response.status_code == 200
    data = response.json()
    assert data["name"] == "rag_chat"
    assert data["version"] == "v2"
    assert "content" in data
    assert "{context}" in data["content"]


def test_prompts_get_version_latest(client: TestClient) -> None:
    """GET /prompts/{name}/latest resolves to highest version."""
    response = client.get("/prompts/rag_chat/latest")
    assert response.status_code == 200
    data = response.json()
    assert data["name"] == "rag_chat"
    assert data["version"] == "v2"


def test_prompts_get_by_alias_on_version_endpoint(client: TestClient) -> None:
    """GET /prompts/{name}/{version}?alias=latest loads by alias."""
    response = client.get("/prompts/rag_chat/v1?alias=latest")
    assert response.status_code == 200
    data = response.json()
    assert data["version"] == "v2"


def test_prompts_metadata_includes_variables(client: TestClient) -> None:
    """GET /prompts/{name}/{version} includes variables in metadata when present."""
    response = client.get("/prompts/rag_chat/v1?include_content=true")
    assert response.status_code == 200
    data = response.json()
    assert "metadata" in data
    assert "variables" in data["metadata"]
    assert "context" in data["metadata"]["variables"]


def test_prompts_not_found(client: TestClient) -> None:
    """GET /prompts/{name} returns 404 for unknown prompt."""
    response = client.get("/prompts/nonexistent")
    assert response.status_code == 404


def test_prompts_version_not_found(client: TestClient) -> None:
    """GET /prompts/{name}/{version} returns 404 for unknown version."""
    response = client.get("/prompts/rag_chat/v99")
    assert response.status_code == 404


def test_prompts_alias_not_found(client: TestClient) -> None:
    """GET /prompts/{name}?alias=X returns 404 for unknown alias."""
    response = client.get("/prompts/rag_chat?alias=production")
    assert response.status_code == 404


def test_prompts_version_endpoint_alias_not_found(client: TestClient) -> None:
    """GET /prompts/{name}/{version}?alias=X returns 404 for unknown alias."""
    response = client.get("/prompts/rag_chat/v1?alias=nonexistent")
    assert response.status_code == 404


def test_prompts_content_optional_by_alias(client: TestClient) -> None:
    """GET /prompts/{name}?alias=latest&include_content=false omits content."""
    response = client.get("/prompts/rag_chat?alias=latest&include_content=false")
    assert response.status_code == 200
    data = response.json()
    assert "content" not in data
    assert data["version"] == "v2"


def test_documents_search(client: TestClient) -> None:
    """GET /documents/search returns chunks with content, document_id, similarity."""
    response = client.get("/documents/search?q=test&top_k=3")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    for item in data:
        assert "content" in item
        assert "document_id" in item
        assert "similarity" in item
        assert isinstance(item["similarity"], (int, float))


def test_documents_search(client, mock_embeddings):
    response = client.get("/documents/search?q=test&top_k=3")
    assert response.status_code == 200


def test_api_key_required_when_set(monkeypatch: pytest.MonkeyPatch) -> None:
    """When API_KEY is set, protected endpoints require X-API-Key header."""
    from api.dependencies import get_settings

    monkeypatch.setenv("API_KEY", "test-secret-key")
    get_settings.cache_clear()
    try:
        with TestClient(app) as client:
            r = client.get("/documents")
            assert r.status_code == 401
            r = client.get("/documents", headers={"X-API-Key": "wrong-key"})
            assert r.status_code == 401
            r = client.get("/documents", headers={"X-API-Key": "test-secret-key"})
            assert r.status_code == 200  # Auth passes; GET /documents is fast (no LLM)
    finally:
        monkeypatch.delenv("API_KEY", raising=False)
        get_settings.cache_clear()
