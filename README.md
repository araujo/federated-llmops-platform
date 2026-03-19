# Federated LLMOps Platform

A local, self-hosted, Docker-based federated LLMOps platform for operating LLM applications in production-like environments. Focuses on prompt management, agent orchestration, RAG, observability, and model gateway.

## Quick Start

```bash
# 1. Copy environment file
cp .env.example .env

# 2. Start all services
docker compose up -d

# 3. Pull Ollama models (required for chat and embeddings)
docker compose exec ollama ollama pull llama3.2
docker compose exec ollama ollama pull nomic-embed-text

# 4. (Optional) Observability - Langfuse tracing + Prometheus metrics
# Langfuse: Open http://localhost:3000, create account, create project, copy API keys to .env:
#   LANGFUSE_PUBLIC_KEY=pk-lf-...
#   LANGFUSE_SECRET_KEY=sk-lf-...
# Prometheus: Scrape GET /metrics for request counts and latency
# Then: docker compose restart api
```

## Services

| Service    | Port | Purpose                    |
|-----------|------|----------------------------|
| API       | 8000 | Main application           |
| LiteLLM   | 4000 | Model gateway              |
| Ollama    | 11434| Local LLM runtime          |
| Postgres  | 5432 | App DB + pgvector          |
| MinIO     | 9000, 9001 | Object storage       |
| Langfuse  | 3000 | Tracing, observability     |

## API Endpoints

- `GET /` - Service info
- `GET /health` - Health check (postgres, minio, litellm, langfuse)
- `GET /metrics` - Prometheus metrics (request count, latency)
- `POST /chat` - Minimal chat (direct LLM via LiteLLM, no RAG)
- `POST /chat/rag` - RAG chat (retrieves from chunks, generates with context)
- `POST /documents/upload` - Upload .txt or .md document (chunks, embeds, stores)
- `GET /documents` - List all documents
- `DELETE /documents/{id}` - Delete document and its chunks
- `POST /chat/smart` - Smart chat (RAG when context exists, else direct LLM)
- `POST /chat/stream`, `/chat/rag/stream`, `/chat/smart/stream` - Streaming chat (tokens as generated)
- `GET /prompts` - List available prompt names
- `GET /prompts/{name}` - List versions and metadata for a prompt
- `GET /prompts/{name}/{version}` - Get prompt metadata (add `?include_content=true` for content)

When `API_KEY` is set in `.env`, chat and document endpoints require the `X-API-Key` header. Set `RATE_LIMIT_PER_MINUTE` to limit requests per IP.

```bash
# Direct chat
curl -X POST http://localhost:8000/chat -H "Content-Type: application/json" -d '{"message":"hello"}'

# RAG chat (requires ingested documents with chunks)
curl -X POST http://localhost:8000/chat/rag -H "Content-Type: application/json" -d '{"message":"What is in the documents?"}'

# Upload document
curl -X POST http://localhost:8000/documents/upload -F "file=@mydoc.txt"

# List documents
curl http://localhost:8000/documents

# Smart chat (auto RAG or direct)
curl -X POST http://localhost:8000/chat/smart -H "Content-Type: application/json" -d '{"message":"hello"}'

# Streaming chat
curl -N -X POST http://localhost:8000/chat/stream -H "Content-Type: application/json" -d '{"message":"hello"}'
```

## Tests

```bash
pip install -e packages/prompts
cd apps/api && pip install -e ".[dev]" && pytest
# With env (Postgres, MinIO, etc. must be running):
POSTGRES_HOST=localhost POSTGRES_PORT=5432 MINIO_HOST=localhost MINIO_PORT=9000 \
  LITELLM_BASE_URL=http://localhost:4000/v1 LANGFUSE_HOST=http://localhost:3000 \
  pytest tests/ -v
```

CI runs on push/PR via GitHub Actions (`.github/workflows/ci.yml`).

## Evaluations

Run evals against the live API (requires API to be running):

```bash
# From repo root (API must be running, e.g. docker compose up -d)
python -m evals.run evals/datasets/rag_eval_dataset.json

# Or from inside the API container
docker compose exec api sh -c "cd /app && python -m evals.run evals/datasets/rag_eval_dataset.json"
```

Results are saved to `evals/results/results_<timestamp>.json`.

## Requirements

- Docker & Docker Compose
- 8GB+ RAM recommended (Ollama + Langfuse)
- No cloud dependencies; runs fully offline after initial pull
