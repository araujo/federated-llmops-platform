# Prompt Registry — MongoDB Data Model (Part 1)

## Collection: `prompts`

### Document Schema

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `_id` | ObjectId | auto | MongoDB document ID |
| `name` | string | ✓ | Prompt name (e.g. `rag_chat`) |
| `version` | string | ✓ | Version (e.g. `v1`, `v2`) |
| `alias` | string | | Optional alias: `latest`, `production`, `staging` |
| `description` | string | | Human-readable description |
| `content` | string | ✓ | Prompt template content |
| `model` | string | | Default model |
| `temperature` | number | | Default temperature (0.7) |
| `max_tokens` | int | | Default max tokens (4096) |
| `variables` | array[string] | | Template variables (e.g. `["context"]`) |
| `tags` | array[string] | | Tags for filtering |
| `status` | string | | `draft` \| `active` \| `archived` |
| `created_at` | datetime | | Creation timestamp |
| `updated_at` | datetime | | Last update timestamp |
| `created_by` | string | | Optional creator identifier |
| `metadata` | object | | Optional extra metadata |

### Uniqueness

- **(name, version)** must be unique across the collection.

### Query Patterns

| Lookup | Query |
|--------|-------|
| By name + version | `{ name: "rag_chat", version: "v1" }` |
| By name + alias | `{ name: "rag_chat", alias: "production" }` |
| By name + "latest" | Resolve: find by name, sort by version desc, take first; or use alias `latest` |

### Indexes

| Index | Keys | Unique | Purpose |
|-------|------|--------|---------|
| `idx_name_version_unique` | (name, version) | ✓ | Uniqueness, exact lookup |
| `idx_name_alias` | (name, alias) | | Alias lookup (sparse) |
| `idx_name` | (name) | | List versions, latest resolution |
| `idx_status` | (status) | | Filter by status |
| `idx_tags` | (tags) | | Metadata-driven retrieval |

### Example Document

```json
{
  "_id": ObjectId("..."),
  "name": "rag_chat",
  "version": "v1",
  "alias": "latest",
  "description": "RAG chat system prompt - answers using provided context only",
  "content": "You are a helpful assistant. Answer the user's question using only the provided context.\n\nContext:\n{context}",
  "model": "ollama/llama3.2",
  "temperature": 0.7,
  "max_tokens": 4096,
  "variables": ["context"],
  "tags": ["rag", "chat"],
  "status": "active",
  "created_at": ISODate("2025-03-19T00:00:00Z"),
  "updated_at": ISODate("2025-03-19T00:00:00Z"),
  "created_by": null,
  "metadata": {}
}
```
