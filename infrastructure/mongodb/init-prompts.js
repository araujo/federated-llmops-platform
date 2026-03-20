// MongoDB init script for prompts collection
// Run once to create collection and indexes

db = db.getSiblingDB("llmops");

// Create prompts collection (if not exists)
db.createCollection("prompts", {
  validator: {
    $jsonSchema: {
      bsonType: "object",
      required: ["name", "version", "content"],
      properties: {
        name: { bsonType: "string", description: "Prompt name" },
        version: { bsonType: "string", description: "Version e.g. v1, v2" },
        alias: { bsonType: ["string", "null"], description: "Optional alias" },
        description: { bsonType: "string" },
        content: { bsonType: "string", description: "Template content" },
        model: { bsonType: "string" },
        temperature: { bsonType: "number" },
        max_tokens: { bsonType: "int" },
        variables: { bsonType: "array", items: { bsonType: "string" } },
        tags: { bsonType: "array", items: { bsonType: "string" } },
        status: { bsonType: "string", enum: ["draft", "active", "archived"] },
        created_at: { bsonType: "date" },
        updated_at: { bsonType: "date" },
        created_by: { bsonType: ["string", "null"] },
        metadata: { bsonType: "object" }
      }
    }
  }
});

// Unique: (name, version)
db.prompts.createIndex(
  { name: 1, version: 1 },
  { unique: true, name: "idx_name_version_unique" }
);

// Query by (name, alias)
db.prompts.createIndex(
  { name: 1, alias: 1 },
  { name: "idx_name_alias", sparse: true }
);

// Query by name (for latest resolution)
db.prompts.createIndex(
  { name: 1 },
  { name: "idx_name" }
);

// Filter by status
db.prompts.createIndex(
  { status: 1 },
  { name: "idx_status" }
);

// Tags for metadata-driven retrieval
db.prompts.createIndex(
  { tags: 1 },
  { name: "idx_tags" }
);

print("Prompts collection and indexes created.");

// Seed default prompts (rag_chat v1, v2)
var existing = db.prompts.countDocuments({ name: "rag_chat" });
if (existing === 0) {
  db.prompts.insertMany([
    {
      name: "rag_chat",
      version: "v1",
      alias: null,
      description: "RAG chat system prompt - answers using provided context only",
      content: "You are a helpful assistant. Answer the user's question using only the provided context. If the context does not contain relevant information, say so. Do not make up information.\n\nContext:\n{context}\n",
      model: "ollama/llama3.2",
      temperature: 0.7,
      max_tokens: 4096,
      variables: ["context"],
      tags: [],
      status: "active",
      created_at: new Date(),
      updated_at: new Date(),
      created_by: null,
      metadata: {}
    },
    {
      name: "rag_chat",
      version: "v2",
      alias: "latest",
      description: "RAG chat v2 - concise, direct answers",
      content: "You are a concise assistant. Use the context below to answer. If the context lacks relevant information, respond briefly that you don't have enough information. Be direct and brief.\n\nContext:\n{context}\n",
      model: "ollama/llama3.2",
      temperature: 0.7,
      max_tokens: 4096,
      variables: ["context"],
      tags: [],
      status: "active",
      created_at: new Date(),
      updated_at: new Date(),
      created_by: null,
      metadata: {}
    }
  ]);
  print("Seeded rag_chat v1 and v2.");
}
