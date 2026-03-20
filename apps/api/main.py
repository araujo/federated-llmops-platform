"""Federated LLMOps Platform API - FastAPI application."""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.dependencies import (
    close_mongo,
    close_pool,
    get_mongo_db,
    get_settings,
    init_mongo,
    init_pool,
)
from api.middleware import RateLimitMiddleware, RequestIDMiddleware
from api.routes import chat, documents, health, metrics, prompts
from prompts import init_registry
from prompts.registry import PromptRegistry


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup: init Postgres pool, MongoDB (or in-memory), prompt registry. Shutdown: close all."""
    settings = get_settings()
    await init_pool(settings)
    use_mongo = init_mongo(settings)
    if use_mongo:
        from prompts.repository_mongo import MongoPromptRepository

        repo = MongoPromptRepository(get_mongo_db())
    else:
        from prompts.repository import InMemoryPromptRepository

        repo = InMemoryPromptRepository()
    init_registry(PromptRegistry(repo))
    yield
    close_mongo()
    await close_pool()


app = FastAPI(
    title="Federated LLMOps Platform API",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(RateLimitMiddleware)
app.add_middleware(RequestIDMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router)
app.include_router(metrics.router)
app.include_router(prompts.router)
app.include_router(chat.router)
app.include_router(documents.router)


@app.get("/")
async def root():
    """Root endpoint."""
    return {"service": "llmops-api", "status": "running"}
