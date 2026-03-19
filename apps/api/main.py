"""Federated LLMOps Platform API - FastAPI application."""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.dependencies import close_pool, get_settings, init_pool
from api.middleware import RateLimitMiddleware, RequestIDMiddleware
from api.routes import chat, documents, health, metrics, prompts


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup: init Postgres pool. Shutdown: close pool."""
    settings = get_settings()
    await init_pool(settings)
    yield
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
