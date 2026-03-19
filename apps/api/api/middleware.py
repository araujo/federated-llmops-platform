"""Middleware: request ID, Prometheus metrics, rate limiting."""

import time
import uuid
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

from api.dependencies import get_settings
from api.metrics_registry import REQUEST_COUNT, REQUEST_LATENCY

# In-memory rate limit: {client_ip: (count, window_start)}
_rate_limit_store: dict[str, tuple[int, float]] = {}


def _get_client_ip(request: Request) -> str:
    """Get client IP from X-Forwarded-For or direct client."""
    forwarded = request.headers.get("X-Forwarded-For")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


def _check_rate_limit(client_ip: str, limit: int) -> bool:
    """Return True if under limit, False if exceeded."""
    now = time.time()
    count, window_start = _rate_limit_store.get(client_ip, (0, 0.0))
    if now - window_start >= 60:  # New window
        _rate_limit_store[client_ip] = (1, now)
        return True
    if count >= limit:
        return False
    _rate_limit_store[client_ip] = (count + 1, window_start)
    return True


class RequestIDMiddleware(BaseHTTPMiddleware):
    """Add X-Request-ID header and record Prometheus metrics."""

    async def dispatch(self, request: Request, call_next) -> Response:
        request_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())
        request.state.request_id = request_id

        start = time.perf_counter()
        response = await call_next(request)
        duration = time.perf_counter() - start

        path = request.scope.get("path", "")
        method = request.scope.get("method", "")
        status = str(response.status_code)
        REQUEST_COUNT.labels(method=method, path=path, status=status).inc()
        REQUEST_LATENCY.labels(method=method, path=path).observe(duration)

        response.headers["X-Request-ID"] = request_id
        return response


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Rate limit by client IP when RATE_LIMIT_PER_MINUTE is set."""

    async def dispatch(self, request: Request, call_next) -> Response:
        settings = get_settings()
        if settings.rate_limit_per_minute <= 0:
            return await call_next(request)

        client_ip = _get_client_ip(request)
        if not _check_rate_limit(client_ip, settings.rate_limit_per_minute):
            return JSONResponse(
                status_code=429,
                content={"detail": "Rate limit exceeded. Try again later."},
                headers={"Retry-After": "60"},
            )
        return await call_next(request)
