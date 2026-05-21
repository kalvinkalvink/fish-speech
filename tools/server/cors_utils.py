"""CORS helpers for browser and Chrome extension clients."""

from __future__ import annotations

import re
from re import Pattern
from typing import Any, Callable

from kui.asgi import HttpResponse, request
from kui.asgi.cors import allow_cors
from kui.asgi.routing import AsyncViewType
from kui.cors import CORSConfig

# Local dev, web pages, and chrome-extension:// origins.
CORS_ORIGIN_PATTERNS: tuple[Pattern[str], ...] = (
    re.compile(r".*"),
)

CORS_ALLOW_HEADERS = (
    "Authorization",
    "Content-Type",
    "Accept",
    "Accept-Language",
    "Content-Language",
    "X-Requested-With",
)


def build_cors_config() -> CORSConfig:
    """Permissive CORS for local API use (WebUI, curl, Chrome extensions)."""
    return CORSConfig(
        allow_origins=list(CORS_ORIGIN_PATTERNS),
        allow_methods=["GET", "POST", "DELETE", "OPTIONS", "HEAD"],
        allow_headers=CORS_ALLOW_HEADERS,
        expose_headers=["Content-Disposition", "Content-Type"],
        allow_credentials=True,
        max_age=86400,
    )


def _origin_allowed(origin: str) -> bool:
    return any(pattern.fullmatch(origin) for pattern in CORS_ORIGIN_PATTERNS)


def apply_cors_to_response(response: HttpResponse) -> HttpResponse:
    """Add CORS headers to responses (including errors the CORS middleware skips)."""
    origin = request.headers.get("origin")
    if not origin or not _origin_allowed(origin):
        return response

    response.headers["Access-Control-Allow-Origin"] = origin
    response.headers["Access-Control-Allow-Credentials"] = "true"
    # Chrome extensions / public sites calling http://127.0.0.1 (Private Network Access).
    response.headers["Access-Control-Allow-Private-Network"] = "true"
    response.headers["Vary"] = "Origin"
    return response


def browser_cors_middleware(endpoint: AsyncViewType) -> AsyncViewType:
    """
    CORS for browsers and Chrome extensions, including Private Network Access.

    Wraps Kui's allow_cors and adds Access-Control-Allow-Private-Network on every
    cross-origin response (including OPTIONS preflight).
    """
    cors_wrapped = allow_cors(**build_cors_config())(endpoint)

    async def wrapper() -> Any:
        from kui.asgi.responses import convert_response

        response = convert_response(await cors_wrapped())
        return apply_cors_to_response(response)

    return wrapper  # type: ignore[return-value]


def private_network_cors_middleware(endpoint: Callable) -> Callable:
    """Deprecated alias; use browser_cors_middleware via Kui http_middlewares."""
    return browser_cors_middleware(endpoint)
