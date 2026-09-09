"""Process-local async HTTP client pooling.

Celery tasks run on a persistent event loop in each worker process. Keeping one
client pool per loop allows httpx to reuse TCP/TLS connections across jobs,
while the loop key prevents an AsyncClient from being used on the wrong loop.
"""

from __future__ import annotations

import asyncio
from weakref import WeakKeyDictionary

import httpx

# Event loops are weak keys so test loops and short-lived loops do not retain
# their clients forever. Production Celery processes use one persistent loop.
_CLIENT_POOLS: WeakKeyDictionary[asyncio.AbstractEventLoop, dict[tuple, httpx.AsyncClient]] = (
    WeakKeyDictionary()
)


def get_shared_http_client(
    *, timeout: float | None = None, transport: httpx.AsyncBaseTransport | None = None
) -> httpx.AsyncClient:
    """Return a connection-pooled client for the current event loop.

    A custom transport is part of the key because tests and callers can use
    isolated transports. The AsyncClient class identity is also part of the
    key so monkeypatched test clients are never mixed with real clients.
    """
    loop = asyncio.get_running_loop()
    clients = _CLIENT_POOLS.setdefault(loop, {})
    key = (timeout, id(transport), id(httpx.AsyncClient))
    if key not in clients:
        if timeout is None:
            client = httpx.AsyncClient(transport=transport) if transport is not None else httpx.AsyncClient()
        else:
            client = httpx.AsyncClient(timeout=timeout, transport=transport)
        clients[key] = client
    return clients[key]


async def close_shared_http_clients(loop: asyncio.AbstractEventLoop | None = None) -> None:
    """Close pooled clients for ``loop``; intended for worker shutdown."""
    loop = loop or asyncio.get_running_loop()
    clients = _CLIENT_POOLS.pop(loop, {})
    for client in clients.values():
        await client.aclose()
