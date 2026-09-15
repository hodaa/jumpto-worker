"""Worker-process asyncio event-loop lifecycle for async services.

A Celery prefork process executes tasks serially, so one persistent loop per
process lets async clients and their connection pools survive task boundaries.
"""

import asyncio
import atexit
import os

from app.client.http import close_shared_http_clients

_EVENT_LOOP: asyncio.AbstractEventLoop | None = None
_EVENT_LOOP_PID: int | None = None


def run_async(coro):
    """Run a coroutine on the worker-process event loop."""
    global _EVENT_LOOP, _EVENT_LOOP_PID
    pid = os.getpid()
    if _EVENT_LOOP is None or _EVENT_LOOP.is_closed() or pid != _EVENT_LOOP_PID:
        _EVENT_LOOP = asyncio.new_event_loop()
        asyncio.set_event_loop(_EVENT_LOOP)
        _EVENT_LOOP_PID = pid
    return _EVENT_LOOP.run_until_complete(coro)


def close_worker_event_loop() -> None:
    """Close pooled async clients and the persistent worker loop at exit."""
    global _EVENT_LOOP
    if _EVENT_LOOP is None or _EVENT_LOOP.is_closed():
        return
    _EVENT_LOOP.run_until_complete(close_shared_http_clients(_EVENT_LOOP))
    _EVENT_LOOP.close()
    _EVENT_LOOP = None


atexit.register(close_worker_event_loop)
