"""Server-Sent Events plumbing shared by every streaming endpoint."""
import asyncio
import json
import logging
import queue
import threading
from collections.abc import Callable, Iterable

from fastapi import Request
from fastapi.responses import StreamingResponse

log = logging.getLogger("asf_cleaner")


def sse(event: str, data) -> str:
    return f"event: {event}\ndata: {json.dumps(data, separators=(',', ':'))}\n\n"


def stream_response(request: Request, make_events: Callable[[threading.Event], Iterable], release: Callable[[], None]) -> StreamingResponse:
    """Run `make_events(cancel)` in a thread and stream its (event, payload) pairs as SSE.

    `release` is called when the job ends (it frees the caller's job slot). Closing the connection
    sets `cancel`, so work nobody will see stops.
    """
    q: queue.Queue = queue.Queue()
    cancel = threading.Event()

    def producer():
        try:
            for ev in make_events(cancel):
                if cancel.is_set():
                    break
                q.put(ev)
        except Exception as err:  # noqa: BLE001 - reported to the client as a fatal event
            log.exception("stream job failed")
            q.put(("fatal", {"message": f"{type(err).__name__}: {err}"}))
        finally:
            q.put(None)
            release()

    threading.Thread(target=producer, daemon=True).start()

    async def body():
        try:
            while True:
                try:
                    item = await asyncio.to_thread(q.get, True, 1.0)
                except queue.Empty:
                    if await request.is_disconnected():
                        break
                    yield ": keep-alive\n\n"
                    continue
                if item is None:
                    break
                yield sse(*item)
        finally:
            cancel.set()  # client left: stop fetching what nobody will see

    return StreamingResponse(body(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})
