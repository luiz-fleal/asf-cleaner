"""HTTP API: clean SAR backscatter at a point, fetched from ASF on demand and streamed back.

Nothing is precomputed. A request searches ASF, reads only the needed pixels of each scene
(HTTP range reads) and streams the cleaned rows as Server-Sent Events as every scene finishes,
so a client can draw the series while it is still being built. Scenes already processed are
served from the same cache the CLI uses. The Earthdata token stays on the server.

    uv run asf-api                       # http://127.0.0.1:8000  (docs at /docs)
    curl -N 'http://127.0.0.1:8000/series/stream?lon=-60.0217&lat=-3.119&start=2026-06-01'
"""
import argparse
import asyncio
import json
import logging
import os
import queue
import threading
import time
from datetime import timedelta
from functools import partial

import pandas as pd
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from . import cache
from .config import EXECUTORS, PROCESSING_VERSION, get_token, resolve_dataset, resolve_range
from .harmonize import harmonize
from .runner import iter_parallel
from .search import search
from .workers import process_nisar, process_opera

log = logging.getLogger("asf_cleaner")

WORKERS = {"NISAR": process_nisar, "OPERA": process_opera}
CACHE_DIR = os.environ.get("ASF_CACHE_DIR", "out")
MAX_RANGE = timedelta(days=366 * 12)
MAX_JOBS = int(os.environ.get("ASF_API_MAX_JOBS", "4"))
_jobs = threading.BoundedSemaphore(MAX_JOBS)

app = FastAPI(title="asf-cleaner API", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.environ.get("ASF_API_CORS", "http://localhost:5173,http://127.0.0.1:5173").split(","),
    allow_methods=["GET"],
    allow_headers=["*"],
)


def _num(v):
    return None if pd.isna(v) else float(v)


def rows_to_json(df: pd.DataFrame) -> list[dict]:
    """Cleaned rows in the shape clients plot: time in epoch ms, backscatter in dB."""
    return [
        {
            "scene": r.scene,
            "sensor": r.sensor,
            "band": r.band,
            "platform": r.platform,
            "t": int(r.start_time.timestamp() * 1000),
            "burst": r.burst_id,
            "track": None if pd.isna(r.track) else int(r.track),
            "direction": r.direction,
            "pol": r.polarization,
            "polClass": r.pol_class,
            "db": _num(r.backscatter_db_median),
            "dbMean": _num(r.backscatter_db_mean),
            "dbCenter": _num(r.backscatter_db_center),
            "valid": _num(r.valid_fraction),
        }
        for r in df.itertuples()
    ]


def events(lon, lat, start_s, end_s, dataset_req, window_m, max_results, workers, use_cache, cancel):
    """Yield (event, payload) pairs: meta, rows (cache first, then one per scene), scene_error, done."""
    t0 = time.perf_counter()
    start, end = resolve_range(start_s, end_s)
    if end - start > MAX_RANGE:
        raise ValueError("date range is longer than 12 years")
    dataset, why = resolve_dataset(dataset_req, start, end)
    scenes = search(dataset, lon, lat, start, end, max_results)

    cache_file = cache.cache_path(CACHE_DIR, dataset, lon, lat, window_m)
    cached = cache.load(cache_file, PROCESSING_VERSION) if use_cache else pd.DataFrame()
    names = {sc["scene"] for sc in scenes}
    have = set(cached["scene"]) & names if len(cached) else set()
    todo = [sc for sc in scenes if sc["scene"] not in have]
    yield "meta", {
        "dataset": dataset, "reason": why, "lon": lon, "lat": lat, "windowM": window_m,
        "start": start.isoformat(), "end": end.isoformat(),
        "total": len(scenes), "cached": len(have), "toFetch": len(todo),
    }
    if have:
        yield "rows", {"source": "cache", "done": len(have), "total": len(scenes),
                       "rows": rows_to_json(harmonize(cached[cached["scene"].isin(have)]))}

    new_rows, done, failed = [], len(have), 0
    try:
        if todo:
            fn = partial(WORKERS[dataset], lon=lon, lat=lat, window_m=window_m, token=get_token())
            for res in iter_parallel(fn, todo, workers, EXECUTORS[dataset], cancel):
                done += 1
                if res.error:
                    failed += 1
                    yield "scene_error", {"scene": res.item["scene"], "message": res.error, "done": done}
                    continue
                new_rows.extend(res.rows)
                yield "rows", {"source": "live", "done": done, "total": len(scenes),
                               "rows": rows_to_json(harmonize(pd.DataFrame(res.rows)))}
    finally:
        if use_cache and new_rows:  # also on cancel, so partial progress is not lost
            cache.save(cache_file, cached, new_rows)
    yield "done", {"fetched": done - len(have) - failed, "failed": failed, "seconds": round(time.perf_counter() - t0, 1)}


def sse(event: str, data) -> str:
    return f"event: {event}\ndata: {json.dumps(data, separators=(',', ':'))}\n\n"


@app.get("/health")
def health():
    try:
        get_token()
        token = True
    except RuntimeError:
        token = False
    return {"status": "ok", "tokenConfigured": token, "maxJobs": MAX_JOBS}


@app.get("/series/stream")
async def series_stream(
    request: Request,
    lon: float = Query(..., ge=-180, le=180, description="Longitude, negative west of Greenwich"),
    lat: float = Query(..., ge=-90, le=90),
    start: str | None = Query(None, description="e.g. 2026-06-01 (default: 180 days before end)"),
    end: str | None = Query(None, description="e.g. 2026-11-13 (default: now)"),
    dataset: str = Query("auto", pattern="^(auto|NISAR|OPERA)$"),
    window_m: float = Query(500, ge=30, le=5000),
    max_results: int = Query(400, ge=1, le=1000),
    workers: int = Query(8, ge=1, le=16),
    cache_enabled: bool = Query(True, alias="cache"),
):
    """Server-Sent Events: `meta`, then `rows` as scenes finish, `scene_error`, and finally `done`."""
    try:  # validate dates now so bad input is an HTTP 400, not a broken stream
        s, e = resolve_range(start, end)
        resolve_dataset(dataset, s, e)
    except ValueError as err:
        raise HTTPException(400, str(err)) from err
    if not _jobs.acquire(blocking=False):
        raise HTTPException(503, "server is busy with other requests; try again shortly")

    q: queue.Queue = queue.Queue()
    cancel = threading.Event()

    def producer():
        try:
            for ev in events(lon, lat, start, end, dataset, window_m, max_results, workers, cache_enabled, cancel):
                if cancel.is_set():
                    break
                q.put(ev)
        except Exception as err:  # noqa: BLE001 - reported to the client as a fatal event
            log.exception("series job failed")
            q.put(("fatal", {"message": f"{type(err).__name__}: {err}"}))
        finally:
            q.put(None)
            _jobs.release()

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
            cancel.set()  # client left: stop fetching scenes nobody will see

    return StreamingResponse(body(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.get("/series")
def series(
    lon: float = Query(..., ge=-180, le=180),
    lat: float = Query(..., ge=-90, le=90),
    start: str | None = None,
    end: str | None = None,
    dataset: str = Query("auto", pattern="^(auto|NISAR|OPERA)$"),
    window_m: float = Query(500, ge=30, le=5000),
    max_results: int = Query(400, ge=1, le=1000),
    workers: int = Query(8, ge=1, le=16),
    cache_enabled: bool = Query(True, alias="cache"),
):
    """Same data as /series/stream, returned as one JSON document once every scene is done."""
    try:
        s, e = resolve_range(start, end)
        resolve_dataset(dataset, s, e)
    except ValueError as err:
        raise HTTPException(400, str(err)) from err
    if not _jobs.acquire(blocking=False):
        raise HTTPException(503, "server is busy with other requests; try again shortly")
    meta, rows, errors, done = {}, [], [], {}
    try:
        for ev, data in events(lon, lat, start, end, dataset, window_m, max_results, workers, cache_enabled, threading.Event()):
            if ev == "meta":
                meta = data
            elif ev == "rows":
                rows.extend(data["rows"])
            elif ev == "scene_error":
                errors.append(data)
            elif ev == "done":
                done = data
    except (RuntimeError, ValueError) as err:
        raise HTTPException(502, str(err)) from err
    finally:
        _jobs.release()
    return {"meta": meta, "rows": rows, "errors": errors, **done}


def main(argv=None):
    import uvicorn

    ap = argparse.ArgumentParser(description="Serve the asf-cleaner HTTP API")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    uvicorn.run(app, host=a.host, port=a.port)


if __name__ == "__main__":
    main()
