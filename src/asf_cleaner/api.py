"""HTTP API: clean SAR backscatter at a point, fetched from ASF on demand and streamed back.

Nothing is precomputed. A request searches ASF, reads only the needed pixels of each scene
(HTTP range reads) and streams the cleaned rows as Server-Sent Events as every scene finishes,
so a client can draw the series while it is still being built. Scenes already processed are
served from the same cache the CLI uses. The Earthdata token stays on the server.

NASA IMERG daily rain for the same point and range is fetched automatically alongside the SAR data
(unless `rain=false`) and streamed as `rain` events, then tied to each scene in a final `sceneRain`
event. A rain failure never affects the SAR stream (it is reported as a `rain_error` event).

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
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from functools import partial

import pandas as pd
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from . import cache, imerg
from .config import EXECUTORS, PROCESSING_VERSION, get_token, resolve_dataset, resolve_range, snap_point
from .harmonize import harmonize
from .runner import iter_parallel, warm
from .search import search
from .workers import process_nisar, process_opera

log = logging.getLogger("asf_cleaner")

WORKERS = {"NISAR": process_nisar, "OPERA": process_opera}
CACHE_DIR = os.environ.get("ASF_CACHE_DIR", "out")
MAX_RANGE = timedelta(days=366 * 12)
MAX_JOBS = int(os.environ.get("ASF_API_MAX_JOBS", "4"))
_jobs = threading.BoundedSemaphore(MAX_JOBS)

WARM_PROCESSES = int(os.environ.get("ASF_WARM_PROCESSES", "2"))  # 0 = start workers on first use


@asynccontextmanager
async def lifespan(_app):
    """Start a couple of NISAR pool processes now so the first request does not wait for them to
    spawn and import h5py/rasterio. Done in a thread: it must not delay the server coming up."""
    if WARM_PROCESSES > 0:
        threading.Thread(target=warm, args=("process", "default", WARM_PROCESSES), daemon=True).start()
    yield


app = FastAPI(title="asf-cleaner API", version="0.1.0", lifespan=lifespan)
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


def _day_ms(iso: str) -> int:
    """Epoch milliseconds of the start of a UTC day given as YYYY-MM-DD."""
    return int(datetime.fromisoformat(iso).replace(tzinfo=timezone.utc).timestamp() * 1000)


def rain_to_json(rows: list[dict]) -> list[dict]:
    """Daily IMERG rows as clients plot them: day in epoch ms (UTC), rain in mm/day."""
    return [{"t": _day_ms(r["date"]), "mm": _num(r["precipitation_mm_day"]), "run": r.get("imerg_run")}
            for r in rows]


def sar_events(lon, lat, start, end, dataset, why, window_m, max_results, workers, use_cache, cancel,
               rain, stats):
    """Yield (event, payload): meta, rows (cache first, then one per scene), scene_error."""
    scenes = search(dataset, lon, lat, start, end, max_results)

    cache_file = cache.cache_path(CACHE_DIR, dataset, lon, lat, window_m)
    cached = cache.load(cache_file, PROCESSING_VERSION) if use_cache else pd.DataFrame()
    names = {sc["scene"] for sc in scenes}
    have = set(cached["scene"]) & names if len(cached) else set()
    todo = [sc for sc in scenes if sc["scene"] not in have]
    yield "meta", {
        "dataset": dataset, "reason": why, "lon": lon, "lat": lat, "windowM": window_m,
        "start": start.isoformat(), "end": end.isoformat(), "rain": rain,
        "total": len(scenes), "cached": len(have), "toFetch": len(todo),
    }
    if have:
        hit = cached[cached["scene"].isin(have)]
        stats["scene_times"].update(zip(hit["scene"], hit["start_time"]))
        yield "rows", {"source": "cache", "done": len(have), "total": len(scenes),
                       "rows": rows_to_json(harmonize(hit))}

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
                stats["scene_times"].update((r["scene"], r["start_time"]) for r in res.rows)
                yield "rows", {"source": "live", "done": done, "total": len(scenes),
                               "rows": rows_to_json(harmonize(pd.DataFrame(res.rows)))}
    finally:
        if use_cache and new_rows:  # also on cancel, so partial progress is not lost
            cache.save(cache_file, cached, new_rows)
    stats["fetched"], stats["failed"] = done - len(have) - failed, failed


def rain_events(lon, lat, start, end, workers, use_cache, cancel, stats):
    """Yield (event, payload): rain (cached days, then fetched days in small batches), rain_error."""
    pending, last = [], time.monotonic()
    meta = {"done": 0, "total": 0}

    def batch(source):
        out = {"source": source, "done": meta["done"], "total": meta["total"], "rows": rain_to_json(pending)}
        pending.clear()
        return "rain", out

    istats: dict = {}
    try:
        for source, rows, done, total in imerg.iter_daily(
                lon, lat, start, end, token_fn=get_token, workers=workers, out=CACHE_DIR,
                use_cache=use_cache, cancel=cancel, stats=istats):
            stats["rain_rows"].extend(rows)
            meta.update(done=done, total=total)
            pending.extend(rows)
            if source == "cache" or len(pending) >= 40 or time.monotonic() - last > 0.5:
                yield batch(source)
                last = time.monotonic()
        if pending:
            yield batch("live")
    except imerg.ImergAccessError as err:  # one-time Earthdata authorization missing
        if pending:
            yield batch("live")
        yield "rain_error", {"message": str(err)}
    except Exception as err:  # noqa: BLE001 - the SAR data must not depend on rain
        log.exception("IMERG rain failed")
        if pending:
            yield batch("live")
        yield "rain_error", {"message": f"{type(err).__name__}: {err}"}
    finally:
        stats["rain_failed"] = istats.get("failed", 0)


def scene_rain(stats) -> list[dict]:
    """Rain before each scene (acquisition day and the 1, 3, 7 and 30 days before it, mm)."""
    scenes = stats["scene_times"]
    if not scenes or not stats["rain_rows"]:
        return []
    df = pd.DataFrame({"scene": list(scenes), "start_time": pd.to_datetime(list(scenes.values()), utc=True)})
    daily = pd.DataFrame(stats["rain_rows"]).drop_duplicates("date", keep="last")
    out = imerg.rain_features(df, daily)
    return [{"scene": r.scene, "rain0": _num(r.rain_d0_mm), "rain1d": _num(r.rain_1d_mm),
             "rain3d": _num(r.rain_3d_mm), "rain7d": _num(r.rain_7d_mm), "rain30d": _num(r.rain_30d_mm)}
            for r in out.itertuples()]


def events(lon, lat, start_s, end_s, dataset_req, window_m, max_results, workers, use_cache, cancel,
           rain=True):
    """Yield (event, payload) pairs: meta, rows, scene_error, rain, rain_error, sceneRain, done.

    SAR scenes and IMERG rain are fetched at the same time and their events are interleaved as they
    happen. `done` is sent last, after the rain has finished, so a client that closes on `done`
    has received everything. A rain failure is reported but never stops the SAR stream.
    """
    t0 = time.perf_counter()
    lon, lat = snap_point(lon, lat, window_m)  # nearby clicks share one cache file
    start, end = resolve_range(start_s, end_s)
    if end - start > MAX_RANGE:
        raise ValueError("date range is longer than 12 years")
    dataset, why = resolve_dataset(dataset_req, start, end)
    stats = {"fetched": 0, "failed": 0, "scene_times": {}, "rain_rows": [], "rain_failed": 0}

    q: queue.Queue = queue.Queue()
    ended = object()

    def pump(name, gen):
        try:
            for ev in gen:
                q.put((name, ev))
        except Exception as err:  # noqa: BLE001 - re-raised below for SAR, reported for rain
            q.put((name, err))
        finally:
            q.put((name, ended))

    streams = {"sar": sar_events(lon, lat, start, end, dataset, why, window_m, max_results, workers,
                                 use_cache, cancel, rain, stats)}
    if rain:
        streams["rain"] = rain_events(lon, lat, start, end, workers, use_cache, cancel, stats)
    for name, gen in streams.items():
        threading.Thread(target=pump, args=(name, gen), daemon=True).start()

    live, meta_sent, held = set(streams), False, []
    while live:
        try:
            name, item = q.get(timeout=0.5)
        except queue.Empty:
            if cancel.is_set():
                return
            continue
        if item is ended:
            live.discard(name)
            continue
        if isinstance(item, Exception):
            if name == "sar":
                cancel.set()  # stop the rain too; the caller reports the failure
                raise item
            item = ("rain_error", {"message": f"{type(item).__name__}: {item}"})
        if name == "rain" and not meta_sent:
            held.append(item)  # cached rain is instant; `meta` still has to be the first event
            continue
        yield item
        if name == "sar" and item[0] == "meta":
            meta_sent = True
            yield from held
            held.clear()
    yield from held

    if rain:
        rows = scene_rain(stats)
        if rows:
            yield "sceneRain", {"rows": rows}
    done = {"fetched": stats["fetched"], "failed": stats["failed"],
            "seconds": round(time.perf_counter() - t0, 1)}
    if rain:
        done.update(rainDays=len({r["date"] for r in stats["rain_rows"]}), rainFailed=stats["rain_failed"])
    yield "done", done


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
    rain: bool = Query(True, description="also fetch IMERG daily rain (rain, sceneRain events)"),
):
    """Server-Sent Events: `meta`, `rows` as scenes finish, `scene_error`, `rain`, `rain_error`,
    `sceneRain`, and finally `done` (or `fatal`)."""
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
            for ev in events(lon, lat, start, end, dataset, window_m, max_results, workers, cache_enabled, cancel, rain):
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
    rain: bool = Query(True, description="also fetch IMERG daily rain"),
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
    rain_rows, scene_rain_rows, rain_error = [], [], None
    try:
        for ev, data in events(lon, lat, start, end, dataset, window_m, max_results, workers, cache_enabled,
                               threading.Event(), rain):
            if ev == "meta":
                meta = data
            elif ev == "rows":
                rows.extend(data["rows"])
            elif ev == "scene_error":
                errors.append(data)
            elif ev == "rain":
                rain_rows.extend(data["rows"])
            elif ev == "sceneRain":
                scene_rain_rows = data["rows"]
            elif ev == "rain_error":
                rain_error = data["message"]
            elif ev == "done":
                done = data
    except (RuntimeError, ValueError) as err:
        raise HTTPException(502, str(err)) from err
    finally:
        _jobs.release()
    out = {"meta": meta, "rows": rows, "errors": errors, **done}
    if rain:
        out.update(rain=sorted(rain_rows, key=lambda r: r["t"]), sceneRain=scene_rain_rows, rainError=rain_error)
    return out


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
