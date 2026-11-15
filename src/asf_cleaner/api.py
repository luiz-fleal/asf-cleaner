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
from datetime import date, timedelta
from functools import partial
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from . import assistant, cache, history, imerg, optical, regions
from . import forecast as fc
from . import risk
from .memo import TTLCache
from .sse import stream_response
from .config import EXECUTORS, PROCESSING_VERSION, get_token, resolve_dataset, resolve_range
from .harmonize import harmonize
from .runner import iter_parallel
from .search import search
from .workers import process_nisar, process_opera

log = logging.getLogger("asf_cleaner")

WORKERS = {"NISAR": process_nisar, "OPERA": process_opera}
CACHE_DIR = os.environ.get("ASF_CACHE_DIR", "out")
MAX_RANGE = timedelta(days=366 * 12)
MAX_JOBS = int(os.environ.get("ASF_API_MAX_JOBS", "6"))
# Separate job slots per kind of work: a slow NISAR read (a minute and a half per scene) must never starve the quick
# rain requests or the forecast, which used to answer 503 because the radar had taken every slot.
_jobs = threading.BoundedSemaphore(MAX_JOBS)  # radar series
_rain_jobs = threading.BoundedSemaphore(MAX_JOBS)  # rain
_forecast_jobs = threading.BoundedSemaphore(int(os.environ.get("ASF_API_MAX_FORECASTS", "2")))

app = FastAPI(title="asf-cleaner API", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.environ.get("ASF_API_CORS", "http://localhost:5173,http://127.0.0.1:5173").split(","),
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


_search_cache = TTLCache(256)


def cached_search(dataset, lon, lat, start, end, max_results):
    """ASF search with a short memory: the page asks for neighbouring date chunks of the same point
    again and again, and a search costs a second or more. Ranges that ended long ago never change."""
    key = (dataset, round(lon, 4), round(lat, 4), start, end, max_results)
    ttl = 3600 if end < pd.Timestamp.now(tz="UTC").to_pydatetime() - timedelta(days=2) else 300
    hit = _search_cache.get(key, ttl)
    if hit is not None:
        return list(hit)
    scenes = search(dataset, lon, lat, start, end, max_results)
    _search_cache.put(key, scenes)
    return list(scenes)


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
    scenes = cached_search(dataset, lon, lat, start, end, max_results)

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
            for res in iter_parallel(fn, todo, workers, EXECUTORS[dataset], cancel, shared=True):
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


class AskBody(BaseModel):
    question: str = Field(max_length=assistant.MAX_QUESTION)
    lang: str = "en"
    context: dict | None = None


@app.post("/assistant")
def ask_assistant(body: AskBody):
    """Open questions from the map's AI bar. Commands never come here; the page handles those itself."""
    if not assistant.configured():
        return {"configured": False, "answer": None}
    try:
        return {"configured": True, "answer": assistant.ask(body.question, body.lang, body.context)}
    except Exception as exc:  # network, quota, bad key: the page shows a calm message, not a stack trace
        log.warning("assistant failed: %s", exc)
        raise HTTPException(status_code=502, detail="The assistant could not answer now.")


@app.get("/health")
def health():
    try:
        get_token()
        token = True
    except RuntimeError:
        token = False
    return {"status": "ok", "tokenConfigured": token, "maxJobs": MAX_JOBS, "maxForecasts": _forecast_jobs._initial_value}


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

    return stream_response(request, lambda cancel: events(lon, lat, start, end, dataset, window_m, max_results, workers, cache_enabled, cancel), _jobs.release)


# ---- rain (NASA GPM IMERG, daily, 0.1 degree) ----------------------------------------------
def rain_row(r: dict) -> dict:
    mm = r["precipitation_mm_day"]
    return {"date": r["date"], "t": int(pd.Timestamp(r["date"], tz="UTC").timestamp() * 1000),
            "mm": None if pd.isna(mm) else round(float(mm), 3), "run": r["imerg_run"]}


def rain_events(lon, lat, start_s, end_s, workers, use_cache, cancel):
    """Yield (event, payload): meta, rows (cache first, then batches as days finish), done.

    One tiny OPeNDAP request per day (a single grid cell), so nothing is downloaded. Cached per grid
    cell, so every point inside the same 0.1 degree cell shares one cache file.
    """
    t0 = time.perf_counter()
    start, end = resolve_range(start_s, end_s)
    if end - start > MAX_RANGE:
        raise ValueError("date range is longer than 12 years")
    d0, d1 = start.date(), min(end.date(), date.today())
    i, j = imerg.grid_index(lon, lat)
    days, missing = imerg.plan_days(d0, d1)

    path = Path(CACHE_DIR) / ".cache" / f"imerg_cell_{i}_{j}.parquet"
    cached = cache.load(path, PROCESSING_VERSION) if use_cache else pd.DataFrame()
    have = dict(zip(cached["date"], cached["imerg_run"])) if len(cached) else {}
    wanted = {d["date"].isoformat() for d in days}
    todo = [d for d in days if d["date"].isoformat() not in have
            or (d["run"] == "final" and have[d["date"].isoformat()] == "late")]
    todo.sort(key=lambda d: d["date"], reverse=True)  # newest first: the days people look at fill in first
    n_cached = len(wanted & set(have)) - len([d for d in todo if d["date"].isoformat() in have])
    yield "meta", {"lon": lon, "lat": lat, "gridLon": round(imerg.LON0 + i * imerg.STEP, 2),
                   "gridLat": round(imerg.LAT0 + j * imerg.STEP, 2), "start": d0.isoformat(), "end": d1.isoformat(),
                   "total": len(days), "cached": n_cached, "toFetch": len(todo), "missing": len(missing)}
    if n_cached:
        hit = cached[cached["date"].isin(wanted - {d["date"].isoformat() for d in todo})]
        yield "rows", {"source": "cache", "done": n_cached, "total": len(days),
                       "rows": [rain_row(r) for r in hit.to_dict("records")]}

    new_rows, done, failed, buf, last = [], n_cached, 0, [], time.perf_counter()
    try:
        if todo:
            fn = partial(imerg.fetch_point, i=i, j=j, token=get_token())
            try:  # fail fast on a rejected token instead of failing every day
                first = fn(todo[0])
            except PermissionError as err:
                raise RuntimeError(str(err)) from err
            new_rows += first
            buf += first
            done += 1
            yield "rows", {"source": "live", "done": done, "total": len(days), "rows": [rain_row(r) for r in buf]}
            buf = []
            for res in iter_parallel(fn, todo[1:], workers, "thread", cancel):
                done += 1
                if res.error:
                    failed += 1
                else:
                    new_rows.extend(res.rows)
                    buf.extend(res.rows)
                if buf and (len(buf) >= 12 or time.perf_counter() - last > 0.4):
                    yield "rows", {"source": "live", "done": done, "total": len(days), "rows": [rain_row(r) for r in buf]}
                    buf, last = [], time.perf_counter()
            if buf:
                yield "rows", {"source": "live", "done": done, "total": len(days), "rows": [rain_row(r) for r in buf]}
    finally:
        if use_cache and new_rows:
            cache.save(path, cached, new_rows, key=["date"])
    yield "done", {"fetched": len(new_rows), "failed": failed, "seconds": round(time.perf_counter() - t0, 1)}


@app.get("/rain/stream")
async def rain_stream(
    request: Request,
    lon: float = Query(..., ge=-180, le=180),
    lat: float = Query(..., ge=-60, le=60, description="IMERG covers 60S to 60N"),
    start: str | None = Query(None, description="default: 180 days before end"),
    end: str | None = Query(None, description="default: now"),
    workers: int = Query(8, ge=1, le=16),
    cache_enabled: bool = Query(True, alias="cache"),
):
    """Server-Sent Events: `meta`, `rows` ({date, t, mm, run}) as days arrive, then `done` (or `fatal`)."""
    try:
        resolve_range(start, end)
    except ValueError as err:
        raise HTTPException(400, str(err)) from err
    if not _rain_jobs.acquire(blocking=False):
        raise HTTPException(503, "server is busy with other requests; try again shortly")
    return stream_response(request, lambda cancel: rain_events(lon, lat, start, end, workers, cache_enabled, cancel), _rain_jobs.release)


@app.get("/rain")
def rain(
    lon: float = Query(..., ge=-180, le=180),
    lat: float = Query(..., ge=-60, le=60),
    start: str | None = None,
    end: str | None = None,
    workers: int = Query(8, ge=1, le=16),
    cache_enabled: bool = Query(True, alias="cache"),
):
    """Same data as /rain/stream, returned as one JSON document once every day is done."""
    try:
        resolve_range(start, end)
    except ValueError as err:
        raise HTTPException(400, str(err)) from err
    if not _rain_jobs.acquire(blocking=False):
        raise HTTPException(503, "server is busy with other requests; try again shortly")
    meta, rows, done = {}, [], {}
    try:
        for ev, data in rain_events(lon, lat, start, end, workers, cache_enabled, threading.Event()):
            if ev == "meta":
                meta = data
            elif ev == "rows":
                rows.extend(data["rows"])
            elif ev == "done":
                done = data
    except (RuntimeError, ValueError) as err:
        raise HTTPException(502, str(err)) from err
    finally:
        _rain_jobs.release()
    rows.sort(key=lambda r: r["t"])
    return {"meta": meta, "rows": rows, **done}


# ---- forecast to 2030 -----------------------------------------------------------------------
_forecast_cache = TTLCache(64)


WINDOWS = (500, 1000, 2000, 5000)


def forecast_events(lon, lat, until, rain_scale, workers, use_cache, cancel, window_m=500):
    """Yield (event, payload): meta, progress (while the history is collected), result, done (or fatal).

    The history is collected on demand (and cached), so the first request for a point takes a while and
    reports its progress; the trained result is remembered for an hour.
    """
    t0 = time.perf_counter()
    window_m = min(WINDOWS, key=lambda w: abs(w - window_m))  # a few sizes, so areas of similar size share one history
    lon, lat = round(lon, 2), round(lat, 2)  # about 1 km: nearby clicks share one history instead of each reading years of scenes
    yield "meta", {"lon": lon, "lat": lat, "until": until, "rainScale": rain_scale, "windowM": window_m}
    key = (lon, lat, until, rain_scale, window_m)
    hit = _forecast_cache.get(key, 3600) if use_cache else None
    if hit is not None:
        yield "result", hit
        yield "done", {"seconds": round(time.perf_counter() - t0, 1), "cached": True}
        return

    radar, geometry, scenes, rain_series, notes = None, {}, 0, None, []
    for ev, payload in history.radar_history(lon, lat, search_fn=cached_search, cache_dir=CACHE_DIR, workers=workers, use_cache=use_cache, cancel=cancel, window_m=window_m):
        if ev == "progress":
            yield ev, payload
        else:
            radar, geometry, scenes = payload
    if cancel.is_set():
        return
    try:
        for ev, payload in history.rain_history(lon, lat, cache_dir=CACHE_DIR, workers=workers, use_cache=use_cache, cancel=cancel):
            if ev == "progress":
                yield ev, payload
            else:
                rain_series = payload
    except RuntimeError as err:  # rain access (GES DISC) not authorized: the radar forecast still works
        notes.append(f"Rain was left out: {err}")
    if cancel.is_set():
        return
    yield "progress", {"phase": "model", "done": 0, "total": 1}
    try:
        result = fc.run_forecast(radar, rain_series, until, rain_scale)
    except fc.InsufficientHistory as err:
        yield "fatal", {"message": f"Not enough history to forecast: {err}"}
        return
    result["notes"] = notes + result["notes"]
    try:  # risks are an extra: a failure here must never cost the forecast
        monthly = fc.monthly_backscatter(radar) if radar is not None and len(radar) else {}
        try:  # the color of the ground helps to recognise the land cover; without it the radar-only model is used
            ground = optical.features(lon, lat)
        except Exception as err:  # noqa: BLE001
            log.warning("no optical features for %s, %s: %s", lon, lat, err)
            ground = None
        result["risk"] = risk.assess(monthly, rain_series, result, optical=ground)
    except Exception as err:  # noqa: BLE001
        log.exception("risk assessment failed")
        result["risk"] = {"available": False, "reason": f"{type(err).__name__}: {err}"}
    result["meta"] = {"lon": lon, "lat": lat, "geometry": geometry, "radarScenes": scenes,
                      "rainMonths": 0 if rain_series is None else int(len(rain_series)),
                      "radar": "OPERA Sentinel-1 C-band, one scene per month", "rain": "NASA GPM IMERG, monthly"}
    _forecast_cache.put(key, result)
    yield "progress", {"phase": "model", "done": 1, "total": 1}
    yield "result", result
    yield "done", {"seconds": round(time.perf_counter() - t0, 1), "cached": False}


def forecast_result(lon: float, lat: float, until: str = "2030-12", rain_scale: float = 1.0, workers: int = 8, use_cache: bool = True) -> dict:
    """The forecast of a place as one result (the same events as /forecast/stream, collected). Used to pre-build regions."""
    result = None
    for ev, data in forecast_events(lon, lat, until, rain_scale, workers, use_cache, threading.Event()):
        if ev == "result":
            result = data
        elif ev == "fatal":
            raise RuntimeError(data["message"])
    if result is None:
        raise RuntimeError("no result")
    return result


_UNTIL = r"^\d{4}-(0[1-9]|1[0-2])$"


@app.get("/forecast/stream")
async def forecast_stream(
    request: Request,
    lon: float = Query(..., ge=-180, le=180),
    lat: float = Query(..., ge=-60, le=60, description="IMERG rain covers 60S to 60N"),
    until: str = Query("2030-12", pattern=_UNTIL),
    rain_scale: float = Query(1.0, ge=0.5, le=1.5, description="what-if: rain compared with normal"),
    window_m: float = Query(500, ge=30, le=5000, description="radar window around the point, for areas (snapped to 500, 1000, 2000 or 5000 m)"),
    workers: int = Query(8, ge=1, le=16),
    cache_enabled: bool = Query(True, alias="cache"),
):
    """Server-Sent Events: `meta`, `progress` ({phase, done, total}) while the history is collected, `result`, `done`."""
    if not _forecast_jobs.acquire(blocking=False):
        raise HTTPException(503, "server is busy with other requests; try again shortly")
    return stream_response(request, lambda cancel: forecast_events(lon, lat, until, rain_scale, workers, cache_enabled, cancel, window_m), _forecast_jobs.release)


@app.get("/regions")
def list_regions():
    """The ready-made regional forecasts and whether each one has been built."""
    return regions.listing()


@app.get("/regions/{region_id}")
def get_region(region_id: str):
    """A ready-made regional forecast, shaped like /forecast plus `samples` and `region`."""
    if region_id not in regions.REGIONS:
        raise HTTPException(404, f"unknown region {region_id}")
    data = regions.load(region_id)
    if data is None:
        raise HTTPException(404, f"region {region_id} has not been built yet: uv run python -m asf_cleaner.regions build {region_id}")
    return data


@app.get("/forecast")
def forecast(
    lon: float = Query(..., ge=-180, le=180),
    lat: float = Query(..., ge=-60, le=60),
    until: str = Query("2030-12", pattern=_UNTIL),
    rain_scale: float = Query(1.0, ge=0.5, le=1.5),
    window_m: float = Query(500, ge=30, le=5000),
    workers: int = Query(8, ge=1, le=16),
    cache_enabled: bool = Query(True, alias="cache"),
):
    """Same forecast as /forecast/stream, returned as one JSON document."""
    if not _forecast_jobs.acquire(blocking=False):
        raise HTTPException(503, "server is busy with other requests; try again shortly")
    result = None
    try:
        for ev, data in forecast_events(lon, lat, until, rain_scale, workers, cache_enabled, threading.Event(), window_m):
            if ev == "result":
                result = data
            elif ev == "fatal":
                raise HTTPException(422, data["message"])
    except RuntimeError as err:
        raise HTTPException(502, str(err)) from err
    finally:
        _forecast_jobs.release()
    return result


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
