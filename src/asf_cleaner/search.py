"""ASF search with automatic pagination.

The ASF endpoint has no offset/cursor, but it supports ``output=count`` and returns newest
scenes first. So a time range with more than PAGE_SIZE matches is bisected by date until each
piece fits in one request; pieces are visited newest-first and we stop at ``max_results``.
"""
import logging
from datetime import datetime, timedelta

import requests

from .config import ASF_SEARCH_URL, TIME_FMT

log = logging.getLogger("asf_cleaner")

PAGE_SIZE = 1000
MIN_SPAN = timedelta(days=1)

DATASETS = {
    "NISAR": {"dataset": "NISAR", "processingLevel": "GCOV"},
    "OPERA": {"dataset": "OPERA-S1", "processingLevel": "RTC"},
}


def _get(kind, lon, lat, t0: datetime, t1: datetime, output, max_results=None):
    params = {
        **DATASETS[kind],
        "intersectsWith": f"POINT({lon} {lat})",
        "start": t0.strftime(TIME_FMT),
        "end": t1.strftime(TIME_FMT),
        "output": output,
    }
    if max_results:
        params["maxResults"] = max_results
    r = requests.get(ASF_SEARCH_URL, params=params, timeout=120)
    r.raise_for_status()
    return r


def _count(kind, lon, lat, t0, t1) -> int:
    return int(_get(kind, lon, lat, t0, t1, "count").text.strip())


def _features(kind, lon, lat, t0, t1, limit) -> list:
    return _get(kind, lon, lat, t0, t1, "geojson", limit).json().get("features", [])


def _collect(kind, lon, lat, t0, t1, need) -> list:
    """Up to `need` features in [t0, t1], newest first."""
    n = _count(kind, lon, lat, t0, t1)
    if n == 0:
        return []
    if n <= PAGE_SIZE or (t1 - t0) <= MIN_SPAN:
        if n > PAGE_SIZE:
            log.warning("%d scenes inside %s..%s exceed one page; keeping the newest %d",
                        n, t0.date(), t1.date(), PAGE_SIZE)
        return _features(kind, lon, lat, t0, t1, min(n, PAGE_SIZE, need))[:need]
    mid = (t0 + (t1 - t0) / 2).replace(microsecond=0)
    newer = _collect(kind, lon, lat, mid, t1, need)
    if len(newer) >= need:
        return newer
    older = _collect(kind, lon, lat, t0, mid - timedelta(seconds=1), need - len(newer))
    return newer + older


def _scene(kind, f):
    p = f["properties"]
    base = {
        "scene": p["sceneName"],
        "start_time": p["startTime"],
        "direction": p.get("flightDirection"),
        "track": p.get("pathNumber"),
        "platform": p.get("platform") or kind,
    }
    if kind == "NISAR":
        if str(p.get("url", "")).endswith(".h5"):
            return {**base, "burst_id": None, "url": p["url"]}
        return None
    # OPERA: the .h5 is metadata only; the backscatter lives in per-pol GeoTIFFs
    tifs = {}
    for u in p.get("additionalUrls", []):
        tag = u.rsplit("_", 1)[-1].removesuffix(".tif")
        if u.endswith(".tif") and tag in ("VV", "VH", "HH", "HV", "mask"):
            tifs[tag] = u
    if any(k != "mask" for k in tifs):
        return {**base, "burst_id": p.get("operaBurstID"), "tifs": tifs}
    return None


def search(kind, lon, lat, start: datetime, end: datetime, max_results=1000):
    total = _count(kind, lon, lat, start, end)
    log.info("%s: %d scenes match %s .. %s", kind, total, start.date(), end.date())
    if total > max_results:
        log.warning(
            "max-results=%d is smaller than the %d scenes available: only the NEWEST %d will be "
            "processed and the older ones are skipped. Raise --max-results to get them all.",
            max_results, total, max_results,
        )
    seen, scenes = set(), []
    for f in _collect(kind, lon, lat, start, end, max_results):
        s = _scene(kind, f)
        if s and s["scene"] not in seen:
            seen.add(s["scene"])
            scenes.append(s)
    return scenes
