"""NASA GPM IMERG daily rain at a point (V07, 0.1 degree grid), fetched with OPeNDAP.

One tiny request per day returns a single grid cell, so no file is downloaded. Per day we use the
Final run (gauge-adjusted) where it exists and the Late run otherwise (Final ends 2025-09-30). The
daily value is the total for the UTC day (mm/day).

Run `python -m asf_cleaner.imerg LON LAT [YYYY-MM-DD]` to test access with your token.
"""
import io
import logging
import sys
from datetime import date, datetime, timedelta
from urllib.parse import quote

import h5py
import numpy as np
import pandas as pd
import requests

from . import cache
from .config import IMERG_COLLECTIONS, IMERG_LOOKBACK_DAYS, PROCESSING_VERSION
from .runner import iter_parallel, retry
from .stream import get_session

log = logging.getLogger("asf_cleaner")

CMR_GRANULES = "https://cmr.earthdata.nasa.gov/search/granules.json"
HYRAX = "https://opendap.earthdata.nasa.gov/collections/{cid}/granules/{name}"
STEP, LON0, LAT0, N_LON, N_LAT = 0.1, -179.95, -89.95, 3600, 1800  # cell centres
RAIN_WINDOWS = (1, 3, 7, 30)

ACCESS_HELP = (
    "IMERG (NASA GES DISC) rejected your Earthdata token (HTTP {code}). GES DISC is authorized "
    "separately from ASF: sign in at https://urs.earthdata.nasa.gov, open Applications > Authorized "
    "Apps > Approve More Applications and approve 'NASA GESDISC DATA ARCHIVE' (and an OPeNDAP "
    "application if one is listed), then rerun. Rain data is skipped for this run."
)


class ImergAccessError(RuntimeError):
    """The token is not accepted by GES DISC / OPeNDAP (needs a one-time authorization)."""


def grid_index(lon: float, lat: float) -> tuple[int, int]:
    """(lon index, lat index) of the 0.1-degree IMERG cell containing the point."""
    if not (-180 <= lon <= 180 and -90 <= lat <= 90):
        raise ValueError(f"lon/lat out of range: {lon}, {lat}")
    i = min(max(int(round((lon - LON0) / STEP)), 0), N_LON - 1)
    j = min(max(int(round((lat - LAT0) / STEP)), 0), N_LAT - 1)
    return i, j


# ---- finding the daily granules (CMR, no auth) ----------------------------------------------
def list_granules(run: str, d0: date, d1: date) -> dict[date, dict]:
    _, cid = IMERG_COLLECTIONS[run]
    found, page = {}, 1
    while True:
        r = requests.get(CMR_GRANULES, timeout=60, params={
            "collection_concept_id": cid, "temporal[]": f"{d0}T00:00:00Z,{d1}T23:59:59Z",
            "page_size": 2000, "page_num": page, "sort_key": "start_date"})
        r.raise_for_status()
        entries = r.json()["feed"]["entry"]
        for e in entries:
            title = e["title"]
            found[date.fromisoformat(e["time_start"][:10])] = {
                "granule": title, "url": HYRAX.format(cid=cid, name=quote(title, safe=""))}
        if len(entries) < 2000:
            return found
        page += 1


def plan_days(d0: date, d1: date) -> tuple[list[dict], list[date]]:
    """Per day: Final granule if it exists, else Late. Returns (days, days with no granule)."""
    final = list_granules("final", d0, d1)
    every = [d0 + timedelta(days=k) for k in range((d1 - d0).days + 1)]
    lacking = [d for d in every if d not in final]
    late = list_granules("late", min(lacking), d1) if lacking else {}
    days, missing = [], []
    for d in every:
        run = "final" if d in final else "late" if d in late else None
        if run:
            days.append({"date": d, "run": run, **(final if run == "final" else late)[d]})
        else:
            missing.append(d)
    return days, missing


# ---- fetching one cell -----------------------------------------------------------------------
def point_url(base: str, i: int, j: int) -> str:
    ce = quote(f"/precipitation[0][{i}][{j}];/lon[{i}];/lat[{j}]", safe="/[];")
    return f"{base}.dap.nc4?dap4.ce={ce}"


def parse_point(content: bytes) -> dict:
    """Read a DAP4 NetCDF-4 point response."""
    try:
        h5 = h5py.File(io.BytesIO(content), "r")
    except OSError:
        raise ValueError(f"OPeNDAP did not return a NetCDF-4 file: {content[:200]!r}") from None
    with h5:
        def find(name):
            hits = []
            h5.visititems(lambda n, o: hits.append(o) if isinstance(o, h5py.Dataset)
                          and n.rsplit("/", 1)[-1] == name else None)
            if not hits:
                raise ValueError(f"'{name}' missing from the OPeNDAP response")
            return hits[0]

        ds = find("precipitation")
        val = float(np.asarray(ds[()]).ravel()[0])
        fill = ds.attrs.get("_FillValue")
        if fill is not None and np.isclose(val, np.asarray(fill).ravel()[0]):
            val = np.nan
        units = ds.attrs.get("units", "")
        return {"precip": val, "units": units.decode() if isinstance(units, bytes) else str(units),
                "lon": float(np.asarray(find("lon")[()]).ravel()[0]),
                "lat": float(np.asarray(find("lat")[()]).ravel()[0])}


def fetch_point(item: dict, *, i: int, j: int, token: str) -> list[dict]:
    def go():
        r = get_session().get(point_url(item["url"], i, j), timeout=90,
                              headers={"Authorization": f"Bearer {token}"})
        if r.status_code in (401, 403):
            raise PermissionError(ACCESS_HELP.format(code=r.status_code))
        if r.status_code in (400, 404):  # bad constraint / granule missing: retrying won't help
            raise ValueError(f"OPeNDAP HTTP {r.status_code} for {item['granule']}: {r.text[:300]!r}")
        r.raise_for_status()
        return parse_point(r.content)

    p = retry(go)
    exp_lon, exp_lat = LON0 + i * STEP, LAT0 + j * STEP
    if abs(p["lon"] - exp_lon) > STEP / 2 or abs(p["lat"] - exp_lat) > STEP / 2:
        raise ValueError(f"grid mismatch: asked for cell ({exp_lon:.2f}, {exp_lat:.2f}), "
                         f"got ({p['lon']:.2f}, {p['lat']:.2f})")
    return [{"date": item["date"].isoformat(), "precipitation_mm_day": p["precip"],
             "imerg_run": item["run"], "granule": item["granule"], "grid_lon": p["lon"],
             "grid_lat": p["lat"], "units": p["units"], "processing_version": PROCESSING_VERSION}]


# ---- the daily series (cached) ---------------------------------------------------------------
def cache_path(out, lon, lat):
    """Cache file of the 0.1-degree IMERG cell containing the point. IMERG has one value per cell, so
    every point inside it shares the same file (and the same fetched days)."""
    from pathlib import Path
    i, j = grid_index(lon, lat)
    return Path(out) / ".cache" / f"imerg_cell{i}_{j}.parquet"


def iter_daily(lon, lat, start: datetime, end: datetime, *, token_fn, workers=8, out="out",
               use_cache=True, cancel=None, stats: dict | None = None):
    """Yield (source, rows, done, total) for the daily rain of [start - lookback, end].

    Cached days come first in one chunk ("cache"), then each fetched day as it arrives ("live"), so
    a caller can stream them. Days fetched are cached even if the caller stops early. `stats`
    (optional dict) receives `total` and `failed`. Raises ImergAccessError if the token is rejected.
    """
    stats = stats if stats is not None else {}
    d0, d1 = start.date() - timedelta(days=IMERG_LOOKBACK_DAYS), end.date()
    i, j = grid_index(lon, lat)
    path = cache_path(out, lon, lat)
    cached = cache.load(path, PROCESSING_VERSION) if use_cache else pd.DataFrame()
    have = dict(zip(cached["date"], cached["imerg_run"])) if len(cached) else {}
    every = [d0 + timedelta(days=k) for k in range((d1 - d0).days + 1)]
    # A day cached from the Final run is settled for good. Everything else (not cached, or cached from
    # Late, which Final may have replaced) has to be looked up in CMR. When nothing is left to look
    # up, which is the usual repeat request, CMR is not called at all.
    settled = {d for d in every if have.get(d.isoformat()) == "final"}
    need = [d for d in every if d not in settled]
    planned, missing = plan_days(need[0], d1) if need else ([], [])
    planned = [p for p in planned if p["date"] not in settled]
    missing = [d for d in missing if d not in settled]
    if missing:
        log.info("IMERG: no granule for %d day(s), e.g. %s (not published yet)", len(missing), missing[-1])

    # refetch a day cached from the Late run once the Final run is available for it
    todo = [d for d in planned if d["date"].isoformat() not in have
            or (d["run"] == "final" and have[d["date"].isoformat()] == "late")]
    total = len(settled) + len(planned)
    stats.update(total=total, failed=0)
    log.info("IMERG: %d days (%s .. %s): %d cached, %d to fetch, cell %.2f, %.2f",
             total, d0, d1, total - len(todo), len(todo), LON0 + i * STEP, LAT0 + j * STEP)

    ready = []
    if len(cached):
        refetch = {d["date"].isoformat() for d in todo}
        in_range = cached[(cached["date"] >= d0.isoformat()) & (cached["date"] <= d1.isoformat())
                          & ~cached["date"].isin(refetch)]
        ready = in_range.to_dict("records")
    done = len(ready)
    if ready:
        yield "cache", ready, done, total

    new_rows = []
    try:
        if todo:
            from functools import partial
            fn = partial(fetch_point, i=i, j=j, token=token_fn())
            try:  # fail fast on a rejected token instead of failing every day
                first = fn(todo[0])
                new_rows += first
                done += len(first)
                todo = todo[1:]
                yield "live", first, done, total
            except PermissionError as e:
                raise ImergAccessError(str(e)) from e
            except Exception as e:
                log.warning("IMERG first request failed (%s); trying the rest", e)
            for res in iter_parallel(fn, todo, workers, "thread", cancel, pool="rain"):
                if res.rows:
                    new_rows += res.rows
                    done += len(res.rows)
                    yield "live", res.rows, done, total
                else:
                    stats["failed"] += 1
                    log.warning("IMERG %s skipped: %s", res.item["date"], res.error)
    finally:
        if use_cache and new_rows:  # also when the caller stops early
            cache.save(path, cached, new_rows, key=["date"])
    if stats["failed"]:
        log.warning("IMERG: %d days failed and are not cached; rerun to retry them", stats["failed"])


def fetch_daily(lon, lat, start: datetime, end: datetime, *, token_fn, workers=8, out="out",
                use_cache=True) -> pd.DataFrame | None:
    """Daily rain for [start - lookback, end] as one table. Raises ImergAccessError if the token is rejected."""
    rows = []
    for _source, chunk, _done, _total in iter_daily(lon, lat, start, end, token_fn=token_fn,
                                                    workers=workers, out=out, use_cache=use_cache):
        rows += chunk
    if not rows:
        return None
    df = pd.DataFrame(rows).drop_duplicates("date", keep="last")
    return df.sort_values("date").reset_index(drop=True)


# ---- rain before each SAR acquisition ---------------------------------------------------------
def rain_features(df: pd.DataFrame, daily: pd.DataFrame) -> pd.DataFrame:
    """Add rain columns (mm) to a SAR table with a UTC `start_time`:

    rain_d0_mm   total of the acquisition's own UTC day (daily totals cannot be split at the
                 overpass time, so this may include rain after the pass)
    rain_Nd_mm   total of the N days BEFORE the acquisition day (N = 1, 3, 7, 30)

    A value is NaN if any day it needs is missing.
    """
    s = pd.Series(daily["precipitation_mm_day"].to_numpy(dtype=float),
                  index=pd.to_datetime(daily["date"]))
    s = s.reindex(pd.date_range(s.index.min(), s.index.max(), freq="D"))
    feats = {"rain_d0_mm": s}
    for n in RAIN_WINDOWS:
        feats[f"rain_{n}d_mm"] = s.rolling(n, min_periods=n).sum().shift(1)
    day = pd.to_datetime(df["start_time"], utc=True).dt.tz_convert("UTC").dt.tz_localize(None).dt.normalize()
    out = df.copy()
    for name, series in feats.items():
        out[name] = day.map(series).to_numpy()
    return out


if __name__ == "__main__":  # access probe
    from .config import get_token
    lon, lat = float(sys.argv[1]), float(sys.argv[2])
    d = date.fromisoformat(sys.argv[3]) if len(sys.argv) > 3 else date.today() - timedelta(days=5)
    i, j = grid_index(lon, lat)
    days, _ = plan_days(d, d)
    print(f"cell index lon={i} lat={j}; granule: {days[0]['granule'] if days else 'NONE'}")
    if days:
        print("url:", point_url(days[0]["url"], i, j))
        try:
            print(fetch_point(days[0], i=i, j=j, token=get_token())[0])
        except Exception as e:
            print("FAILED:", type(e).__name__, e)
