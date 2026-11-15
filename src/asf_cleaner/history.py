"""The long history the forecast learns from, collected on demand and cached.

Both functions are generators of (event, payload) pairs: `progress` events while work is under way and one
final `data` event with the result, so a streaming endpoint can forward the progress as it happens.

- Radar: one OPERA C-band scene per month, always from the geometry (burst, track, direction) the point has most
  scenes in, so the incidence angle does not change from month to month. Scenes land in the same cache the
  /series endpoint uses.
- Rain: monthly GPM IMERG (final run) from the monthly product, and for the months it does not cover yet the sum of
  the daily Late run (shared with the /rain cache). Only complete months are used.
"""
from __future__ import annotations

import threading
from collections import Counter
from datetime import date, datetime, timedelta, timezone
from functools import partial
from pathlib import Path

import pandas as pd

from . import cache, imerg
from .config import EXECUTORS, PROCESSING_VERSION, get_token
from .harmonize import harmonize
from .runner import iter_parallel
from .workers import process_opera

RADAR_START = datetime(2016, 1, 1, tzinfo=timezone.utc)
MONTHLY_LAST = date(2025, 9, 30)  # the final monthly product ends here
FRESH_DAYS = 4  # a month is only used once it ended this many days ago
MIN_DAY_COVERAGE = 0.9


# ---- radar -------------------------------------------------------------------------------------
def pick_monthly_scenes(scenes: list[dict]) -> tuple[list[dict], dict]:
    """One scene per month (the one closest to the 15th) from the busiest geometry."""
    if not scenes:
        return [], {}
    key = lambda s: (s.get("burst_id"), s.get("track"), s.get("direction"))  # noqa: E731
    best, _ = Counter(key(s) for s in scenes).most_common(1)[0]
    chosen: dict[str, dict] = {}
    for s in scenes:
        if key(s) != best:
            continue
        t = pd.Timestamp(s["start_time"]).tz_convert("UTC") if pd.Timestamp(s["start_time"]).tzinfo else pd.Timestamp(s["start_time"], tz="UTC")
        m = f"{t:%Y-%m}"
        gap = abs((t - pd.Timestamp(f"{m}-15", tz="UTC")).total_seconds())
        if m not in chosen or gap < chosen[m][0]:
            chosen[m] = (gap, s)
    picked = [v[1] for _, v in sorted(chosen.items())]
    return picked, {"burst": best[0], "track": best[1], "direction": best[2]}


def radar_history(lon, lat, *, search_fn, cache_dir, workers=8, use_cache=True, window_m=500, cancel=None, start=RADAR_START):
    end = datetime.now(timezone.utc).replace(microsecond=0)
    yield "progress", {"phase": "radar-search", "done": 0, "total": 1}
    scenes = search_fn("OPERA", lon, lat, start, end, 2000)
    picked, geometry = pick_monthly_scenes(scenes)
    yield "progress", {"phase": "radar-search", "done": 1, "total": 1}

    path = cache.cache_path(cache_dir, "opera", lon, lat, window_m)
    cached = cache.load(path, PROCESSING_VERSION) if use_cache else pd.DataFrame()
    names = {s["scene"] for s in picked}
    have = set(cached["scene"]) & names if len(cached) else set()
    todo = [s for s in picked if s["scene"] not in have]
    new_rows, done = [], len(have)
    yield "progress", {"phase": "radar", "done": done, "total": len(picked)}
    try:
        if todo:
            fn = partial(process_opera, lon=lon, lat=lat, window_m=window_m, token=get_token())
            for res in iter_parallel(fn, todo, workers, EXECUTORS["OPERA"], cancel):
                done += 1
                if not res.error:
                    new_rows.extend(res.rows)
                yield "progress", {"phase": "radar", "done": done, "total": len(picked)}
    finally:
        if use_cache and new_rows:
            cache.save(path, cached, new_rows)
    parts = [d for d in (cached[cached["scene"].isin(names)] if len(cached) else cached, pd.DataFrame(new_rows)) if len(d)]
    if not parts:
        yield "data", (pd.DataFrame(), geometry, len(picked))
        return
    rows = harmonize(pd.concat(parts, ignore_index=True))
    yield "data", (rows, geometry, len(picked))


# ---- rain --------------------------------------------------------------------------------------
def _month_end(m: pd.Period) -> date:
    return m.end_time.date()


def rain_history(lon, lat, *, cache_dir, workers=8, use_cache=True, today: date | None = None, cancel=None):
    """Monthly rain (mm) at the point's IMERG cell since 2016."""
    today = today or date.today()
    i, j = imerg.grid_index(lon, lat)
    token = get_token()
    first = pd.Period(RADAR_START.strftime("%Y-%m"), freq="M")
    last_full = pd.Period(today - timedelta(days=FRESH_DAYS), freq="M") - 1  # month that ended at least FRESH_DAYS ago

    # 1) the monthly product (immutable, cached for good)
    mpath = Path(cache_dir) / ".cache" / f"imerg_monthly_cell_{i}_{j}.parquet"
    mcached = cache.load(mpath, PROCESSING_VERSION) if use_cache else pd.DataFrame()
    have = set(mcached["month"]) if len(mcached) else set()
    monthly_to = min(last_full, pd.Period(MONTHLY_LAST, freq="M"))
    yield "progress", {"phase": "rain-monthly", "done": 0, "total": 1}
    granules = imerg.list_granules("monthly", first.start_time.date(), _month_end(monthly_to)) if first <= monthly_to else {}
    todo = [{"date": d, "granule": g["granule"], "url": g["url"]} for d, g in sorted(granules.items()) if f"{d:%Y-%m}" not in have]
    new = []
    done = len(granules) - len(todo)
    yield "progress", {"phase": "rain-monthly", "done": done, "total": len(granules)}
    try:
        if todo:
            fn = partial(imerg.fetch_monthly_point, i=i, j=j, token=token)
            try:
                first_rows = fn(todo[0])
            except PermissionError as err:
                raise RuntimeError(str(err)) from err
            new += first_rows
            done += 1
            yield "progress", {"phase": "rain-monthly", "done": done, "total": len(granules)}
            for res in iter_parallel(fn, todo[1:], workers, "thread", cancel):
                done += 1
                if not res.error:
                    new.extend(res.rows)
                yield "progress", {"phase": "rain-monthly", "done": done, "total": len(granules)}
    finally:
        if use_cache and new:
            cache.save(mpath, mcached, new, key=["month"])
    parts = [d for d in (mcached, pd.DataFrame(new)) if len(d)]
    series = {}
    if parts:
        m = pd.concat(parts, ignore_index=True).drop_duplicates("month", keep="last")
        series = {pd.Period(r.month, freq="M"): float(r.mm) for r in m.itertuples() if pd.notna(r.mm)}

    # 2) months after the monthly product ends: sum of the daily Late run
    recent_from = monthly_to + 1
    if recent_from <= last_full:
        d0, d1 = recent_from.start_time.date(), _month_end(last_full)
        dpath = Path(cache_dir) / ".cache" / f"imerg_cell_{i}_{j}.parquet"
        dcached = cache.load(dpath, PROCESSING_VERSION) if use_cache else pd.DataFrame()
        dhave = set(dcached["date"]) if len(dcached) else set()
        days, _ = imerg.plan_days(d0, d1)
        dtodo = sorted((d for d in days if d["date"].isoformat() not in dhave), key=lambda d: d["date"], reverse=True)
        total, done, newd = len(days), len(days) - len(dtodo), []
        yield "progress", {"phase": "rain-daily", "done": done, "total": total}
        try:
            if dtodo:
                fn = partial(imerg.fetch_point, i=i, j=j, token=token)
                for res in iter_parallel(fn, dtodo, workers, "thread", cancel):
                    done += 1
                    if not res.error:
                        newd.extend(res.rows)
                    if done % 10 == 0 or done == total:
                        yield "progress", {"phase": "rain-daily", "done": done, "total": total}
        finally:
            if use_cache and newd:
                cache.save(dpath, dcached, newd, key=["date"])
        dparts = [d for d in (dcached, pd.DataFrame(newd)) if len(d)]
        if dparts:
            dd = pd.concat(dparts, ignore_index=True).drop_duplicates("date", keep="last")
            dd = dd[dd["precipitation_mm_day"].notna() & (dd["date"] >= d0.isoformat()) & (dd["date"] <= d1.isoformat())]
            dd["month"] = pd.PeriodIndex(pd.to_datetime(dd["date"]), freq="M")
            for mon, grp in dd.groupby("month"):
                if len(grp) >= MIN_DAY_COVERAGE * mon.days_in_month:
                    series[mon] = float(grp["precipitation_mm_day"].mean() * mon.days_in_month)
    out = pd.Series(series, dtype=float).sort_index()
    out.index = pd.PeriodIndex(out.index, freq="M")
    yield "data", out
