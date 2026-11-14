import argparse
import logging
import time
from functools import partial
from pathlib import Path

import pandas as pd

from . import cache
from .config import (
    EXECUTORS,
    PROCESSING_VERSION,
    Settings,
    get_token,
    resolve_dataset,
    resolve_range,
)
from .harmonize import harmonize
from .runner import run_parallel
from .search import search
from .workers import process_nisar, process_opera

log = logging.getLogger("asf_cleaner")
WORKERS = {"NISAR": process_nisar, "OPERA": process_opera}


def run(s: Settings) -> pd.DataFrame | None:
    t0 = time.perf_counter()
    scenes = search(s.dataset, s.lon, s.lat, s.start, s.end, s.max_results)

    cache_file = cache.cache_path(s.out, s.dataset, s.lon, s.lat, s.window_m)
    cached = cache.load(cache_file, PROCESSING_VERSION) if s.use_cache else pd.DataFrame()
    have = set(cached["scene"]) if len(cached) else set()
    todo = [sc for sc in scenes if sc["scene"] not in have]
    log.info("%d scenes: %d already cached, %d to fetch (workers=%d, %s pool)",
             len(scenes), len(scenes) - len(todo), len(todo), s.workers, EXECUTORS[s.dataset])

    results = []
    if todo:  # the token is only needed when something must be fetched
        fn = partial(WORKERS[s.dataset], lon=s.lon, lat=s.lat, window_m=s.window_m, token=get_token())
        results = run_parallel(fn, todo, s.workers, EXECUTORS[s.dataset], label=lambda sc: sc["scene"])
    new_rows = [r for res in results if res.rows for r in res.rows]
    failed = sum(1 for res in results if res.error)
    if s.use_cache and new_rows:
        cache.save(cache_file, cached, new_rows)  # also reached after Ctrl-C (partial progress)

    names = {sc["scene"] for sc in scenes}
    parts = [d for d in (cached[cached["scene"].isin(names)] if len(cached) else cached,
                         pd.DataFrame(new_rows)) if len(d)]
    elapsed = time.perf_counter() - t0
    fetched = len(results) - failed
    log.info("done in %.1fs: %d from cache, %d fetched (%.2f scenes/s), %d failed",
             elapsed, len(scenes) - len(todo), fetched, fetched / elapsed if elapsed else 0, failed)
    if failed:
        log.warning("%d scenes failed and are not cached; rerun to retry them", failed)
    return pd.concat(parts, ignore_index=True) if parts else None


def main(argv=None):
    d = Settings()
    ap = argparse.ArgumentParser(description="Stream & clean ASF NISAR / OPERA-S1 data at a point")
    ap.add_argument("--dataset", default=d.dataset, choices=["auto", *WORKERS],
                    help="auto: NISAR if the range is inside NISAR coverage, else OPERA")
    ap.add_argument("--lon", type=float, default=d.lon)
    ap.add_argument("--lat", type=float, default=d.lat)
    ap.add_argument("--start", help="e.g. 2026-06-01 (default: 180 days before --end)")
    ap.add_argument("--end", help="e.g. 2026-11-13 (default: now)")
    ap.add_argument("--max-results", type=int, default=d.max_results,
                    help="cap on total scenes; newest are kept and a warning is logged if exceeded")
    ap.add_argument("--window-m", type=float, default=d.window_m, help="ground footprint in meters")
    ap.add_argument("--workers", type=int, default=d.workers,
                    help="scenes processed in parallel (1 = sequential)")
    ap.add_argument("--no-cache", action="store_true", help="refetch everything, ignore the cache")
    ap.add_argument("--out", default=d.out)
    a = ap.parse_args(argv)
    if a.workers < 1:
        ap.error("--workers must be >= 1")
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    start, end = resolve_range(a.start, a.end)
    dataset, why = resolve_dataset(a.dataset, start, end)
    log.info("mode: %s - %s", dataset, why)
    s = Settings(lon=a.lon, lat=a.lat, start=start, end=end, max_results=a.max_results,
                 window_m=a.window_m, out=a.out, dataset=dataset, workers=a.workers,
                 use_cache=not a.no_cache)
    raw = run(s)
    if raw is None:
        log.warning("no data produced (no scenes at this point/range, or all scenes failed)")
        return
    df = harmonize(raw)
    out = Path(s.out)
    out.mkdir(parents=True, exist_ok=True)
    stem = out / f"{s.dataset.lower()}_{s.lon}_{s.lat}"
    df.to_csv(f"{stem}.csv", index=False)
    df.to_parquet(f"{stem}.parquet", index=False)
    log.info("wrote %s.{csv,parquet} (%d rows)", stem, len(df))


if __name__ == "__main__":
    main()
