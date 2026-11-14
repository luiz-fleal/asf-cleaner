"""Run per-scene work in parallel (threads or spawned processes) with isolation and retries."""
import logging
import multiprocessing
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from dataclasses import dataclass

import requests

log = logging.getLogger("asf_cleaner")

NO_RETRY = (PermissionError, ValueError)  # auth failure / point outside the grid: won't change
TRANSIENT = (OSError, requests.RequestException, TimeoutError)


def retry(fn, attempts: int = 3, base_delay: float = 2.0):
    """Call fn(); retry transient network/IO errors with exponential backoff."""
    for i in range(attempts):
        try:
            return fn()
        except NO_RETRY:
            raise
        except TRANSIENT as e:
            if i == attempts - 1:
                raise
            delay = base_delay * 2**i
            log.debug("transient error (%s); retrying in %.1fs", e, delay)
            time.sleep(delay)


@dataclass
class Result:
    item: object
    rows: list | None
    error: str | None
    seconds: float


def _timed(fn, item):
    """Top-level (picklable). Errors become strings so they always cross process boundaries."""
    t = time.perf_counter()
    try:
        return fn(item), None, time.perf_counter() - t
    except Exception as e:
        return None, f"{type(e).__name__}: {e}", time.perf_counter() - t


def run_parallel(fn, items, workers: int, kind: str, label=str) -> list[Result]:
    """Apply fn to every item. kind: 'thread' or 'process'. workers=1 runs inline.

    Failures are isolated per item. On Ctrl-C, pending work is cancelled and the results that
    finished are returned, so partial progress can still be saved.
    """
    items, results, n = list(items), [], len(items)
    if n == 0:
        return results

    def record(item, rows, err, secs):
        results.append(Result(item, rows, err, secs))
        if err:
            log.warning("[%d/%d] skip %s: %s", len(results), n, label(item), err)
        else:
            log.info("[%d/%d] ok   %s (%.1fs)", len(results), n, label(item), secs)

    workers = max(1, min(workers, n))
    if workers == 1:
        try:
            for it in items:
                record(it, *_timed(fn, it))
        except KeyboardInterrupt:
            log.warning("interrupted; keeping %d finished scenes", len(results))
        return results

    if kind == "process":
        pool = ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn"))
    else:
        pool = ThreadPoolExecutor(max_workers=workers)
    futures = {pool.submit(_timed, fn, it): it for it in items}
    try:
        for f in as_completed(futures):
            try:
                record(futures[f], *f.result())
            except Exception as e:  # pool-level failure (e.g. a worker process died)
                record(futures[f], None, f"{type(e).__name__}: {e}", 0.0)
    except KeyboardInterrupt:
        log.warning("interrupted; cancelling pending scenes, keeping %d finished", len(results))
        pool.shutdown(wait=False, cancel_futures=True)
        return results
    pool.shutdown()
    return results
