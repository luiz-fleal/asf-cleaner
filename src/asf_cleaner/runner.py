"""Run per-scene work in parallel (threads or spawned processes) with isolation and retries."""
import logging
import multiprocessing
import os
import threading
import time
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, ThreadPoolExecutor, as_completed, wait
from concurrent.futures.process import BrokenProcessPool
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


_shared: ProcessPoolExecutor | None = None
_shared_lock = threading.Lock()
SHARED_PROCESSES = int(os.environ.get("ASF_API_PROC_WORKERS", "3"))


def shared_process_pool() -> ProcessPoolExecutor:
    """One long-lived process pool for the API: spawning interpreters (and importing h5py, pandas...)
    costs seconds, so requests reuse the same workers instead of starting new ones every time."""
    global _shared
    with _shared_lock:
        if _shared is None or getattr(_shared, "_broken", False):
            _shared = ProcessPoolExecutor(max_workers=SHARED_PROCESSES, mp_context=multiprocessing.get_context("spawn"))
        return _shared


def iter_parallel(fn, items, workers: int, kind: str, cancel: threading.Event | None = None, shared: bool = False):
    """Like run_parallel, but yields each Result as soon as it finishes (for streaming clients).

    Setting `cancel` (or closing the generator) stops waiting and cancels scenes not yet started.
    """
    items = list(items)
    n = len(items)
    if n == 0:
        return
    workers = max(1, min(workers, n))
    if workers == 1:
        for it in items:
            if cancel is not None and cancel.is_set():
                return
            rows, err, secs = _timed(fn, it)
            yield Result(it, rows, err, secs)
        return

    use_shared = shared and kind == "process"
    if use_shared:
        pool = shared_process_pool()
    elif kind == "process":
        pool = ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn"))
    else:
        pool = ThreadPoolExecutor(max_workers=workers)
    try:
        futures = {pool.submit(_timed, fn, it): it for it in items}
    except BrokenProcessPool:  # a worker died earlier: start a fresh pool once
        if not use_shared:
            raise
        pool = shared_process_pool()
        futures = {pool.submit(_timed, fn, it): it for it in items}
    pending = set(futures)
    try:
        while pending:
            if cancel is not None and cancel.is_set():
                break
            # short timeout so a cancel is noticed within half a second, not when a scene ends
            finished, pending = wait(pending, timeout=0.5, return_when=FIRST_COMPLETED)
            for f in finished:
                try:
                    yield Result(futures[f], *f.result())
                except Exception as e:  # pool-level failure (e.g. a worker process died)
                    yield Result(futures[f], None, f"{type(e).__name__}: {e}", 0.0)
    finally:
        if use_shared:  # keep the workers; only drop what this request had not started
            for f in pending:
                f.cancel()
        else:
            pool.shutdown(wait=False, cancel_futures=True)
