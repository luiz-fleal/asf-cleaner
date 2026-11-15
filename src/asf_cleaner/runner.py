"""Run per-scene work in parallel (threads or spawned processes) with isolation and retries."""
import logging
import multiprocessing
import os
import threading
import time
from concurrent.futures import (
    FIRST_COMPLETED,
    BrokenExecutor,
    ProcessPoolExecutor,
    ThreadPoolExecutor,
    as_completed,
    wait,
)
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


# ---- shared, long-lived pools (used by the API) -------------------------------------------------
# Spawning a process pool costs seconds (every worker re-imports numpy, h5py, rasterio...), so the
# API keeps one pool per (name, kind) for its whole lifetime instead of creating one per request.
# The pool size is the global cap on concurrent scene jobs; each request still limits itself to its
# own `workers` by keeping only that many scenes in flight.
_pools: dict[tuple[str, str], object] = {}
_pools_lock = threading.Lock()
_POOL_DEFAULTS = {("default", "process"): 8, ("default", "thread"): 32, ("rain", "thread"): 12}


def pool_size(name: str, kind: str) -> int:
    """Worker count of a shared pool; override with ASF_<NAME>_<KIND>_POOL, e.g. ASF_DEFAULT_PROCESS_POOL."""
    default = _POOL_DEFAULTS.get((name, kind), 8)
    return max(1, int(os.environ.get(f"ASF_{name.upper()}_{kind.upper()}_POOL", default)))


def _worker_init():
    """Runs once in every pool process: import the heavy modules before the first scene arrives."""
    from . import workers  # noqa: F401


def shared_pool(kind: str, name: str = "default"):
    """The long-lived executor for (name, kind), created on first use."""
    key = (name, kind)
    with _pools_lock:
        pool = _pools.get(key)
        if pool is None:
            size = pool_size(name, kind)
            if kind == "process":
                pool = ProcessPoolExecutor(max_workers=size, mp_context=multiprocessing.get_context("spawn"),
                                           initializer=_worker_init)
            else:
                pool = ThreadPoolExecutor(max_workers=size, thread_name_prefix=f"asf-{name}")
            _pools[key] = pool
        return pool


def _discard_pool(name: str, kind: str, pool) -> None:
    """Drop a broken pool (only if it is still the registered one) so the next call builds a new one."""
    with _pools_lock:
        if _pools.get((name, kind)) is pool:
            del _pools[(name, kind)]
    pool.shutdown(wait=False, cancel_futures=True)


def _noop():
    return None


def warm(kind: str = "process", name: str = "default", n: int | None = None) -> None:
    """Start `n` pool workers now, so the first request does not wait for them to spawn and import."""
    ex = shared_pool(kind, name)
    for _ in range(min(n if n is not None else pool_size(name, kind), pool_size(name, kind))):
        ex.submit(_noop)


def _submit(name: str, kind: str, fn, item):
    """Submit one item to the shared pool, rebuilding the pool once if it turns out to be broken."""
    for attempt in (0, 1):
        ex = shared_pool(kind, name)
        try:
            return ex.submit(_timed, fn, item), ex
        except RuntimeError:  # BrokenProcessPool, or a pool that was shut down
            _discard_pool(name, kind, ex)
            if attempt:
                raise


def iter_parallel(fn, items, workers: int, kind: str, cancel: threading.Event | None = None,
                  pool: str = "default"):
    """Like run_parallel, but yields each Result as soon as it finishes (for streaming clients).

    Runs on the shared pool `pool` (see shared_pool) and keeps at most `workers` items in flight, so
    one request cannot occupy the whole pool. Setting `cancel` (or closing the generator) stops
    waiting and cancels scenes not yet started; scenes already running finish in the background.
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

    queue_ = iter(items)
    pending: dict = {}  # future -> (item, pool it was submitted to)

    def submit_next() -> None:
        it = next(queue_, None)
        if it is not None:
            fut, ex = _submit(pool, kind, fn, it)
            pending[fut] = (it, ex)

    try:
        for _ in range(workers):
            submit_next()
        while pending:
            if cancel is not None and cancel.is_set():
                break
            # short timeout so a cancel is noticed within half a second, not when a scene ends
            finished, _ = wait(set(pending), timeout=0.5, return_when=FIRST_COMPLETED)
            for f in finished:
                it, ex = pending.pop(f)
                submit_next()  # refill the window before handing the result to a possibly slow consumer
                try:
                    yield Result(it, *f.result())
                except BrokenExecutor as e:  # a worker process died: the next scenes get a fresh pool
                    _discard_pool(pool, kind, ex)
                    yield Result(it, None, f"{type(e).__name__}: {e}", 0.0)
                except Exception as e:  # pool-level failure
                    yield Result(it, None, f"{type(e).__name__}: {e}", 0.0)
    finally:
        for f in pending:
            f.cancel()
