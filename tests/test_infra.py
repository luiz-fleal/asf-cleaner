"""Unit tests for the small infrastructure pieces: expiring cache, SSE framing, process/thread runner."""
import json
import threading
import time

import pytest

from asf_cleaner import memo, runner, sse


# ---- TTLCache ------------------------------------------------------------------------------
def test_ttl_cache_hit_miss_and_expiry():
    c = memo.TTLCache(4)
    assert c.get("k", 10) is None
    c.put("k", [1, 2])
    assert c.get("k", 10) == [1, 2]
    time.sleep(0.05)
    assert c.get("k", 0.01) is None  # older than the ttl asked for
    assert c.get("k", 10) == [1, 2]  # the same entry is still good for a longer ttl


def test_ttl_cache_evicts_oldest_and_clears():
    c = memo.TTLCache(2)
    for k in "abc":
        c.put(k, k)
    assert c.get("a", 10) is None and c.get("b", 10) == "b" and c.get("c", 10) == "c"
    c.clear()
    assert c.get("c", 10) is None


def test_ttl_cache_is_thread_safe():
    c = memo.TTLCache(50)

    def work(n):
        for i in range(200):
            c.put((n, i % 20), i)
            c.get((n, (i + 3) % 20), 5)

    ts = [threading.Thread(target=work, args=(n,)) for n in range(8)]
    [t.start() for t in ts]
    [t.join() for t in ts]
    assert len(c._data) <= 50


# ---- SSE framing -----------------------------------------------------------------------------
def test_sse_frame_is_compact_json():
    frame = sse.sse("rows", {"a": 1, "b": [1, 2]})
    assert frame == 'event: rows\ndata: {"a":1,"b":[1,2]}\n\n'
    name, data = frame.strip().split("\n")
    assert name == "event: rows" and json.loads(data.removeprefix("data: ")) == {"a": 1, "b": [1, 2]}


def test_sse_keeps_non_ascii_valid():
    assert json.loads(sse.sse("x", {"t": "São Paulo"}).split("data: ")[1]) == {"t": "São Paulo"}


# ---- runner ------------------------------------------------------------------------------------
def square(x):
    return [x * x]


def boom(x):
    if x == 3:
        raise ValueError("bad item")
    return [x]


def slow(x):
    time.sleep(0.3)
    return [x]


def test_iter_parallel_yields_every_result_with_isolated_errors():
    out = list(runner.iter_parallel(boom, range(6), 3, "thread"))
    assert sorted(r.item for r in out) == list(range(6))
    bad = [r for r in out if r.error]
    assert len(bad) == 1 and bad[0].item == 3 and "bad item" in bad[0].error
    assert sorted(r.rows[0] for r in out if not r.error) == [0, 1, 2, 4, 5]


def test_iter_parallel_inline_and_empty():
    assert [r.rows for r in runner.iter_parallel(square, [2, 3], 1, "thread")] == [[4], [9]]
    assert list(runner.iter_parallel(square, [], 4, "thread")) == []


def test_iter_parallel_cancel_stops_early():
    cancel = threading.Event()
    got = []
    t0 = time.perf_counter()
    for r in runner.iter_parallel(slow, range(40), 2, "thread", cancel):
        got.append(r)
        cancel.set()
    assert len(got) < 40 and time.perf_counter() - t0 < 5


def test_run_parallel_collects_results_and_errors():
    res = runner.run_parallel(boom, range(5), 2, "thread")
    assert len(res) == 5 and sum(1 for r in res if r.error) == 1


def test_retry_retries_transient_but_not_permission_errors():
    calls = []

    def flaky():
        calls.append(1)
        if len(calls) < 3:
            raise TimeoutError("slow")
        return "ok"

    assert runner.retry(flaky, attempts=3, base_delay=0) == "ok" and len(calls) == 3
    with pytest.raises(PermissionError):
        runner.retry(lambda: (_ for _ in ()).throw(PermissionError("no")), attempts=3, base_delay=0)


def test_shared_process_pool_is_reused_and_survives_requests():
    first = runner.shared_process_pool()
    assert runner.shared_process_pool() is first
    out = list(runner.iter_parallel(square, range(4), 3, "process", shared=True))
    assert sorted(r.rows[0] for r in out) == [0, 1, 4, 9]
    assert runner.shared_process_pool() is first  # not shut down by the request
    again = list(runner.iter_parallel(square, [5], 3, "process", shared=True))
    assert again[0].rows == [25]
