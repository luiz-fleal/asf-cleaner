import logging
import threading
import time
from functools import partial

import pandas as pd
import pytest

from asf_cleaner import cache, cli, runner
from asf_cleaner.stream import get_session


# ---- retry ---------------------------------------------------------------------------------
def test_retry_succeeds_on_third_attempt(monkeypatch):
    monkeypatch.setattr(runner.time, "sleep", lambda s: None)
    calls = []

    def flaky():
        calls.append(1)
        if len(calls) < 3:
            raise OSError("connection reset")
        return "ok"

    assert runner.retry(flaky) == "ok" and len(calls) == 3


@pytest.mark.parametrize("exc", [PermissionError("401"), ValueError("outside grid")])
def test_retry_skips_permanent_errors(exc, monkeypatch):
    monkeypatch.setattr(runner.time, "sleep", lambda s: None)
    calls = []

    def bad():
        calls.append(1)
        raise exc

    with pytest.raises(type(exc)):
        runner.retry(bad)
    assert len(calls) == 1


def test_retry_gives_up(monkeypatch):
    monkeypatch.setattr(runner.time, "sleep", lambda s: None)
    with pytest.raises(OSError):
        runner.retry(lambda: (_ for _ in ()).throw(OSError("x")), attempts=2)


# ---- run_parallel --------------------------------------------------------------------------
def slow(x):
    time.sleep(0.2)
    return [x]


def test_threads_run_concurrently():
    t = time.perf_counter()
    res = runner.run_parallel(slow, range(8), workers=4, kind="thread")
    assert time.perf_counter() - t < 1.0  # serial would be 1.6s
    assert sorted(r.rows[0] for r in res) == list(range(8))


def test_threads_use_several_threads():
    seen = set()

    def fn(x):
        seen.add(threading.get_ident())
        time.sleep(0.1)
        return [x]

    runner.run_parallel(fn, range(8), workers=4, kind="thread")
    assert len(seen) > 1


def test_failure_isolation_and_inline():
    def fn(x):
        if x == 2:
            raise ValueError("boom")
        return [x]

    for workers in (1, 3):
        res = runner.run_parallel(fn, range(5), workers=workers, kind="thread")
        assert len(res) == 5
        bad = [r for r in res if r.error]
        assert len(bad) == 1 and "ValueError: boom" in bad[0].error and bad[0].item == 2


def test_process_pool():
    res = runner.run_parallel(partial(pow, 2), [1, 2, 3, 4], workers=2, kind="process")
    assert sorted(r.rows for r in res) == [2, 4, 8, 16]


def test_empty_items():
    assert runner.run_parallel(slow, [], workers=4, kind="thread") == []


# ---- session -------------------------------------------------------------------------------
def test_session_reused_with_retries():
    s = get_session()
    assert s is get_session()
    assert s.get_adapter("https://x.org").max_retries.total == 4
    other = []
    t = threading.Thread(target=lambda: other.append(get_session()))
    t.start(); t.join()
    assert other[0] is not s  # one session per thread


# ---- cache ---------------------------------------------------------------------------------
def row(scene, pol="VV", version=2):
    return {"scene": scene, "polarization": pol, "processing_version": version, "v": 1.0}


def test_cache_roundtrip_and_merge(tmp_path):
    p = cache.cache_path(tmp_path, "OPERA", -47.9, -15.8, 500)
    assert cache.load(p, 2).empty
    cache.save(p, pd.DataFrame(), [row("a"), row("a", "VH")])
    old = cache.load(p, 2)
    cache.save(p, old, [row("b"), row("a")])  # duplicate a/VV replaced, not doubled
    got = cache.load(p, 2)
    assert len(got) == 3 and set(got.scene) == {"a", "b"}
    assert not list(p.parent.glob("*.tmp"))


def test_cache_version_mismatch_ignored(tmp_path):
    p = cache.cache_path(tmp_path, "NISAR", 1, 2, 500)
    cache.save(p, pd.DataFrame(), [row("a", version=1)])
    assert cache.load(p, 2).empty
    pd.DataFrame([{"scene": "a", "polarization": "HH"}]).to_parquet(p)  # old file, no column
    assert cache.load(p, 2).empty


def test_cache_path_depends_on_window(tmp_path):
    assert cache.cache_path(tmp_path, "OPERA", 1, 2, 500) != cache.cache_path(tmp_path, "OPERA", 1, 2, 300)


def test_corrupt_cache_ignored(tmp_path, caplog):
    p = cache.cache_path(tmp_path, "OPERA", 1, 2, 500)
    p.parent.mkdir(parents=True)
    p.write_bytes(b"not parquet")
    with caplog.at_level(logging.WARNING, logger="asf_cleaner"):
        assert cache.load(p, 2).empty
    assert "unreadable" in caplog.text


# ---- CLI integration (no network) ----------------------------------------------------------
def scene(n):
    return {"scene": f"s{n}", "burst_id": "B", "track": 24, "direction": "DESCENDING",
            "start_time": f"2018-01-{n:02d}T08:00:00Z", "platform": "S1A", "tifs": {}}


def fake_rows(sc):
    return [{"sensor": "OPERA", **{k: sc[k] for k in ("scene", "burst_id", "track", "direction",
             "start_time", "platform")}, "polarization": p, "valid_fraction": 1.0,
             "backscatter_db_mean": -10.0, "backscatter_db_median": -10.0,
             "backscatter_db_center": -10.0, "window_m": 500, "window_px": "17x17",
             "pixel_spacing_m": 30.0, "normalization": "gamma0", "processing_version": 2}
            for p in ("VV", "VH")]


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setenv("EARTHDATA_TOKEN", "x")
    state = {"scenes": [scene(1), scene(2), scene(3)], "fetched": [], "fail": set()}
    monkeypatch.setattr(cli, "search", lambda *a, **k: list(state["scenes"]))

    def worker(sc, **kw):
        state["fetched"].append(sc["scene"])
        if sc["scene"] in state["fail"]:
            raise OSError("network down")
        return fake_rows(sc)

    monkeypatch.setitem(cli.WORKERS, "OPERA", worker)
    args = ["--dataset", "OPERA", "--start", "2018-01-01", "--end", "2018-02-28",
            "--workers", "1", "--no-imerg", "--out", str(tmp_path)]
    state["tmp"] = tmp_path
    state["run"] = lambda *extra: cli.main(args + list(extra))
    state["out"] = lambda: pd.read_parquet(tmp_path / "opera_-47.9_-15.8.parquet")
    return state


def test_second_run_uses_cache_and_needs_no_token(env, monkeypatch):
    env["run"]()
    assert env["fetched"] == ["s1", "s2", "s3"] and len(env["out"]()) == 6
    env["fetched"].clear()
    monkeypatch.delenv("EARTHDATA_TOKEN")
    monkeypatch.chdir(env["tmp"])  # so no .env file can supply a token
    env["run"]()
    assert env["fetched"] == [] and len(env["out"]()) == 6


def test_extending_range_fetches_only_new(env):
    env["run"]()
    env["fetched"].clear()
    env["scenes"].append(scene(4))
    env["run"]()
    assert env["fetched"] == ["s4"] and len(env["out"]()) == 8


def test_output_only_has_requested_scenes_but_cache_keeps_all(env):
    env["run"]()
    env["scenes"][:] = [scene(2)]  # narrower request
    env["fetched"].clear()
    env["run"]()
    assert set(env["out"]().scene) == {"s2"} and env["fetched"] == []
    env["scenes"][:] = [scene(1), scene(2), scene(3)]  # earlier work was not lost
    env["run"]()
    assert env["fetched"] == [] and len(env["out"]()) == 6


def test_failed_scene_retried_next_run(env):
    env["fail"].add("s2")
    env["run"]()
    assert set(env["out"]().scene) == {"s1", "s3"}
    env["fetched"].clear()
    env["fail"].clear()
    env["run"]()
    assert env["fetched"] == ["s2"] and set(env["out"]().scene) == {"s1", "s2", "s3"}


def test_no_cache_refetches(env):
    env["run"]()
    env["fetched"].clear()
    env["run"]("--no-cache")
    assert env["fetched"] == ["s1", "s2", "s3"]


def test_parallel_workers_flag(env):
    env["run"]("--workers", "3")
    assert sorted(env["fetched"]) == ["s1", "s2", "s3"]
