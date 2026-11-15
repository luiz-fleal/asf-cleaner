"""Integration tests: a real uvicorn server on a real socket, real HTTP and SSE streaming, fake upstreams.

They check what unit tests with TestClient cannot: events really arrive one by one while work is still going,
a client that leaves cancels the work, and the job slots are given back.
"""
import json
import socket
import threading
import time

import httpx
import numpy as np
import pandas as pd
import pytest
import uvicorn

from asf_cleaner import api
from asf_cleaner.config import PROCESSING_VERSION

SCENES = [
    {"scene": f"S{i:02d}", "start_time": f"2024-{i:02d}-10T09:00:00Z", "direction": "DESCENDING", "track": 10,
     "platform": "Sentinel-1A", "burst_id": "T010_IW3", "tifs": {"VV": "x", "VH": "y"}}
    for i in range(1, 13)
]
STARTED, FINISHED = [], []


def fake_worker(sc, *, lon, lat, window_m, token):
    STARTED.append(sc["scene"])
    time.sleep(0.15)
    FINISHED.append(sc["scene"])
    base = {"sensor": "OPERA", "scene": sc["scene"], "burst_id": sc["burst_id"], "track": sc["track"], "direction": sc["direction"],
            "start_time": sc["start_time"], "platform": sc["platform"], "processing_version": PROCESSING_VERSION, "valid_fraction": 1.0,
            "window_px": "17x17", "window_m": window_m, "pixel_spacing_m": 30.0, "normalization": "gamma0"}
    return [{**base, "polarization": p, "backscatter_db_mean": v, "backscatter_db_median": v, "backscatter_db_center": v} for p, v in (("VV", -8.0), ("VH", -14.0))]


@pytest.fixture
def server(tmp_path, monkeypatch):
    STARTED.clear()
    FINISHED.clear()
    monkeypatch.setattr(api, "search", lambda *a, **k: list(SCENES))
    monkeypatch.setattr(api, "WORKERS", {"OPERA": fake_worker, "NISAR": fake_worker})
    monkeypatch.setattr(api, "EXECUTORS", {"OPERA": "thread", "NISAR": "thread"})
    monkeypatch.setattr(api, "get_token", lambda: "tok")
    monkeypatch.setattr(api, "CACHE_DIR", str(tmp_path))
    api._search_cache.clear()
    api._forecast_cache.clear()
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    srv = uvicorn.Server(uvicorn.Config(api.app, host="127.0.0.1", port=port, log_level="warning"))
    th = threading.Thread(target=srv.run, daemon=True)
    th.start()
    while not srv.started:
        time.sleep(0.02)
    yield f"http://127.0.0.1:{port}"
    srv.should_exit = True
    th.join(5)


def events(resp):
    name = None
    for line in resp.iter_lines():
        if line.startswith("event: "):
            name = line[7:]
        elif line.startswith("data: "):
            yield name, json.loads(line[6:])


URL = "/series/stream?lon=-60.0&lat=-3.0&start=2024-01-01&end=2024-12-31&workers=2&cache=false"


def test_series_events_arrive_while_work_is_still_running(server):
    seen = []
    with httpx.stream("GET", server + URL, timeout=30) as r:
        assert r.status_code == 200 and r.headers["content-type"].startswith("text/event-stream")
        t0 = time.perf_counter()
        for name, data in events(r):
            seen.append((name, time.perf_counter() - t0, len(FINISHED)))
    names = [n for n, *_ in seen]
    assert names[0] == "meta" and names[-1] == "done" and names.count("rows") == 12
    first_rows = next(s for s in seen if s[0] == "rows")
    last_rows = [s for s in seen if s[0] == "rows"][-1]
    assert first_rows[2] < 12 and first_rows[1] < last_rows[1] - 0.3  # the first rows came well before the last scene finished


def test_leaving_cancels_the_work_and_frees_the_slot(server):
    with httpx.stream("GET", server + URL.replace("workers=2", "workers=1"), timeout=30) as r:
        for name, _ in events(r):
            if name == "rows":
                break  # the client closes the connection after the first scene
    time.sleep(1.5)
    n = len(STARTED)
    time.sleep(1.0)
    assert n < 12 and len(STARTED) == n  # nothing new starts after the client left
    ok = httpx.get(server + "/health")
    assert ok.json()["status"] == "ok"
    for _ in range(api.MAX_JOBS):  # all slots are free again
        assert api._jobs.acquire(blocking=False)
    for _ in range(api.MAX_JOBS):
        api._jobs.release()


def test_a_busy_server_answers_503_instead_of_queueing(server):
    held = [api._jobs.acquire(blocking=False) for _ in range(api.MAX_JOBS)]
    try:
        assert httpx.get(server + URL, timeout=10).status_code == 503
        assert httpx.get(server + "/health").status_code == 200  # health never needs a slot
    finally:
        for h in held:
            if h:
                api._jobs.release()


def test_cors_allows_the_dev_frontend_only(server):
    ok = httpx.get(server + "/health", headers={"Origin": "http://localhost:5173"})
    assert ok.headers.get("access-control-allow-origin") == "http://localhost:5173"
    other = httpx.get(server + "/health", headers={"Origin": "http://evil.example"})
    assert "access-control-allow-origin" not in other.headers


def test_concurrent_clients_do_not_mix_their_streams(server):
    out = {}

    def run(i):
        with httpx.stream("GET", server + URL.replace("lon=-60.0", f"lon=-6{i}.0"), timeout=60) as r:
            out[i] = [(n, d) for n, d in events(r)]

    ts = [threading.Thread(target=run, args=(i,)) for i in range(3)]
    [t.start() for t in ts]
    [t.join() for t in ts]
    for i in range(3):
        assert out[i][0][1]["lon"] == -float(f"6{i}") and out[i][-1][0] == "done"
        assert sum(len(d["rows"]) for n, d in out[i] if n == "rows") == 24  # 12 scenes x 2 polarizations


def test_json_and_stream_endpoints_agree(server):
    streamed = []
    with httpx.stream("GET", server + URL, timeout=30) as r:
        for n, d in events(r):
            if n == "rows":
                streamed += d["rows"]
    body = httpx.get(server + URL.replace("/series/stream", "/series"), timeout=30).json()
    key = lambda r: (r["scene"], r["pol"])  # noqa: E731
    assert sorted(map(key, streamed)) == sorted(map(key, body["rows"])) and body["fetched"] == 12


# ---- forecast and risk through the real server ---------------------------------------------------
def fake_history(monkeypatch):
    idx = pd.period_range("2016-06", "2026-10", freq="M")
    rng = np.random.default_rng(1)
    t = np.arange(len(idx))
    rows = []
    for pol, cls, base in (("VV", "co", -7.0), ("VH", "cross", -13.0)):
        drop = -3.0 * (t >= len(idx) - 12) if cls == "cross" else 0  # a clearing a year ago, seen in VH only
        for p, v in zip(idx, base + 0.7 * np.sin(2 * np.pi * idx.month.to_numpy() / 12) + drop + rng.normal(0, 0.2, len(idx))):
            rows.append({"start_time": p.start_time.tz_localize("UTC") + pd.Timedelta(days=10), "polarization": pol, "pol_class": cls,
                         "valid_fraction": 1.0, "backscatter_db_median": v})
    rain = pd.Series(200 + 150 * np.cos(2 * np.pi * (idx.month - 3) / 12) + rng.normal(0, 20, len(idx)), index=idx).clip(lower=0)

    def radar(*a, **k):
        for i in range(1, 4):
            yield "progress", {"phase": "radar", "done": i, "total": 3}
        yield "data", (pd.DataFrame(rows), {"track": 10}, len(idx))

    def rain_h(*a, **k):
        yield "progress", {"phase": "rain-monthly", "done": 1, "total": 1}
        yield "data", rain

    monkeypatch.setattr(api.history, "radar_history", radar)
    monkeypatch.setattr(api.optical, "features", lambda lon, lat: {k: 0.0 for k in api.optical.FEATURES})
    monkeypatch.setattr(api.risk, "load_model", lambda *args, **kw: None)  # tests never depend on the trained file
    monkeypatch.setattr(api.history, "rain_history", rain_h)


def test_forecast_over_http_includes_risks_with_the_clearing_on_top(server, monkeypatch):
    fake_history(monkeypatch)
    names = []
    result = None
    with httpx.stream("GET", server + "/forecast/stream?lon=-60.0&lat=-3.0&until=2030-12", timeout=60) as r:
        for n, d in events(r):
            names.append(n)
            if n == "result":
                result = d
    assert names[0] == "meta" and names[-1] == "done" and names.count("progress") >= 4
    assert result["until"] == "2030-12" and set(result["backscatter"]) == {"co", "cross"}
    risk = result["risk"]
    assert risk["available"] and risk["overall"]["main"] == "vegetation_loss" and risk["overall"]["level"] == "high"
    assert {r_["id"] for r_ in risk["risks"]} == {"vegetation_loss", "flood", "drought_fire", "unusual_change"}


def test_a_failing_risk_step_never_costs_the_forecast(server, monkeypatch):
    fake_history(monkeypatch)
    monkeypatch.setattr(api.risk, "assess", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))
    body = httpx.get(server + "/forecast?lon=-60.0&lat=-3.0", timeout=60).json()
    assert body["risk"]["available"] is False and "boom" in body["risk"]["reason"] and body["backscatter"]


def test_the_trained_forecast_is_served_from_memory_the_second_time(server, monkeypatch):
    fake_history(monkeypatch)
    httpx.get(server + "/forecast?lon=-60.0&lat=-3.0", timeout=60)
    monkeypatch.setattr(api.history, "radar_history", lambda *a, **k: (_ for _ in ()).throw(AssertionError("history must not be read again")))
    t0 = time.perf_counter()
    r = httpx.get(server + "/forecast?lon=-60.004&lat=-3.003", timeout=60)  # snaps to the same 0.01 degree cell
    assert r.status_code == 200 and time.perf_counter() - t0 < 1.0 and "risk" in r.json()


def test_slow_radar_jobs_never_starve_rain_or_forecast(server, monkeypatch):
    """The radar had taken every slot and rain/forecast answered 503: each kind of work has its own slots now."""
    held = [api._jobs.acquire(blocking=False) for _ in range(api.MAX_JOBS)]
    try:
        assert httpx.get(server + URL, timeout=10).status_code == 503  # radar is full
        monkeypatch.setattr(api.imerg, "plan_days", lambda d0, d1: ([], []))
        r = httpx.get(server + "/rain/stream?lon=-60&lat=-3&start=2026-01-01&end=2026-01-05", timeout=10)
        assert r.status_code == 200 and "event: done" in r.text
        fake_history(monkeypatch)
        assert httpx.get(server + "/forecast?lon=-60.0&lat=-3.0", timeout=60).status_code == 200
    finally:
        for h in held:
            if h:
                api._jobs.release()


def test_health_reports_the_slots(server):
    h = httpx.get(server + "/health").json()
    assert h["maxJobs"] == api.MAX_JOBS and h["maxForecasts"] >= 1
