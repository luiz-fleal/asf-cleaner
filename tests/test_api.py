"""API tests: no network and no token (ASF search and the per-scene workers are replaced)."""
import json
from datetime import date, timedelta

import pytest
from fastapi.testclient import TestClient

from asf_cleaner import api, imerg
from asf_cleaner.config import PROCESSING_VERSION

SCENES = [
    {"scene": f"S{i}", "start_time": f"2024-0{i}-10T09:00:00Z", "direction": "DESCENDING", "track": 10,
     "platform": "Sentinel-1A", "burst_id": "T010_IW3", "tifs": {"VV": "x", "VH": "y"}}
    for i in (1, 2, 3)
]


def fake_worker(sc, *, lon, lat, window_m, token):
    base = {"sensor": "OPERA", "scene": sc["scene"], "burst_id": sc["burst_id"], "track": sc["track"],
            "direction": sc["direction"], "start_time": sc["start_time"], "platform": sc["platform"],
            "processing_version": PROCESSING_VERSION, "valid_fraction": 1.0, "window_px": "17x17",
            "window_m": window_m, "pixel_spacing_m": 30.0, "normalization": "gamma0"}
    return [{**base, "polarization": p, "backscatter_db_mean": v, "backscatter_db_median": v, "backscatter_db_center": v}
            for p, v in (("VV", -8.0), ("VH", -14.0))]


def no_rain(*a, **k):
    """IMERG stand-in that finds nothing (keeps the SAR tests off the network)."""
    return iter(())


def make_rain(fail=None, calls=None):
    """IMERG stand-in: 103 days from 2023-12-01 (mm = day of month); the first 50 come as 'cache'."""
    def fake(lon, lat, start, end, *, stats=None, **kw):
        if calls is not None:
            calls.append((lon, lat))
        if fail == "access":
            raise imerg.ImergAccessError("authorize NASA GESDISC DATA ARCHIVE")
        days = [date(2023, 12, 1) + timedelta(days=k) for k in range(103)]
        mk = lambda ds: [{"date": d.isoformat(), "precipitation_mm_day": float(d.day), "imerg_run": "final"} for d in ds]  # noqa: E731
        if stats is not None:
            stats.update(total=len(days), failed=0)
        yield "cache", mk(days[:50]), 50, 103
        for n, d in enumerate(days[50:], start=51):
            if fail == "mid" and n == 80:
                raise RuntimeError("cmr down")
            yield "live", mk([d]), n, 103
    return fake


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(api.imerg, "iter_daily", no_rain)
    monkeypatch.setattr(api, "search", lambda *a, **k: list(SCENES))
    monkeypatch.setattr(api, "WORKERS", {"OPERA": fake_worker, "NISAR": fake_worker})
    monkeypatch.setattr(api, "get_token", lambda: "tok")
    monkeypatch.setattr(api, "CACHE_DIR", str(tmp_path))
    return TestClient(api.app)


def parse(text):
    out = []
    for block in text.strip().split("\n\n"):
        lines = [ln for ln in block.splitlines() if not ln.startswith(":")]
        if lines:
            out.append((lines[0].removeprefix("event: "), json.loads(lines[1].removeprefix("data: "))))
    return out


URL = "/series/stream?lon=-60.0217&lat=-3.119&start=2018-01-01&end=2024-12-31&workers=2"


def test_health(client):
    assert client.get("/health").json()["tokenConfigured"] is True


def test_bad_input_is_rejected(client):
    assert client.get("/series/stream?lon=300&lat=0").status_code == 422
    assert client.get("/series/stream?lon=0&lat=0&start=2026-02-01&end=2026-01-01").status_code == 400


def test_stream_sends_rows_as_scenes_finish(client):
    r = client.get(URL)
    ev = parse(r.text)
    names = [e for e, _ in ev]
    assert names[0] == "meta" and names[-1] == "done"
    assert names.count("rows") == 3  # one per scene, nothing precomputed
    meta = ev[0][1]
    assert meta["dataset"] == "OPERA" and meta["total"] == 3 and meta["toFetch"] == 3
    rows = [row for e, d in ev if e == "rows" for row in d["rows"]]
    assert len(rows) == 6 and {r["pol"] for r in rows} == {"VV", "VH"}
    assert all(r["db"] in (-8.0, -14.0) and r["polClass"] in ("co", "cross") for r in rows)
    assert ev[-1][1]["fetched"] == 3 and ev[-1][1]["failed"] == 0


def test_second_request_is_served_from_cache(client):
    client.get(URL)
    ev = parse(client.get(URL).text)
    assert ev[0][1]["cached"] == 3 and ev[0][1]["toFetch"] == 0
    assert ev[1][0] == "rows" and ev[1][1]["source"] == "cache" and len(ev[1][1]["rows"]) == 6


def test_failed_scene_does_not_break_the_stream(client, monkeypatch):
    def flaky(sc, **k):
        if sc["scene"] == "S2":
            raise ValueError("point falls outside the scene")
        return fake_worker(sc, **k)

    monkeypatch.setattr(api, "WORKERS", {"OPERA": flaky, "NISAR": flaky})
    ev = parse(client.get(URL + "&cache=false").text)
    assert [e for e, _ in ev].count("scene_error") == 1
    assert ev[-1][1]["failed"] == 1 and ev[-1][1]["fetched"] == 2


def test_blocking_endpoint_matches_the_stream(client):
    j = client.get("/series?lon=-60.0217&lat=-3.119&start=2018-01-01&end=2024-12-31&workers=2").json()
    assert j["meta"]["total"] == 3 and len(j["rows"]) == 6 and j["failed"] == 0


# ---- IMERG rain is fetched with the SAR data and returned to the client ---------------------
def rain_stream(client, monkeypatch, **kw):
    calls = []
    monkeypatch.setattr(api.imerg, "iter_daily", make_rain(calls=calls, **kw))
    ev = parse(client.get(URL).text)
    return ev, calls


def test_rain_is_requested_automatically_with_the_sar_data(client, monkeypatch):
    ev, calls = rain_stream(client, monkeypatch)  # no rain parameter: it is on by default
    names = [e for e, _ in ev]
    assert calls == [(-60.0217, -3.119)] and ev[0][1]["rain"] is True
    assert names.count("rows") == 3 and names.count("rain") >= 2  # the 50 cached days, then batches
    days = [r for e, d in ev if e == "rain" for r in d["rows"]]
    assert len(days) == 103 and len({r["t"] for r in days}) == 103
    assert days[0] == {"t": 1701388800000, "mm": 1.0, "run": "final"}  # 2023-12-01 00:00 UTC, in ms
    first = next(d for e, d in ev if e == "rain")
    assert first["source"] == "cache" and first["total"] == 103 and len(first["rows"]) == 50


def test_meta_is_always_the_first_event_even_when_rain_is_instant(client, monkeypatch):
    import time
    real = lambda *a, **k: list(SCENES)
    monkeypatch.setattr(api, "search", lambda *a, **k: (time.sleep(0.3), real(*a, **k))[1])  # slow ASF search
    ev, _ = rain_stream(client, monkeypatch)  # the cached rain is ready immediately
    assert ev[0][0] == "meta" and ev[1][0] in ("rain", "rows")
    assert [e for e, _ in ev].count("rain") >= 2  # nothing was lost by holding it back


def test_done_is_sent_last_after_all_the_rain(client, monkeypatch):
    ev, _ = rain_stream(client, monkeypatch)
    names = [e for e, _ in ev]
    assert names[-1] == "done" and names[-2] == "sceneRain"
    assert max(i for i, n in enumerate(names) if n == "rain") < names.index("sceneRain")
    assert ev[-1][1]["rainDays"] == 103 and ev[-1][1]["rainFailed"] == 0 and ev[-1][1]["fetched"] == 3


def test_each_scene_gets_the_rain_before_it(client, monkeypatch):
    ev, _ = rain_stream(client, monkeypatch)
    rows = {r["scene"]: r for e, d in ev if e == "sceneRain" for r in d["rows"]}
    assert set(rows) == {"S1", "S2", "S3"}
    s1 = rows["S1"]  # acquired 2024-01-10; rain = day of month
    assert s1 == {"scene": "S1", "rain0": 10.0, "rain1d": 9.0, "rain3d": 24.0, "rain7d": 42.0, "rain30d": 486.0}


def test_rain_false_skips_imerg(client, monkeypatch):
    calls = []
    monkeypatch.setattr(api.imerg, "iter_daily", make_rain(calls=calls))
    ev = parse(client.get(URL + "&rain=false").text)
    names = [e for e, _ in ev]
    assert calls == [] and ev[0][1]["rain"] is False
    assert "rain" not in names and "sceneRain" not in names and "rainDays" not in ev[-1][1]


@pytest.mark.parametrize("fail,message", [("access", "authorize NASA GESDISC"), ("mid", "cmr down")])
def test_a_rain_failure_never_breaks_the_sar_stream(client, monkeypatch, fail, message):
    ev, _ = rain_stream(client, monkeypatch, fail=fail)
    names = [e for e, _ in ev]
    assert names.count("rows") == 3 and names[-1] == "done" and ev[-1][1]["fetched"] == 3
    err = [d for e, d in ev if e == "rain_error"]
    assert len(err) == 1 and message in err[0]["message"]
    assert "fatal" not in names
    if fail == "mid":  # what arrived before the failure is still delivered
        assert sum(len(d["rows"]) for e, d in ev if e == "rain") >= 50


def test_blocking_endpoint_returns_the_rain_too(client, monkeypatch):
    monkeypatch.setattr(api.imerg, "iter_daily", make_rain())
    j = client.get("/series?lon=-60.0217&lat=-3.119&start=2018-01-01&end=2024-12-31&workers=2").json()
    assert len(j["rows"]) == 6 and len(j["rain"]) == 103 and j["rainError"] is None
    assert [r["t"] for r in j["rain"]] == sorted(r["t"] for r in j["rain"])
    assert len(j["sceneRain"]) == 3 and j["rainDays"] == 103
    off = client.get("/series?lon=-60.0217&lat=-3.119&start=2018-01-01&end=2024-12-31&rain=false").json()
    assert "rain" not in off and len(off["rows"]) == 6


def test_a_sar_failure_is_fatal_and_reported(client, monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("ASF is down")
    monkeypatch.setattr(api, "search", boom)
    monkeypatch.setattr(api.imerg, "iter_daily", make_rain())
    ev = parse(client.get(URL).text)
    assert ev[-1][0] == "fatal" and "ASF is down" in ev[-1][1]["message"]


# ---- automatic dataset switching through the API --------------------------------------------
@pytest.mark.parametrize("start,end,requested,expected", [
    ("2026-06-01", "2026-11-13", "auto", "NISAR"),     # inside NISAR coverage
    ("2018-01-01", "2023-12-31", "auto", "OPERA"),     # historical
    ("2025-01-01", "2026-11-13", "auto", "OPERA"),     # spans the boundary: one continuous series
    ("2026-06-01", "2026-11-13", "OPERA", "OPERA"),    # forced
    ("2018-01-01", "2023-12-31", "NISAR", "NISAR"),    # forced, even without coverage
])
def test_dataset_is_switched_automatically_from_the_date_range(client, monkeypatch, start, end, requested, expected):
    asked = []
    monkeypatch.setattr(api, "search", lambda kind, *a, **k: asked.append(kind) or list(SCENES))
    ev = parse(client.get(f"/series/stream?lon=-60&lat=-3&start={start}&end={end}&dataset={requested}&rain=false").text)
    assert ev[0][1]["dataset"] == expected and asked == [expected] and ev[0][1]["reason"]


def test_dataset_defaults_to_auto_with_the_recent_window(client, monkeypatch):
    asked = []
    monkeypatch.setattr(api, "search", lambda kind, *a, **k: asked.append(kind) or list(SCENES))
    parse(client.get("/series/stream?lon=-60&lat=-3&rain=false").text)  # what the front-end sends today
    assert asked == ["NISAR"]


# ---- nearby clicks share one cache file ----------------------------------------------------
def test_points_a_few_metres_apart_share_the_cache(client):
    first = parse(client.get("/series/stream?lon=-60.021734&lat=-3.119012&start=2018-01-01&end=2024-12-31").text)
    assert first[0][1]["lon"] == -60.0217 and first[0][1]["lat"] == -3.119  # the point actually processed
    assert first[0][1]["toFetch"] == 3
    second = parse(client.get("/series/stream?lon=-60.021701&lat=-3.118996&start=2018-01-01&end=2024-12-31").text)
    assert second[0][1]["cached"] == 3 and second[0][1]["toFetch"] == 0
