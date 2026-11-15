"""API tests: no network and no token (ASF search and the per-scene workers are replaced)."""
import json

import pytest
import requests
from fastapi.testclient import TestClient

from asf_cleaner import api
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


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(api, "search", lambda *a, **k: list(SCENES))
    monkeypatch.setattr(api, "WORKERS", {"OPERA": fake_worker, "NISAR": fake_worker})
    monkeypatch.setattr(api, "get_token", lambda: "tok")
    monkeypatch.setattr(api, "CACHE_DIR", str(tmp_path))
    api._search_cache.clear()  # searches are remembered between requests, and tests fake the search differently
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


def notes_of(ev):
    return {d["code"]: d for e, d in ev if e == "notice"}


def test_stream_tells_the_sensor_chosen_and_judges_the_data(client):
    ev = parse(client.get(URL).text)
    notes = notes_of(ev)
    assert "dataset-opera" in notes and "nisar-partial" not in notes  # 2018..2024: OPERA, chosen from the period alone
    assert "few-scenes" in notes  # 3 scenes: usable, not steady
    done = ev[-1][1]
    assert done["dataset"] == "OPERA" and done["assessment"]["confidence"] == "limited" and done["assessment"]["scenes"] == 3
    names = [e for e, _ in ev]
    assert names[0] == "meta" and names[-1] == "done" and "notice" in names
    assert max(i for i, n in enumerate(names) if n == "notice") < len(names) - 1  # every notice comes before `done`


def test_period_without_scenes_is_reported_as_insufficient(client, monkeypatch):
    monkeypatch.setattr(api, "search", lambda *a, **k: [])
    ev = parse(client.get(URL).text)
    assert "no-scenes" in notes_of(ev) and ev[-1][1]["assessment"]["confidence"] == "insufficient"
    assert ev[-1][0] == "done" and ev[-1][1]["fetched"] == 0


def test_failed_scenes_and_a_cut_short_period_are_reported(client, monkeypatch):
    def flaky(sc, **k):
        if sc["scene"] == "S2":
            raise ValueError("point falls outside the scene")
        return fake_worker(sc, **k)

    def many(*a, **k):
        return api.Scenes(SCENES, 90)  # ASF has 90 matches, only 3 were kept

    monkeypatch.setattr(api, "WORKERS", {"OPERA": flaky, "NISAR": flaky})
    monkeypatch.setattr(api, "search", many)
    ev = parse(client.get(URL + "&cache=false&max_results=3").text)
    notes = notes_of(ev)
    assert notes["scenes-failed"]["vars"] == {"failed": 1, "total": 3}
    assert notes["truncated"]["vars"] == {"matches": 90, "kept": 3}
    assert ev[0][0] == "meta" and ev[0][1]["matches"] == 90


def test_a_failure_before_the_stream_starts_is_put_in_words(client, monkeypatch):
    def down(*a, **k):
        raise requests.ConnectionError("ASF is down")

    monkeypatch.setattr(api, "search", down)
    ev = parse(client.get(URL).text)
    last = ev[-1]
    assert last[0] == "fatal" and last[1]["code"] == "archive-unreachable" and "ASF is down" in last[1]["detail"]
    assert "ConnectionError" not in last[1]["message"]


def test_missing_token_is_a_clear_fatal(client, monkeypatch):
    def no_token():
        raise RuntimeError("EARTHDATA_TOKEN not set. Generate a token ...")

    monkeypatch.setattr(api, "get_token", no_token)
    ev = parse(client.get(URL + "&cache=false").text)
    assert ev[-1][0] == "fatal" and ev[-1][1]["code"] == "token-missing"


def test_blocking_endpoint_matches_the_stream(client):
    j = client.get("/series?lon=-60.0217&lat=-3.119&start=2018-01-01&end=2024-12-31&workers=2").json()
    assert j["meta"]["total"] == 3 and len(j["rows"]) == 6 and j["failed"] == 0
    assert any(n["code"] == "dataset-opera" for n in j["notices"]) and j["assessment"]["scenes"] == 3


# ---- rain (IMERG) ------------------------------------------------------------------------
from datetime import date  # noqa: E402


def rain_client(tmp_path, monkeypatch, fail_with=None):
    days = [{"date": date(2026, 1, d), "run": "late", "granule": f"g{d}", "url": "u"} for d in range(1, 9)]
    monkeypatch.setattr(api.imerg, "plan_days", lambda d0, d1: (days, []))
    calls = []

    def fake_point(item, *, i, j, token):
        calls.append(item["date"])
        if fail_with:
            raise fail_with
        return [{"date": item["date"].isoformat(), "precipitation_mm_day": float(item["date"].day),
                 "imerg_run": "late", "granule": item["granule"], "grid_lon": -60.05, "grid_lat": -3.15,
                 "units": "mm/day", "processing_version": PROCESSING_VERSION}]

    monkeypatch.setattr(api.imerg, "fetch_point", fake_point)
    monkeypatch.setattr(api, "get_token", lambda: "tok")
    monkeypatch.setattr(api, "CACHE_DIR", str(tmp_path))
    return TestClient(api.app), calls


RAIN = "/rain/stream?lon=-60.0217&lat=-3.119&start=2026-01-01&end=2026-01-08&workers=2"


def test_rain_streams_days_and_caches_per_cell(tmp_path, monkeypatch):
    client, calls = rain_client(tmp_path, monkeypatch)
    ev = parse(client.get(RAIN).text)
    assert ev[0][0] == "meta" and ev[-1][0] == "done"
    assert ev[0][1]["total"] == 8 and ev[0][1]["gridLon"] == -60.05
    rows = [r for e, d in ev if e == "rows" for r in d["rows"]]
    assert sorted(r["mm"] for r in rows) == [float(d) for d in range(1, 9)]
    assert len(calls) == 8
    # a second point inside the same 0.1 degree cell is served entirely from the cache
    ev2 = parse(client.get(RAIN.replace("-60.0217", "-60.03")).text)
    assert ev2[0][1]["cached"] == 8 and ev2[0][1]["toFetch"] == 0 and len(calls) == 8
    assert [e for e, _ in ev2] == ["meta", "rows", "done"]


def test_rain_reports_days_it_could_not_read_and_days_not_published(tmp_path, monkeypatch):
    client, _ = rain_client(tmp_path, monkeypatch)
    real = api.imerg.fetch_point

    def flaky(item, **k):
        if item["date"].day == 3:
            raise ValueError("granule missing")
        return real(item, **k)

    monkeypatch.setattr(api.imerg, "fetch_point", flaky)
    monkeypatch.setattr(api.imerg, "plan_days", lambda d0, d1: ([{"date": date(2026, 1, d), "run": "late", "granule": f"g{d}", "url": "u"} for d in range(1, 9)], [date(2026, 1, 9)]))
    ev = parse(client.get(RAIN.replace("end=2026-01-08", "end=2026-01-09")).text)
    notes = notes_of(ev)
    assert notes["rain-missing"]["vars"]["n"] == 1 and notes["rain-failed"]["vars"]["n"] == 1
    assert ev[-1][0] == "done" and ev[-1][1]["failed"] == 1 and ev[-1][1]["days"] == 7


def test_rain_json_matches_stream(tmp_path, monkeypatch):
    client, _ = rain_client(tmp_path, monkeypatch)
    body = client.get("/rain?lon=-60.0217&lat=-3.119&start=2026-01-01&end=2026-01-08").json()
    assert [r["mm"] for r in body["rows"]] == [float(d) for d in range(1, 9)]


def test_rain_rejected_token_is_fatal(tmp_path, monkeypatch):
    client, _ = rain_client(tmp_path, monkeypatch, fail_with=PermissionError("authorize GES DISC"))
    ev = parse(client.get(RAIN).text)
    assert ev[-1][0] == "fatal" and "GES DISC" in ev[-1][1]["message"]


def test_rain_outside_imerg_coverage_is_rejected(client):
    assert client.get("/rain/stream?lon=0&lat=75").status_code == 422


def test_search_is_remembered_between_requests(client, monkeypatch):
    api._search_cache.clear()
    calls = []
    monkeypatch.setattr(api, "search", lambda *a, **k: calls.append(1) or list(SCENES))
    client.get(URL + "&cache=false")
    client.get(URL + "&cache=false")
    assert len(calls) == 1


# ---- forecast ----------------------------------------------------------------------------------
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402


def fake_history(monkeypatch, radar_ok=True):
    idx = pd.period_range("2016-06", "2026-10", freq="M")
    rows = [{"start_time": p.start_time.tz_localize("UTC") + pd.Timedelta(days=10), "polarization": pol, "pol_class": cls,
             "valid_fraction": 1.0, "backscatter_db_median": -8 + off + 1.2 * np.sin(2 * np.pi * p.month / 12)}
            for pol, cls, off in (("VV", "co", 0.0), ("VH", "cross", -6.0)) for p in idx]
    rain = pd.Series(200 + 150 * np.cos(2 * np.pi * (idx.month - 3) / 12), index=idx)

    def radar(*a, **k):
        yield "progress", {"phase": "radar", "done": 1, "total": 1}
        yield "data", (pd.DataFrame(rows) if radar_ok else pd.DataFrame(), {"track": 10}, len(idx))

    def rain_h(*a, **k):
        yield "progress", {"phase": "rain-monthly", "done": 1, "total": 1}
        yield "data", rain

    monkeypatch.setattr(api.history, "radar_history", radar)
    monkeypatch.setattr(api.optical, "features", lambda lon, lat: {k: 0.0 for k in api.optical.FEATURES})
    monkeypatch.setattr(api.risk, "load_model", lambda *args, **kw: None)  # tests never depend on the trained file
    monkeypatch.setattr(api.history, "rain_history", rain_h)
    api._forecast_cache.clear()


FORECAST = "/forecast/stream?lon=-60.0217&lat=-3.119&until=2030-12"


def test_forecast_streams_progress_then_result(client, monkeypatch):
    fake_history(monkeypatch)
    ev = parse(client.get(FORECAST).text)
    names = [e for e, _ in ev]
    assert names[0] == "meta" and names[-1] == "done" and "progress" in names
    result = next(d for e, d in ev if e == "result")
    assert set(result["backscatter"]) == {"co", "cross"} and result["rain"]["forecast"][-1]["t"] > result["rain"]["history"][-1]["t"]
    assert result["meta"]["radarScenes"] > 100 and result["until"] == "2030-12"


def test_forecast_result_is_remembered(client, monkeypatch):
    fake_history(monkeypatch)
    client.get(FORECAST)
    ev = parse(client.get(FORECAST).text)
    assert [e for e, _ in ev] == ["meta", "result", "done"] and ev[-1][1]["cached"] is True


def test_forecast_json_and_bad_input(client, monkeypatch):
    fake_history(monkeypatch)
    assert "backscatter" in client.get("/forecast?lon=-60&lat=-3").json()
    assert client.get("/forecast/stream?lon=-60&lat=-3&until=2030-13").status_code == 422
    assert client.get("/forecast/stream?lon=-60&lat=70").status_code == 422
    assert client.get("/forecast/stream?lon=-60&lat=-3&rain_scale=3").status_code == 422


def test_forecast_without_history_is_a_clear_error(client, monkeypatch):
    fake_history(monkeypatch, radar_ok=False)
    monkeypatch.setattr(api.history, "rain_history", lambda *a, **k: iter([("data", pd.Series(dtype=float))]))
    ev = parse(client.get(FORECAST + "&cache=false").text)
    assert ev[-1][0] == "fatal" and "Not enough history" in ev[-1][1]["message"]
    assert ev[-1][1]["code"] == "insufficient-history"


def test_forecast_window_is_snapped_and_part_of_the_cache_key(client, monkeypatch):
    seen = []
    real = api.history.radar_history

    def spy(*a, **k):
        seen.append(k.get("window_m"))
        return real(*a, **k)

    fake_history(monkeypatch)
    real = api.history.radar_history
    monkeypatch.setattr(api.history, "radar_history", spy)
    client.get("/forecast?lon=-60&lat=-3&window_m=1700")
    client.get("/forecast?lon=-60&lat=-3&window_m=2400")  # snaps to the same 2000 m: served from memory
    client.get("/forecast?lon=-60&lat=-3&window_m=100")  # a different window, so a new history
    assert seen == [2000, 500]
