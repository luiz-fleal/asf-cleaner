"""API tests: no network and no token (ASF search and the per-scene workers are replaced)."""
import json

import pytest
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
