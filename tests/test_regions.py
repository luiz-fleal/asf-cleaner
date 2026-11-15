"""Ready-made regional forecasts: averaging the sample places, the files and the endpoints (no network)."""
import json

import pytest
from fastapi.testclient import TestClient

from asf_cleaner import api, regions

T = [1_700_000_000_000 + i * 2_600_000_000 for i in range(4)]  # four months, in ms


def series(offset, pol="VV"):
    pts = lambda base: [{"t": t, "v": base + i * 0.1, "lo": base + i * 0.1 - 0.5, "hi": base + i * 0.1 + 0.5} for i, t in enumerate(T)]  # noqa: E731
    return {"model": "seasonal", "reliable": True, "pol": pol, "trendPerDecade": offset, "bandWidth": 0.8,
            "history": [{"t": t - 10**11, "v": -8 + offset} for t in T], "forecast": pts(-8 + offset),
            "backtest": {"mae": 0.3 + offset / 10, "rmse": 0.4, "naiveMae": 0.5, "skill": 0.4, "folds": 3, "months": 36, "candidates": {}},
            "yearly": []}


def risk(level, score, main="drought_fire", cover="forest"):
    items = [{"id": "drought_fire", "applicable": True, "score": score, "level": level, "evidence": [{"text": "x", "vars": {}}]},
             {"id": "vegetation_loss", "applicable": cover != "urban", "score": 0.1, "level": "low" if cover != "urban" else "n/a", "evidence": []}]
    return {"available": True, "risks": items, "overall": {"score": score, "level": level, "main": main}, "landCover": {"class": cover, "confidence": 0.9}}


def result(offset, level, score, cover="forest"):
    return {"until": "2030-12", "rainScale": 1.0, "backscatter": {"co": series(offset), "cross": series(offset - 6, "VH")}, "rain": series(offset * 100 + 150),
            "risk": risk(level, score, cover=cover), "notes": ["a", "b"], "meta": {"radarScenes": 100, "rainMonths": 120}}


SAMPLES = [("Calm", -60.0, -3.0), ("Dry", -55.0, -7.0)]


def test_aggregate_averages_series_and_keeps_the_riskiest_sample():
    out = regions.aggregate([result(0.0, "low", 0.1), result(1.0, "high", 0.8)], SAMPLES)
    co = out["backscatter"]["co"]
    assert co["model"] == "average of 2 places" and co["places"] == 2
    assert co["forecast"][0]["v"] == pytest.approx((-8 + -7) / 2) and co["forecast"][0]["lo"] < co["forecast"][0]["v"] < co["forecast"][0]["hi"]
    assert co["trendPerDecade"] == pytest.approx(0.5) and co["backtest"]["mae"] == pytest.approx(0.35)
    drought = next(r for r in out["risk"]["risks"] if r["id"] == "drought_fire")
    assert drought["score"] == 0.8 and drought["place"] == "Dry" and drought["meanScore"] == pytest.approx(0.45)  # the hot spot is not averaged away
    assert out["risk"]["overall"] == {"score": 0.8, "level": "high", "main": "drought_fire", "place": "Dry"}
    assert [s["name"] for s in out["samples"]] == ["Calm", "Dry"] and out["samples"][1]["level"] == "high"
    assert out["meta"] == {"radarScenes": 200, "rainMonths": 120}
    assert out["samples"][0]["years"] == {}  # these fake series have no complete year


def test_a_risk_that_does_not_apply_anywhere_stays_not_applicable():
    out = regions.aggregate([result(0, "low", 0.1, "urban"), result(0, "low", 0.2, "urban")], SAMPLES)
    veg = next(r for r in out["risk"]["risks"] if r["id"] == "vegetation_loss")
    assert veg["applicable"] is False and veg["level"] == "n/a"
    assert out["risk"]["landCover"]["class"] == "urban" and out["risk"]["landCover"]["confidence"] == 1.0


def test_aggregate_needs_at_least_one_result():
    with pytest.raises(ValueError):
        regions.aggregate([], [])


def test_build_writes_a_file_skips_failures_and_can_skip_finished_regions(tmp_path, monkeypatch):
    monkeypatch.setattr(regions, "REGIONS_DIR", tmp_path)
    monkeypatch.setitem(regions.REGIONS, "test", {"name": "Test", "center": [-60, -3], "zoom": 5, "bounds": [-61, -4, -59, -2], "samples": [*SAMPLES, ("Broken", 0.0, 0.0)]})
    calls = []

    def compute(lon, lat):
        calls.append((lon, lat))
        if lon == 0.0:
            raise RuntimeError("no scenes")
        return result(0.5, "moderate", 0.4)

    data = regions.build("test", compute)
    assert len(calls) == 3 and data["samples"] and len(data["samples"]) == 2 and data["region"]["name"] == "Test" and "builtAt" in data
    assert json.loads((tmp_path / "test.json").read_text())["risk"]["overall"]["level"] == "moderate"
    regions.build("test", compute, only_missing=True)
    assert len(calls) == 3  # the file was already there


def test_the_endpoints_list_and_serve_regions(tmp_path, monkeypatch):
    monkeypatch.setattr(regions, "REGIONS_DIR", tmp_path)
    client = TestClient(api.app)
    listed = {r["id"]: r for r in client.get("/regions").json()}
    assert set(listed) == set(regions.REGIONS) and not any(r["ready"] for r in listed.values())
    assert client.get("/regions/amazon").status_code == 404
    assert client.get("/regions/nowhere").status_code == 404
    (tmp_path / "amazon.json").write_text(json.dumps(regions.aggregate([result(0, "low", 0.2)], SAMPLES[:1])))
    assert client.get("/regions/amazon").json()["risk"]["overall"]["level"] == "low"
    assert next(r for r in client.get("/regions").json() if r["id"] == "amazon")["ready"] is True


def test_every_region_has_samples_inside_its_bounds():
    for rid, r in regions.REGIONS.items():
        w, s, e, n = r["bounds"]
        assert r["samples"], rid
        for name, lon, lat in r["samples"]:
            assert w <= lon <= e and s <= lat <= n, (rid, name)
