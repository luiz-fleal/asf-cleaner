"""Unit tests for collecting the forecast history (monthly radar scene picking and monthly rain)."""
from datetime import date

import numpy as np
import pandas as pd
import pytest

from asf_cleaner import history, imerg
from asf_cleaner.config import PROCESSING_VERSION


def scene(day, track=10, burst="B1", direction="DESCENDING"):
    return {"scene": f"S{day}-{track}-{burst}", "start_time": f"{day}T09:00:00Z", "track": track, "burst_id": burst, "direction": direction, "tifs": {}}


# ---- radar scene choice ----------------------------------------------------------------------
def test_pick_one_scene_per_month_closest_to_the_15th():
    scenes = [scene("2020-01-03"), scene("2020-01-14"), scene("2020-01-27"), scene("2020-02-20"), scene("2020-02-09")]
    picked, geom = history.pick_monthly_scenes(scenes)
    assert [s["start_time"][:10] for s in picked] == ["2020-01-14", "2020-02-09"] or [s["start_time"][:10] for s in picked] == ["2020-01-14", "2020-02-20"]
    assert geom == {"burst": "B1", "track": 10, "direction": "DESCENDING"}
    assert len({s["start_time"][:7] for s in picked}) == len(picked)


def test_pick_uses_only_the_busiest_geometry():
    a = [scene(f"2020-0{m}-10", track=10) for m in range(1, 6)]
    b = [scene(f"2020-0{m}-12", track=99, burst="B2") for m in range(1, 3)]
    picked, geom = history.pick_monthly_scenes(a + b)
    assert geom["track"] == 10 and all(s["track"] == 10 for s in picked) and len(picked) == 5


def test_pick_with_no_scenes():
    assert history.pick_monthly_scenes([]) == ([], {})


# ---- radar history ---------------------------------------------------------------------------
def opera_row(sc, pol, db):
    return {"sensor": "OPERA", "scene": sc["scene"], "burst_id": sc["burst_id"], "track": sc["track"], "direction": sc["direction"],
            "start_time": sc["start_time"], "platform": "S1A", "polarization": pol, "backscatter_db_median": db, "backscatter_db_mean": db,
            "backscatter_db_center": db, "valid_fraction": 1.0, "window_px": "17x17", "window_m": 500, "pixel_spacing_m": 30.0,
            "normalization": "gamma0", "processing_version": PROCESSING_VERSION}


def test_radar_history_reads_each_scene_once_and_reuses_the_cache(tmp_path, monkeypatch):
    scenes = [scene(f"{y}-{m:02d}-14") for y in (2020, 2021) for m in range(1, 13)]
    calls = []

    def fake(sc, *, lon, lat, window_m, token):
        calls.append(sc["scene"])
        return [opera_row(sc, "VV", -8.0), opera_row(sc, "VH", -14.0)]

    monkeypatch.setattr(history, "process_opera", fake)
    monkeypatch.setattr(history, "get_token", lambda: "t")

    def run():
        return list(history.radar_history(-60.0, -3.0, search_fn=lambda *a: scenes, cache_dir=str(tmp_path), workers=2))

    ev = run()
    rows, geometry, n = ev[-1][1]
    assert ev[-1][0] == "data" and n == 24 and len(calls) == 24 and len(rows) == 48
    assert [e for e, p in ev if e == "progress" and p["phase"] == "radar"][-1:] and ev[-2][1]["done"] == 24
    run()
    assert len(calls) == 24  # the second run came from the cache


def test_radar_history_survives_failed_scenes(tmp_path, monkeypatch):
    scenes = [scene(f"2020-{m:02d}-14") for m in range(1, 7)]

    def fake(sc, *, lon, lat, window_m, token):
        if sc["scene"].startswith("S2020-03"):
            raise OSError("down")
        return [opera_row(sc, "VV", -8.0)]

    monkeypatch.setattr(history, "process_opera", fake)
    monkeypatch.setattr(history, "get_token", lambda: "t")
    monkeypatch.setattr("asf_cleaner.runner.time.sleep", lambda s: None)
    ev = list(history.radar_history(-60.0, -3.0, search_fn=lambda *a: scenes, cache_dir=str(tmp_path), workers=2))
    assert len(ev[-1][1][0]) == 5


def test_radar_history_without_scenes(tmp_path, monkeypatch):
    monkeypatch.setattr(history, "get_token", lambda: "t")
    ev = list(history.radar_history(-60.0, -3.0, search_fn=lambda *a: [], cache_dir=str(tmp_path)))
    assert ev[-1][0] == "data" and ev[-1][1][0].empty and ev[-1][1][2] == 0


# ---- monthly rain ------------------------------------------------------------------------------
def fake_imerg(monkeypatch, monthly_mm=100.0, fail_days=()):
    monthly_calls, day_calls = [], []

    def list_granules(run, d0, d1):
        assert run == "monthly"
        months = pd.period_range(d0, d1, freq="M")
        return {m.start_time.date(): {"granule": f"M{m}", "url": "u"} for m in months if m.start_time.date() <= date(2025, 9, 1)}

    def fetch_monthly(item, *, i, j, token):
        monthly_calls.append(item["granule"])
        return [{"month": f"{item['date']:%Y-%m}", "mm": monthly_mm, "source": "monthly", "processing_version": PROCESSING_VERSION}]

    def plan_days(d0, d1):
        days = pd.date_range(d0, d1)
        return [{"date": d.date(), "run": "late", "granule": f"D{d.date()}", "url": "u"} for d in days], []

    def fetch_point(item, *, i, j, token):
        day_calls.append(item["date"])
        if item["date"] in fail_days:
            raise OSError("down")
        return [{"date": item["date"].isoformat(), "precipitation_mm_day": 2.0, "imerg_run": "late", "granule": "g", "grid_lon": 0, "grid_lat": 0,
                 "units": "mm/day", "processing_version": PROCESSING_VERSION}]

    monkeypatch.setattr(imerg, "list_granules", list_granules)
    monkeypatch.setattr(imerg, "fetch_monthly_point", fetch_monthly)
    monkeypatch.setattr(imerg, "plan_days", plan_days)
    monkeypatch.setattr(imerg, "fetch_point", fetch_point)
    monkeypatch.setattr(history, "get_token", lambda: "t")
    monkeypatch.setattr("asf_cleaner.runner.time.sleep", lambda s: None)
    return monthly_calls, day_calls


def test_rain_history_joins_monthly_product_and_daily_sums(tmp_path, monkeypatch):
    monthly_calls, day_calls = fake_imerg(monkeypatch)
    ev = list(history.rain_history(-60.0, -3.0, cache_dir=str(tmp_path), today=date(2026, 3, 20), workers=4))
    s = ev[-1][1]
    assert ev[-1][0] == "data" and s.index.freqstr == "M"
    assert s[pd.Period("2020-05", "M")] == 100.0  # from the monthly product
    assert s[pd.Period("2025-11", "M")] == pytest.approx(2.0 * 30)  # daily Late summed over a complete month
    assert s.index.max() == pd.Period("2026-02", "M")  # March is not complete yet
    assert not any(d.month == 3 and d.year == 2026 for d in day_calls)


def test_rain_history_caches_both_sources(tmp_path, monkeypatch):
    monthly_calls, day_calls = fake_imerg(monkeypatch)
    kw = dict(cache_dir=str(tmp_path), today=date(2026, 3, 20), workers=4)
    list(history.rain_history(-60.0, -3.0, **kw))
    n_m, n_d = len(monthly_calls), len(day_calls)
    assert n_m > 80 and n_d > 100
    list(history.rain_history(-60.0, -3.0, **kw))
    assert len(monthly_calls) == n_m and len(day_calls) == n_d


def test_rain_history_drops_months_with_too_many_missing_days(tmp_path, monkeypatch):
    bad = {date(2025, 12, d) for d in range(1, 16)}  # half of December is missing
    fake_imerg(monkeypatch, fail_days=bad)
    s = list(history.rain_history(-60.0, -3.0, cache_dir=str(tmp_path), today=date(2026, 3, 20), workers=4))[-1][1]
    assert pd.Period("2025-12", "M") not in s.index and pd.Period("2025-11", "M") in s.index


def test_rain_history_reports_a_rejected_token(tmp_path, monkeypatch):
    fake_imerg(monkeypatch)

    def denied(item, *, i, j, token):
        raise PermissionError("authorize GES DISC")

    monkeypatch.setattr(imerg, "fetch_monthly_point", denied)
    with pytest.raises(RuntimeError, match="GES DISC"):
        list(history.rain_history(-60.0, -3.0, cache_dir=str(tmp_path), today=date(2026, 3, 20)))


# ---- monthly point fetch -------------------------------------------------------------------------
def test_fetch_monthly_point_converts_rate_to_month_total(monkeypatch):
    class Resp:
        status_code, content, text = 200, b"x", ""

        def raise_for_status(self):
            pass

    class Sess:
        def get(self, url, **k):
            assert "/Grid/precipitation" in url
            return Resp()

    monkeypatch.setattr(imerg, "get_session", lambda: Sess())
    monkeypatch.setattr(imerg, "parse_point", lambda c: {"precip": 0.2, "units": "mm/hr", "lon": 0.0, "lat": 0.0})
    row = imerg.fetch_monthly_point({"date": date(2024, 2, 1), "granule": "g", "url": "https://h/x"}, i=1, j=2, token="t")[0]
    assert row["month"] == "2024-02" and row["mm"] == pytest.approx(0.2 * 29 * 24)  # leap February
    monkeypatch.setattr(imerg, "parse_point", lambda c: {"precip": float("nan"), "units": "mm/hr", "lon": 0.0, "lat": 0.0})
    assert np.isnan(imerg.fetch_monthly_point({"date": date(2024, 2, 1), "granule": "g", "url": "https://h/x"}, i=1, j=2, token="t")[0]["mm"])


def test_monthly_point_url_uses_the_grid_group():
    u = imerg.monthly_point_url("https://h/x.nc4", 10, 20)
    assert "%5B" not in u and "/Grid/precipitation[0][10][20]" in u.replace("%2F", "/") or "Grid" in u
