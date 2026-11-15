from datetime import date, datetime, timedelta, timezone

import h5py
import numpy as np
import pandas as pd
import pytest

from asf_cleaner import imerg

UTC = timezone.utc


# ---- grid and parsing ----------------------------------------------------------------------
def test_grid_index():
    assert imerg.grid_index(-60.025, -1.74118) == (1199, 882)  # cell centre -60.05, -1.75
    assert imerg.grid_index(180, 90) == (3599, 1799) and imerg.grid_index(-180, -90) == (0, 0)
    with pytest.raises(ValueError):
        imerg.grid_index(200, 0)


def nc4_bytes(tmp_path, value=12.5, lon=-60.05, lat=-1.75, fill=-9999.9):
    p = tmp_path / "pt.nc4"
    with h5py.File(p, "w") as f:
        d = f.create_dataset("precipitation", data=np.array([[[value]]], "float32"))
        d.attrs["units"], d.attrs["_FillValue"] = "mm/day", np.float32(fill)
        f["lon"], f["lat"] = np.array([lon], "float32"), np.array([lat], "float32")
    return p.read_bytes()


def test_parse_point(tmp_path):
    r = imerg.parse_point(nc4_bytes(tmp_path))
    assert r["precip"] == 12.5 and r["units"] == "mm/day"
    assert np.isclose(r["lon"], -60.05) and np.isclose(r["lat"], -1.75)
    assert np.isnan(imerg.parse_point(nc4_bytes(tmp_path, value=-9999.9))["precip"])
    with pytest.raises(ValueError, match="NetCDF-4"):
        imerg.parse_point(b"<html>login</html>")


def test_point_url():
    u = imerg.point_url("https://h/x.nc4", 1199, 882)
    assert u.startswith("https://h/x.nc4.dap.nc4?dap4.ce=") and "precipitation[0][1199][882]" in u


# ---- choosing Final vs Late ----------------------------------------------------------------
def fake_granules(monkeypatch, final_until, late_from, late_until):
    def lg(run, d0, d1):
        lo, hi = (d0, min(d1, final_until)) if run == "final" else (max(d0, late_from), min(d1, late_until))
        return {lo + timedelta(days=k): {"granule": f"{run}-{lo + timedelta(days=k)}", "url": "u"}
                for k in range((hi - lo).days + 1)} if hi >= lo else {}
    monkeypatch.setattr(imerg, "list_granules", lg)


def test_plan_days_final_then_late(monkeypatch):
    fake_granules(monkeypatch, date(2025, 9, 30), date(2020, 1, 1), date(2025, 10, 4))
    days, missing = imerg.plan_days(date(2025, 9, 28), date(2025, 10, 6))
    assert [d["run"] for d in days] == ["final"] * 3 + ["late"] * 4
    assert missing == [date(2025, 10, 5), date(2025, 10, 6)]  # not published yet


# ---- fetching one cell ---------------------------------------------------------------------
class Resp:
    def __init__(self, code=200, content=b"", text=""):
        self.status_code, self.content, self.text = code, content, text

    def raise_for_status(self):
        if self.status_code >= 400:
            raise OSError(self.status_code)


def with_session(monkeypatch, resp):
    class S:
        def get(self, *a, **k):
            return resp
    monkeypatch.setattr(imerg, "get_session", lambda: S())


ITEM = {"date": date(2026, 1, 5), "run": "late", "granule": "g", "url": "https://h/g"}


def test_fetch_point_ok(monkeypatch, tmp_path):
    with_session(monkeypatch, Resp(content=nc4_bytes(tmp_path)))
    row = imerg.fetch_point(ITEM, i=1199, j=882, token="t")[0]
    assert row["date"] == "2026-01-05" and row["precipitation_mm_day"] == 12.5 and row["imerg_run"] == "late"


@pytest.mark.parametrize("code", [401, 403])
def test_fetch_point_auth_error_has_instructions(monkeypatch, code):
    with_session(monkeypatch, Resp(code))
    with pytest.raises(PermissionError, match="GESDISC"):
        imerg.fetch_point(ITEM, i=1, j=1, token="t")


def test_fetch_point_bad_request_not_retried(monkeypatch):
    with_session(monkeypatch, Resp(400, text="bad constraint"))
    with pytest.raises(ValueError, match="400"):
        imerg.fetch_point(ITEM, i=1, j=1, token="t")


def test_fetch_point_grid_mismatch(monkeypatch, tmp_path):
    with_session(monkeypatch, Resp(content=nc4_bytes(tmp_path, lon=10.0)))
    with pytest.raises(ValueError, match="grid mismatch"):
        imerg.fetch_point(ITEM, i=1199, j=882, token="t")


# ---- the cached daily series ---------------------------------------------------------------
START, END = datetime(2026, 2, 1, tzinfo=UTC), datetime(2026, 2, 10, tzinfo=UTC)


@pytest.fixture
def daily(monkeypatch, tmp_path):
    calls = {"fetched": [], "run": "late", "fail": set(), "deny": False, "plans": [], "tmp": tmp_path}

    def plan(d0, d1):
        calls["plans"].append((d0, d1))
        return [{"date": d0 + timedelta(days=k), "run": calls["run"], "granule": "g", "url": "u"}
                for k in range((d1 - d0).days + 1)], []

    def fp(item, *, i, j, token):
        if calls["deny"]:
            raise PermissionError("denied")
        calls["fetched"].append(item["date"])
        if item["date"] in calls["fail"]:
            raise OSError("flaky")
        return [{"date": item["date"].isoformat(), "precipitation_mm_day": 1.0 + item["date"].day,
                 "imerg_run": item["run"], "granule": "g", "grid_lon": 0.0, "grid_lat": 0.0,
                 "units": "mm/day", "processing_version": imerg.PROCESSING_VERSION}]

    monkeypatch.setattr(imerg, "plan_days", plan)
    monkeypatch.setattr(imerg, "fetch_point", fp)
    calls["run_it"] = lambda **k: imerg.fetch_daily(-60.025, -1.74118, START, END, token_fn=lambda: "t",
                                                    workers=1, out=str(tmp_path), **k)
    return calls


def test_daily_covers_lookback_and_caches(daily):
    df = daily["run_it"]()
    assert df["date"].iloc[0] == "2026-01-02" and df["date"].iloc[-1] == "2026-02-10"  # 30 days before
    assert len(df) == 40 and len(daily["fetched"]) == 40
    daily["fetched"].clear()
    assert len(daily["run_it"]()) == 40 and daily["fetched"] == []  # all cached


def test_late_days_upgraded_when_final_appears(daily):
    daily["run_it"]()
    daily["fetched"].clear()
    daily["run"] = "final"
    df = daily["run_it"]()
    assert len(daily["fetched"]) == 40 and set(df["imerg_run"]) == {"final"}
    daily["fetched"].clear()
    daily["run"] = "late"  # never downgrade a cached Final value
    assert set(daily["run_it"]()["imerg_run"]) == {"final"} and daily["fetched"] == []


def test_rejected_token_stops_immediately(daily):
    daily["deny"] = True
    with pytest.raises(imerg.ImergAccessError):
        daily["run_it"]()


def test_failed_days_not_cached_and_retried(daily):
    daily["fail"] = {date(2026, 2, 3)}
    assert len(daily["run_it"]()) == 39
    daily["fetched"].clear()
    daily["fail"] = set()
    assert len(daily["run_it"]()) == 40 and daily["fetched"] == [date(2026, 2, 3)]


# ---- rain before each acquisition ----------------------------------------------------------
def test_rain_features():
    days = pd.date_range("2026-01-01", "2026-03-01")
    daily_df = pd.DataFrame({"date": days.strftime("%Y-%m-%d"), "precipitation_mm_day": np.arange(len(days), dtype=float)})
    # day index k has rain k mm; acquisition on 2026-02-10 (k = 40) at 09:38 UTC
    sar = pd.DataFrame({"start_time": pd.to_datetime(["2026-02-10T09:38:00Z", "2026-01-03T09:38:00Z"], utc=True)})
    out = imerg.rain_features(sar, daily_df)
    r = out.iloc[0]
    assert r.rain_d0_mm == 40
    assert r.rain_1d_mm == 39                       # day before only
    assert r.rain_3d_mm == 37 + 38 + 39             # three days before, excluding the day itself
    assert r.rain_7d_mm == sum(range(33, 40))
    assert r.rain_30d_mm == sum(range(10, 40))
    assert np.isnan(out.iloc[1].rain_7d_mm) and out.iloc[1].rain_1d_mm == 1  # window incomplete -> NaN


def test_rain_features_missing_day_gives_nan():
    daily_df = pd.DataFrame({"date": ["2026-01-01", "2026-01-02", "2026-01-04", "2026-01-05"],
                             "precipitation_mm_day": [1.0, 2.0, 4.0, 5.0]})
    out = imerg.rain_features(pd.DataFrame({"start_time": pd.to_datetime(["2026-01-05T08:00:00Z"], utc=True)}), daily_df)
    assert np.isnan(out.rain_3d_mm[0]) and out.rain_1d_mm[0] == 4.0 and out.rain_d0_mm[0] == 5.0


# ---- cache key and CMR lookups -------------------------------------------------------------
def test_cache_is_shared_by_every_point_in_a_cell(tmp_path):
    a = imerg.cache_path(tmp_path, -60.021, -1.7405)
    b = imerg.cache_path(tmp_path, -60.049, -1.7499)  # same 0.1 degree cell, different point
    c = imerg.cache_path(tmp_path, -60.2, -1.74)
    assert a == b and a != c


def test_nearby_point_reuses_the_cached_days(daily):
    daily["run_it"]()
    daily["fetched"].clear()
    df = imerg.fetch_daily(-60.049, -1.7499, START, END, token_fn=lambda: "t", workers=1,
                           out=str(daily["tmp"]))
    assert len(df) == 40 and daily["fetched"] == []


def test_fully_final_cache_never_calls_cmr(daily, monkeypatch):
    daily["run"] = "final"
    daily["run_it"]()
    daily["plans"].clear()
    daily["fetched"].clear()
    assert len(daily["run_it"]()) == 40
    assert daily["plans"] == [] and daily["fetched"] == []


def test_cmr_only_looks_up_days_that_are_not_settled(daily):
    daily["run"] = "final"
    daily["run_it"]()
    daily["plans"].clear()
    daily["fetched"].clear()
    later = END + timedelta(days=3)
    df = imerg.fetch_daily(-60.025, -1.74118, START, later, token_fn=lambda: "t", workers=1,
                           out=str(daily["tmp"]))
    assert len(df) == 43
    assert daily["plans"] == [(date(2026, 2, 11), date(2026, 2, 13))]  # just the new days
    assert daily["fetched"] == [date(2026, 2, 11), date(2026, 2, 12), date(2026, 2, 13)]
