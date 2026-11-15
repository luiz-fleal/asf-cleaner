"""Forecast tests on synthetic data with a known answer (no network)."""
import numpy as np
import pandas as pd
import pytest

from asf_cleaner import forecast as fc


def monthly(start="2016-01", end="2026-10", f=None, noise=0.0, seed=0, gaps=0.0):
    idx = pd.period_range(start, end, freq="M")
    rng = np.random.default_rng(seed)
    t = np.arange(len(idx))
    m = idx.month.to_numpy()
    y = f(t, m) + rng.normal(0, noise, len(idx))
    s = pd.Series(y, index=idx)
    if gaps:
        s = s[rng.random(len(s)) > gaps]
    return s


def backscatter(t, m):  # dB: -8, cycle of 1.5 dB, falling 0.4 dB per year
    return -8 + 1.5 * np.sin(2 * np.pi * m / 12) - 0.4 * t / 12


def rain(t, m):  # mm per month: wet season peaks in March
    return 200 + 150 * np.cos(2 * np.pi * (m - 3) / 12)


def test_recovers_cycle_and_trend():
    y = monthly(f=backscatter, noise=0.15)
    out = fc.forecast_series(y, fc.BACKSCATTER_SPECS, pd.Period("2030-12"), name="vv")
    assert out["model"] in ("seasonal+trend",)  # the trend must be found
    assert out["trendPerDecade"] == pytest.approx(-4.0, abs=0.6)
    last = out["forecast"][-1]
    truth = backscatter(np.array([(pd.Period("2030-12") - pd.Period("2016-01")).n]), np.array([12]))[0]
    assert last["v"] == pytest.approx(truth, abs=0.5)
    assert last["lo"] < last["v"] < last["hi"]
    assert out["backtest"]["skill"] > 0 and out["reliable"] is True
    assert out["forecast"][0]["t"] > out["history"][-1]["t"]


def test_band_widens_with_the_horizon():
    y = monthly(f=backscatter, noise=0.3)
    f = fc.forecast_series(y, fc.BACKSCATTER_SPECS, pd.Period("2030-12"), name="vv")["forecast"]
    assert (f[-1]["hi"] - f[-1]["lo"]) > (f[0]["hi"] - f[0]["lo"])


def test_band_covers_the_noiseless_truth():
    y = monthly(f=backscatter, noise=0.3, seed=1)
    out = fc.forecast_series(y, fc.BACKSCATTER_SPECS, pd.Period("2026-12"), name="vv")
    for pt in out["forecast"]:
        p = pd.Timestamp(pt["t"], unit="ms")
        t = (pd.Period(p, freq="M") - pd.Period("2016-01")).n
        assert pt["lo"] <= backscatter(t, p.month) <= pt["hi"]


def test_gaps_are_tolerated():
    y = monthly(f=backscatter, noise=0.2, gaps=0.3, seed=2)
    out = fc.forecast_series(y, fc.BACKSCATTER_SPECS, pd.Period("2030-12"), name="vv")
    assert len(out["forecast"]) == (pd.Period("2030-12") - y.index.max()).n


def test_too_little_history_is_refused():
    with pytest.raises(fc.InsufficientHistory):
        fc.forecast_series(monthly("2024-01", "2026-10", f=backscatter), fc.BACKSCATTER_SPECS, pd.Period("2030-12"))
    with pytest.raises(fc.InsufficientHistory):  # span is long but almost empty
        fc.forecast_series(monthly(f=backscatter, gaps=0.8, seed=3), fc.BACKSCATTER_SPECS, pd.Period("2030-12"))


def test_end_must_be_after_the_data():
    with pytest.raises(ValueError):
        fc.forecast_series(monthly(f=backscatter), fc.BACKSCATTER_SPECS, pd.Period("2026-01"))


def test_rain_forecast_is_positive_and_seasonal():
    r = monthly(f=rain, noise=25, seed=4).clip(lower=0)
    out = fc.forecast_series(r, fc.RAIN_SPECS, pd.Period("2030-12"), log_target=True, name="rain")
    assert all(p["lo"] >= 0 and p["v"] >= 0 for p in out["forecast"])
    by_month = {pd.Timestamp(p["t"], unit="ms").month: p["v"] for p in out["forecast"][-12:]}
    assert by_month[3] > by_month[9] * 2  # wet March vs dry September
    assert {y["year"] for y in out["yearly"]} >= {2027, 2030}


def test_rain_scale_scenario_scales_the_rain_forecast():
    r = monthly(f=rain, noise=25, seed=5).clip(lower=0)
    base = fc.forecast_series(r, fc.RAIN_SPECS, pd.Period("2028-12"), log_target=True, name="rain")
    wet = fc.forecast_series(r, fc.RAIN_SPECS, pd.Period("2028-12"), log_target=True, rain_scale=1.2, name="rain")
    assert wet["forecast"][5]["v"] == pytest.approx(base["forecast"][5]["v"] * 1.2, rel=1e-6)


def test_rain_scenario_moves_a_rain_driven_backscatter_forecast():
    r = monthly(f=rain, noise=60, seed=6).clip(lower=0)
    anom = fc.rain_anomaly(r, r.index.max())
    y = monthly(f=lambda t, m: -9 + 0.0 * t, noise=0.05, seed=7) + 1.2 * anom.reindex(r.index).fillna(0)
    future = pd.period_range("2026-11", "2027-10", freq="M")
    spec = fc.Spec("rain", trend=True, rain=True)
    dry, _ = fc._fit_predict(y, spec, anom, future, pd.Series(-1.0, index=future))
    wet, _ = fc._fit_predict(y, spec, anom, future, pd.Series(1.0, index=future))
    assert (wet - dry).mean() == pytest.approx(2.4, abs=0.4)  # coefficient 1.2 per unit of anomaly


def test_run_forecast_end_to_end_from_scene_rows():
    idx = pd.period_range("2016-06", "2026-10", freq="M")
    t = np.arange(len(idx))
    rows = []
    for pol, cls, off in (("VV", "co", 0.0), ("VH", "cross", -6.0)):
        for p, ti in zip(idx, t):
            rows.append({"start_time": p.start_time.tz_localize("UTC") + pd.Timedelta(days=10), "polarization": pol, "pol_class": cls,
                         "valid_fraction": 1.0, "backscatter_db_median": backscatter(ti, p.month) + off})
    out = fc.run_forecast(pd.DataFrame(rows), monthly("2016-01", "2026-10", f=rain, noise=20).clip(lower=0), until="2030-12")
    assert set(out["backscatter"]) == {"co", "cross"} and out["backscatter"]["co"]["pol"] == "VV"
    assert out["rain"]["forecast"][-1]["t"] > out["rain"]["history"][-1]["t"]
    assert any("Future rain" in n for n in out["notes"])


def test_low_quality_scenes_are_ignored():
    rows = pd.DataFrame({
        "start_time": pd.to_datetime(["2020-01-10", "2020-01-20"], utc=True), "polarization": ["VV", "VV"], "pol_class": ["co", "co"],
        "valid_fraction": [0.3, 0.95], "backscatter_db_median": [-30.0, -8.0],
    })
    s = fc.monthly_backscatter(rows)["co"][1]
    assert s.iloc[0] == -8.0
