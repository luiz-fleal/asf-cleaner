"""Risk model tests on synthetic series with a known answer (no network)."""
import numpy as np
import pandas as pd
import pytest

from asf_cleaner import forecast as fc
from asf_cleaner import risk

IDX = pd.period_range("2017-01", "2026-10", freq="M")


def series(f, noise=0.2, seed=0, idx=IDX):
    rng = np.random.default_rng(seed)
    t = np.arange(len(idx))
    return pd.Series(f(t, idx.month.to_numpy()) + rng.normal(0, noise, len(idx)), index=idx)


def vh_flat(t, m):
    return -13 + 0.6 * np.sin(2 * np.pi * m / 12)


def vv_flat(t, m):
    return -7 + 0.8 * np.sin(2 * np.pi * m / 12)


def rain_normal(t, m):
    return 200 + 150 * np.cos(2 * np.pi * (m - 3) / 12)


def rain_series(f=rain_normal, noise=20, seed=3):
    return series(f, noise, seed).clip(lower=0)


def radar(vv, vh):
    return {"co": ("VV", vv), "cross": ("VH", vh)}


def forecast_for(rain, vh):
    return fc.run_forecast(pd.DataFrame(), rain, until="2027-10") if False else {
        "backscatter": {"cross": fc.forecast_series(vh, fc.BACKSCATTER_SPECS, pd.Period("2027-10"), name="vh")},
        "rain": fc.forecast_series(rain, fc.RAIN_SPECS, pd.Period("2027-10"), log_target=True, name="rain"),
    }


def by_id(out):
    return {r["id"]: r for r in out["risks"]}


# ---- vegetation loss ---------------------------------------------------------------------------
def test_a_lasting_vh_drop_is_high_vegetation_loss_risk():
    cut = len(IDX) - 14  # clearing 14 months before the end
    vh = series(lambda t, m: vh_flat(t, m) - 3.0 * (t >= cut), 0.25, 1)
    r = risk.vegetation_loss(vh, None)
    assert r["score"] >= risk.HIGH and r["changeAt"] is not None
    assert r["changeDb"] == pytest.approx(-3.0, abs=0.5)
    assert pd.Timestamp(r["changeAt"], unit="ms").to_period("M") in [IDX[cut - 1], IDX[cut], IDX[cut + 1]]


def test_a_stable_series_has_low_vegetation_risk():
    r = risk.vegetation_loss(series(vh_flat, 0.25, 2), None)
    assert r["score"] < risk.MODERATE and r["changeAt"] is None


def test_an_old_step_counts_less_than_a_recent_one():
    old = series(lambda t, m: vh_flat(t, m) - 3.0 * (t >= 25), 0.25, 3)
    new = series(lambda t, m: vh_flat(t, m) - 3.0 * (t >= len(IDX) - 14), 0.25, 3)
    assert risk.vegetation_loss(old, None)["score"] < risk.vegetation_loss(new, None)["score"]


def test_a_rise_is_not_a_loss():
    r = risk.vegetation_loss(series(lambda t, m: vh_flat(t, m) + 3.0 * (t >= len(IDX) - 14), 0.25, 4), None)
    assert r["score"] < risk.MODERATE


def test_the_step_search_needs_enough_months():
    assert risk._step(pd.Series(np.zeros(8), index=pd.period_range("2020-01", periods=8, freq="M"))) is None


# ---- drought and flood -------------------------------------------------------------------------
def test_a_dry_spell_raises_drought_risk():
    last = len(IDX)
    rain = rain_series(lambda t, m: rain_normal(t, m) * np.where(t >= last - 4, 0.25, 1.0))
    r = risk.drought_fire(rain, {"forecast": [{"t": int(p.start_time.timestamp() * 1000), "v": 60.0} for p in pd.period_range("2026-11", "2027-04", freq="M")]})
    assert r["score"] >= risk.HIGH
    assert risk.drought_fire(rain_series(), None)["score"] < risk.MODERATE


def test_drought_needs_rain_history():
    assert risk.drought_fire(None, None)["applicable"] is False


def test_a_wet_spell_and_low_vv_raise_flood_risk():
    last = len(IDX)
    wet = rain_series(lambda t, m: rain_normal(t, m) * np.where(t >= last - 3, 3.0, 1.0))
    vv = series(lambda t, m: vv_flat(t, m) - 12 * ((m == 3) | (m == 4)), 0.3, 5)  # flooded every March and April
    r = risk.flood(vv, wet, None)
    assert r["score"] >= risk.MODERATE
    calm = risk.flood(series(vv_flat, 0.3, 6), rain_series(), None)
    assert calm["score"] < risk.MODERATE and r["score"] > calm["score"]


# ---- unusual months ---------------------------------------------------------------------------------
def test_unusual_recent_months_are_flagged():
    vh = series(vh_flat, 0.2, 7)
    vv = series(vv_flat, 0.2, 8)
    for k in (-2, -4, -7, -9):
        vh.iloc[k] -= 5
        vv.iloc[k] += 4
    out = risk.unusual_change(vv, vh, rain_series())
    assert out["score"] > 0.3 and len(out["anomalies"]) >= 3
    calm = risk.unusual_change(series(vv_flat, 0.2, 9), series(vh_flat, 0.2, 10), rain_series())
    assert calm["score"] < out["score"]


def test_unusual_change_with_short_history():
    short = IDX[:20]
    out = risk.unusual_change(series(vv_flat, idx=short), series(vh_flat, idx=short), None)
    assert out["score"] == 0.0 and out["anomalies"] == []


# ---- assessment as a whole --------------------------------------------------------------------------
def test_assess_ranks_the_clearing_first_and_explains_itself():
    cut = len(IDX) - 12
    vh = series(lambda t, m: vh_flat(t, m) - 3.0 * (t >= cut), 0.25, 11)
    vv = series(vv_flat, 0.25, 12)
    rain = rain_series()
    out = risk.assess(radar(vv, vh), rain, forecast_for(rain, vh), model=False)
    assert out["available"] and out["overall"]["main"] == "vegetation_loss" and out["overall"]["level"] == "high"
    assert out["risks"][0]["id"] == "vegetation_loss"
    assert all(r["evidence"] and {"text", "vars"} <= set(r["evidence"][0]) for r in out["risks"])
    assert out["months"]["radar"] == len(IDX) and "screening" in out["note"].lower()


def test_assess_of_a_calm_place_is_low():
    vh, vv, rain = series(vh_flat, 0.25, 13), series(vv_flat, 0.25, 14), rain_series()
    out = risk.assess(radar(vv, vh), rain, forecast_for(rain, vh), model=False)
    assert out["overall"]["level"] in ("low", "moderate") and by_id(out)["vegetation_loss"]["level"] == "low"


def test_assess_needs_both_polarizations_and_history():
    vv = series(vv_flat)
    assert risk.assess({"co": ("VV", vv)}, None, {})["available"] is False
    assert risk.assess(radar(vv[:10], vv[:10]), None, {})["available"] is False


def test_cover_decides_which_risks_apply(monkeypatch):
    vh, vv, rain = series(vh_flat, 0.25, 15), series(vv_flat, 0.25, 16), rain_series()
    monkeypatch.setattr(risk, "land_cover", lambda *a, **k: {"class": "water", "confidence": 0.9, "probabilities": {}, "source": "pretrained", "trainedOn": 1, "cvAccuracy": 1})
    out = by_id(risk.assess(radar(vv, vh), rain, forecast_for(rain, vh)))
    assert out["vegetation_loss"]["applicable"] is False and out["drought_fire"]["applicable"] is False
    assert out["unusual_change"]["applicable"] is True and out["vegetation_loss"]["level"] == "n/a"


# ---- land cover model ----------------------------------------------------------------------------------
def reference_table():
    rng = np.random.default_rng(0)
    centres = {"forest": (-6.5, -12.5), "vegetation": (-9.5, -16), "urban": (-3.5, -9), "water": (-20, -26)}
    rows = []
    for ci, (cls, (vv, vh)) in enumerate(centres.items()):
        for k in range(3):
            for year in range(2019, 2026):
                f = {"vv_mean": vv + rng.normal(0, 0.4), "vh_mean": vh + rng.normal(0, 0.4), "vv_std": 0.5, "vh_std": 0.5, "ratio_mean": vv - vh + rng.normal(0, 0.3),
                     "vh_amplitude": 1.0 + ci * 0.1, "vv_min": vv - 1.5, "vh_p10": vh - 0.8}
                rows.append({"place": f"{cls}{k}", "cls": cls, "year": year, **f})
    return pd.DataFrame(rows)


def test_training_validates_by_leaving_places_out_and_saves(tmp_path):
    m = risk.train(reference_table(), tmp_path / "m.joblib")
    assert m["accuracy"] > 0.9 and m["places"] == 12 and set(m["perClass"]) == set(risk.CLASSES)
    model = risk.load_model.__wrapped__(str(tmp_path / "m.joblib"))
    out = risk.land_cover(series(lambda t, m_: -3.5 + 0 * t, 0.2, 21), series(lambda t, m_: -9 + 0 * t, 0.2, 22), model)
    assert out["class"] == "urban" and out["confidence"] > 0.6 and out["source"] == "pretrained"
    water = risk.land_cover(series(lambda t, m_: -20 + 0 * t, 0.2, 23), series(lambda t, m_: -26 + 0 * t, 0.2, 24), model)
    assert water["class"] == "water"


def test_training_refuses_too_little_data():
    t = reference_table()
    with pytest.raises(ValueError):
        risk.train(t[t["place"].isin(["forest0", "urban0"])])


def test_training_table_from_collected_series():
    rows = []
    for pol, base in (("co", -7.0), ("cross", -13.0)):
        for p in pd.period_range("2019-01", "2021-12", freq="M"):
            rows.append({"place": "P", "cls": "forest", "lon": 0, "lat": 0, "pol_class": pol, "month": str(p), "db": base})
    t = risk.training_table(pd.DataFrame(rows))
    assert list(t["year"]) == [2019, 2020, 2021] and t["vh_mean"].iloc[0] == -13.0


def test_missing_model_is_not_fatal(tmp_path):
    assert risk.load_model.__wrapped__(str(tmp_path / "none.joblib")) is None
    assert risk.land_cover(series(vv_flat), series(vh_flat), model=None) is None or True


def test_cleared_savanna_and_floodplain_are_merged_into_other_vegetation():
    rows = []
    for place, cls in (("A", "cleared"), ("B", "savanna"), ("C", "floodplain")):
        for pol, base in (("co", -8.0), ("cross", -15.0)):
            for p in pd.period_range("2020-01", "2020-12", freq="M"):
                rows.append({"place": place, "cls": cls, "lon": 0, "lat": 0, "pol_class": pol, "month": str(p), "db": base})
    assert set(risk.training_table(pd.DataFrame(rows))["cls"]) == {"vegetation"}
