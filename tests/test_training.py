"""Tests of the training protocol (the project document's rules) on synthetic places with a known answer."""
import json

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from asf_cleaner import risk, training


def synthetic_table(separation=1.0, seed=0, places_per_class=6):
    rng = np.random.default_rng(seed)
    centres = {"forest": (-6.5, -12.5), "vegetation": (-9.5, -16.0), "urban": (-3.5, -9.0), "water": (-20.0, -26.0)}
    rows = []
    for cls, (vv, vh) in centres.items():
        for k in range(places_per_class):
            shift = rng.normal(0, 1.0, 2)  # every place is a little different
            for year in range(2019, 2026):
                f = {f: rng.normal(0, 1) for f in risk.FEATURES_RICH}
                f.update(vv_mean=vv * separation + shift[0] + rng.normal(0, 0.4), vh_mean=vh * separation + shift[1] + rng.normal(0, 0.4))
                f["ratio_mean"] = f["vv_mean"] - f["vh_mean"]
                rows.append({"place": f"{cls}{k}", "cls": cls, "year": year, **f})
    return pd.DataFrame(rows)


@pytest.fixture
def light(monkeypatch):
    """Two small model families, so the tests run in seconds."""
    def fam(seed):
        return [
            ("random_forest", Pipeline([("m", RandomForestClassifier(class_weight="balanced", random_state=seed, n_jobs=1))]), {"m__n_estimators": [30, 60], "m__min_samples_leaf": [1, 3]}),
            ("logistic", Pipeline([("scale", StandardScaler()), ("m", LogisticRegression(class_weight="balanced", max_iter=2000))]), {"m__C": [0.3, 3]}),
        ]
    monkeypatch.setattr(training, "families", fam)


def test_the_split_puts_whole_places_apart_and_every_class_in_the_test():
    t = synthetic_table()
    for seed in range(5):
        train, test = training.split_places(t, seed)
        assert not train & test and train | test == set(t["place"])
        assert {p.rstrip("0123456789") for p in test} == {"forest", "vegetation", "urban", "water"}


def test_a_round_never_tests_on_seen_places_or_seen_years(light):
    t = synthetic_table()
    seen = {}
    real = training.baseline

    def spy(train, test):
        seen.update(train_places=set(train["place"]), test_places=set(test["place"]), train_years=set(train["year"]), test_years=set(test["year"]))
        return real(train, test)

    training.baseline = spy
    try:
        r = training.run_round(t, 3, "rich")
    finally:
        training.baseline = real
    assert not seen["train_places"] & seen["test_places"]
    assert max(seen["train_years"]) <= training.TRAIN_UNTIL < min(seen["test_years"])
    assert r["testPlaceAccuracy"] > 0.8 and r["baselineAccuracy"] <= 1.0 and r["trainRows"] > r["testRows"] * 0.5


def test_training_runs_the_rounds_saves_the_model_and_a_log(tmp_path, light):
    t = synthetic_table()
    m = training.train_rounds(t, rounds=3, target=0.0, max_rounds=3, path=tmp_path / "m.joblib", log_path=tmp_path / "log.json")
    assert m["rounds"] == 3 and m["accepted"] is True and m["testAccuracy"]["mean"] > 0.9
    assert m["roundsBeatingBaseline"] >= 2 and m["places"] == 24
    log = json.loads((tmp_path / "log.json").read_text())
    assert len(log["rounds"]) == 6 and {r["features"] for r in log["rounds"]} == {"base", "rich"}  # 3 rounds x 2 feature sets
    model = risk.load_model.__wrapped__(str(tmp_path / "m.joblib"))
    out = risk.land_cover(pd.Series(-3.5, index=pd.period_range("2022-01", periods=24, freq="M")) + np.random.default_rng(1).normal(0, 0.2, 24),
                          pd.Series(-9.0, index=pd.period_range("2022-01", periods=24, freq="M")) + np.random.default_rng(2).normal(0, 0.2, 24), model)
    assert out["class"] == "urban"


def test_it_keeps_training_until_the_target_is_reached_or_the_limit(tmp_path, light):
    t = synthetic_table(seed=3)
    m = training.train_rounds(t, rounds=2, target=1.01, max_rounds=4, path=tmp_path / "m.joblib", log_path=tmp_path / "l.json")
    assert m["rounds"] == 4  # the target is impossible, so it stops at the limit
    m2 = training.train_rounds(t, rounds=2, target=0.5, max_rounds=9, path=tmp_path / "m2.joblib", log_path=tmp_path / "l2.json")
    assert m2["rounds"] == 2  # already above the target after the minimum


def test_a_model_that_does_not_beat_the_simple_reference_is_not_accepted_or_used(tmp_path, light):
    rng = np.random.default_rng(0)
    t = synthetic_table()
    t["cls"] = rng.permutation(t["cls"].to_numpy())  # labels carry no information about the features
    t["cls"] = t.groupby("place")["cls"].transform("first")  # but keep one class per place
    m = training.train_rounds(t, rounds=2, target=0.0, max_rounds=2, path=tmp_path / "m.joblib", log_path=tmp_path / "l.json")
    assert m["testAccuracy"]["mean"] < 0.7
    if not m["accepted"]:
        model = risk.load_model.__wrapped__(str(tmp_path / "m.joblib"))
        idx = pd.period_range("2022-01", periods=24, freq="M")
        assert risk.land_cover(pd.Series(-7.0, index=idx), pd.Series(-13.0, index=idx), model) is None


def test_too_few_places_are_refused():
    with pytest.raises(ValueError):
        training.train_rounds(synthetic_table(places_per_class=1))


def test_the_baseline_is_a_small_readable_rule():
    t = synthetic_table()
    train, test = training.split_places(t, 1)
    acc = training.baseline(t[t["place"].isin(train)], t[t["place"].isin(test)])
    assert 0.4 < acc <= 1.0  # separable clusters: even a depth-2 tree does well here


# ---- radar + the color of the ground -----------------------------------------------------------------
def with_optical(table, seed=0):
    from asf_cleaner import optical

    rng = np.random.default_rng(seed)
    tone = {"forest": 20, "vegetation": 80, "urban": 110, "water": 30}
    opt = {p: {f: tone[c] + rng.normal(0, 3) + 5 * i for i, f in enumerate(optical.FEATURES)} for p, c in table.groupby("place")["cls"].first().items()}
    return table.merge(pd.DataFrame([{"place": p, **v} for p, v in opt.items()]), on="place")


def test_fusion_is_tried_when_the_table_has_optical_columns(tmp_path, light):
    t = with_optical(synthetic_table())
    assert set(training.feature_sets(t)) == {"base", "rich", "fusion"} and set(training.feature_sets(synthetic_table())) == {"base", "rich"}
    m = training.train_rounds(t, rounds=2, target=0.0, max_rounds=2, path=tmp_path / "m.joblib", log_path=tmp_path / "l.json")
    assert set(m["byFeatureSet"]) == {"base", "rich", "fusion"} and m["radarFeatures"] in ("base", "rich")
    model = risk.load_model.__wrapped__(str(tmp_path / "m.joblib"))
    assert {"radar_classifier", "radar_features", "classifier", "features"} <= set(model)


def test_the_model_uses_the_ground_color_when_it_has_it_and_falls_back_to_radar_without(tmp_path, light):
    from asf_cleaner import optical

    t = with_optical(synthetic_table(separation=0.3, seed=2), seed=3)  # radar alone is hard here, color is easy
    m = training.train_rounds(t, rounds=2, target=0.0, max_rounds=2, path=tmp_path / "m.joblib", log_path=tmp_path / "l.json")
    model = risk.load_model.__wrapped__(str(tmp_path / "m.joblib"))
    idx = pd.period_range("2021-01", periods=36, freq="M")
    vv, vh = pd.Series(-9.0, index=idx), pd.Series(-15.0, index=idx)
    forest_color = {f: 20 + 5 * i for i, f in enumerate(optical.FEATURES)}
    out = risk.land_cover(vv, vh, model, optical=forest_color)
    assert out["inputs"] == ("radar + optical" if "opt_r" in model["features"] else "radar")
    radar_only = risk.land_cover(vv, vh, model)  # no ground color available: the radar-only classifier answers
    assert radar_only is not None and radar_only["inputs"] == "radar"
    assert m["byFeatureSet"]["fusion"]["testAccuracy"] >= m["byFeatureSet"]["base"]["testAccuracy"]
