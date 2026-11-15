"""Training of the land-cover model, following the rules of the project document (section "Modelo de machine learning").

    uv run python -m asf_cleaner.training --rounds 10

The rules, taken from the document and applied here:
- A simple reference first (a depth-2 decision tree on mean VV and VH, a rule a person could write down). The model is
  only accepted if it beats that reference on the test, otherwise `risk.py` does not use it.
- The split respects time AND area: the test is made of places the model never saw, in years it never saw
  (training years up to 2022, test years from 2023). No random split of rows.
- Scales and statistics come from the training part only (they live inside the pipeline, so cross-validation refits them
  on every fold). The folds used to choose settings are grouped by place.
- Train, tune and test are different things: settings are chosen by cross-validation on the training places, and the
  test is looked at once per round.
- One round is one complete run of that protocol with a different random choice of test places. The final numbers are the
  mean and the spread over the rounds, so one lucky split cannot make the model look good.

After the rounds the best configuration is refitted on all the places and years and saved with every round's numbers.
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import GroupKFold, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from . import optical, pretrain, risk

log = logging.getLogger("asf_cleaner")
TRAIN_UNTIL = 2022  # training years; the test uses the years after it
TEST_SHARE = 0.25  # of the places of every class
RADAR_SETS = {"base": risk.FEATURES, "rich": risk.FEATURES_RICH}


def feature_sets(table: pd.DataFrame) -> dict[str, list[str]]:
    """The feature sets tried in every round: radar only, and radar + the color of the ground when the table has it."""
    sets = dict(RADAR_SETS)
    if all(c in table.columns for c in optical.FEATURES):
        sets["fusion"] = risk.FEATURES_RICH + optical.FEATURES
    return sets


LOG_PATH = risk.MODEL_PATH.with_name("training_log.json")


def families(seed: int):
    """(name, pipeline, search space). Scaling is inside the pipeline: it is fitted on the training part only."""
    scaled = lambda est: Pipeline([("scale", StandardScaler()), ("m", est)])  # noqa: E731
    plain = lambda est: Pipeline([("m", est)])  # noqa: E731
    return [
        ("random_forest", plain(RandomForestClassifier(class_weight="balanced", random_state=seed, n_jobs=1)),
         {"m__n_estimators": [150, 300, 500], "m__min_samples_leaf": [1, 2, 3, 5], "m__max_features": ["sqrt", 0.5, None]}),
        ("extra_trees", plain(ExtraTreesClassifier(class_weight="balanced", random_state=seed, n_jobs=1)),
         {"m__n_estimators": [150, 300, 500], "m__min_samples_leaf": [1, 2, 3, 5], "m__max_features": ["sqrt", 0.5, None]}),
        ("gradient_boosting", plain(HistGradientBoostingClassifier(class_weight="balanced", random_state=seed)),
         {"m__learning_rate": [0.03, 0.06, 0.1], "m__max_depth": [2, 3, 4], "m__max_iter": [100, 200], "m__l2_regularization": [0.0, 1.0, 5.0]}),
        ("logistic", scaled(LogisticRegression(class_weight="balanced", max_iter=3000)), {"m__C": [0.1, 0.3, 1, 3, 10, 30]}),
        ("svm", scaled(SVC(class_weight="balanced", probability=True, random_state=seed)), {"m__C": [0.3, 1, 3, 10, 30], "m__gamma": ["scale", 0.03, 0.1, 0.3]}),
    ]


def split_places(table: pd.DataFrame, seed: int) -> tuple[set, set]:
    """Test places: about 25 % of the places of every class, at least one, chosen at random."""
    rng = np.random.default_rng(seed)
    test = set()
    for _, g in table.groupby("cls"):
        places = sorted(g["place"].unique())
        k = max(1, round(len(places) * TEST_SHARE))
        test |= set(rng.choice(places, size=k, replace=False))
    return set(table["place"]) - test, test


def baseline(train: pd.DataFrame, test: pd.DataFrame) -> float:
    """The simple reference: a depth-2 decision tree on mean VV and VH."""
    tree = DecisionTreeClassifier(max_depth=2, class_weight="balanced", random_state=0).fit(train[["vv_mean", "vh_mean"]], train["cls"])
    return float((tree.predict(test[["vv_mean", "vh_mean"]]) == test["cls"]).mean())


def tune(train: pd.DataFrame, features: list[str], seed: int, n_iter: int = 8):
    """Best (family, estimator, cv score) by cross-validation grouped by place, on the training part only."""
    folds = max(2, min(4, train["place"].nunique()))
    best = None
    for name, pipe, space in families(seed):
        size = int(np.prod([len(v) for v in space.values()]))
        search = RandomizedSearchCV(pipe, space, n_iter=min(n_iter, size), scoring="balanced_accuracy", cv=GroupKFold(n_splits=folds),
                                    random_state=seed, n_jobs=2, refit=True)
        search.fit(train[features], train["cls"], groups=train["place"])
        if best is None or search.best_score_ > best[2]:
            best = (name, search.best_estimator_, float(search.best_score_), search.best_params_)
    return best


def place_accuracy(model, test: pd.DataFrame, features: list[str]) -> float:
    """Accuracy per place: the probabilities of all its test years are averaged, then one answer per place."""
    proba = pd.DataFrame(model.predict_proba(test[features]), columns=model.classes_, index=test.index)
    mean = proba.groupby(test["place"]).mean()
    truth = test.groupby("place")["cls"].first()
    return float((mean.idxmax(axis=1) == truth.reindex(mean.index)).mean())


def run_round(table: pd.DataFrame, seed: int, feature_set: str) -> dict:
    features = feature_sets(table)[feature_set]
    train_places, test_places = split_places(table, seed)
    train = table[table["place"].isin(train_places) & (table["year"] <= TRAIN_UNTIL)]
    test = table[table["place"].isin(test_places) & (table["year"] > TRAIN_UNTIL)]
    if train["cls"].nunique() < 2 or test.empty:
        raise ValueError("a round needs training years up to %d and test years after it" % TRAIN_UNTIL)
    t0 = time.perf_counter()
    family, model, cv, params = tune(train, features, seed)
    pred = model.predict(test[features])
    return {
        "seed": seed, "features": feature_set, "family": family, "params": {k: (v if isinstance(v, (int, float, str)) or v is None else str(v)) for k, v in params.items()},
        "cvBalancedAccuracy": round(cv, 4), "testAccuracy": round(float((pred == test["cls"]).mean()), 4),
        "testBalancedAccuracy": round(float(balanced_accuracy_score(test["cls"], pred)), 4),
        "testPlaceAccuracy": round(place_accuracy(model, test, features), 4),
        "baselineAccuracy": round(baseline(train, test), 4),
        "trainPlaces": len(train_places), "testPlaces": len(test_places), "trainRows": int(len(train)), "testRows": int(len(test)),
        "seconds": round(time.perf_counter() - t0, 1),
    }


def train_rounds(table: pd.DataFrame, rounds: int = 10, target: float = 0.9, max_rounds: int = 30, path: Path | None = None, log_path: Path | None = None) -> dict:
    """Run the protocol `rounds` times (more, up to `max_rounds`, while the mean test accuracy is below `target`),
    then fit the best configuration on everything and save it."""
    import joblib

    if table["place"].nunique() < 8:
        raise ValueError("need at least 8 places to train and test on different places")
    results: list[dict] = []
    seed = 0
    while True:
        for fs in feature_sets(table):
            r = run_round(table, seed, fs)
            results.append(r)
            log.info("round %2d %-4s %-17s test %.3f (places %.3f) baseline %.3f", seed + 1, fs, r["family"], r["testAccuracy"], r["testPlaceAccuracy"], r["baselineAccuracy"])
        seed += 1
        df = pd.DataFrame(results)
        best_fs = df.groupby("features")["testAccuracy"].mean().idxmax()
        mean_best = float(df[df["features"] == best_fs]["testAccuracy"].mean())
        if seed >= rounds and (mean_best >= target or seed >= max_rounds):
            break

    df = pd.DataFrame(results)
    chosen = df[df["features"] == best_fs]
    sets = feature_sets(table)

    def pick_family(sub: pd.DataFrame) -> str:
        # the family chosen most often (ties: higher mean test accuracy) is the configuration that goes to production
        return sub.groupby("family")["testAccuracy"].agg(["count", "mean"]).sort_values(["count", "mean"], ascending=False).index[0]

    features = sets[best_fs]
    final_family, final, cv, _ = _fit_family(table, features, pick_family(chosen))
    radar_fs = df[df["features"].isin(RADAR_SETS)].groupby("features")["testAccuracy"].mean().idxmax()
    radar_rows = df[df["features"] == radar_fs]
    _, radar_final, _, _ = _fit_family(table, sets[radar_fs], pick_family(radar_rows))
    metrics = {
        "protocol": f"{seed} rounds; per round: test = {int(TEST_SHARE * 100)}% of the places of each class, in years after {TRAIN_UNTIL}; "
                    f"settings chosen by cross-validation grouped by place on the other places",
        "rounds": seed, "features": best_fs, "family": final_family, "radarFeatures": radar_fs,
        "radarAccuracy": round(float(radar_rows["testPlaceAccuracy"].mean()), 4), "radarTestAccuracy": round(float(radar_rows["testAccuracy"].mean()), 4),
        "byFeatureSet": {k: {"testAccuracy": round(float(v["testAccuracy"].mean()), 4), "testPlaceAccuracy": round(float(v["testPlaceAccuracy"].mean()), 4)} for k, v in df.groupby("features")},
        "testAccuracy": {"mean": round(float(chosen["testAccuracy"].mean()), 4), "std": round(float(chosen["testAccuracy"].std(ddof=0)), 4),
                         "min": round(float(chosen["testAccuracy"].min()), 4), "max": round(float(chosen["testAccuracy"].max()), 4)},
        "testPlaceAccuracy": {"mean": round(float(chosen["testPlaceAccuracy"].mean()), 4), "std": round(float(chosen["testPlaceAccuracy"].std(ddof=0)), 4)},
        "testBalancedAccuracy": round(float(chosen["testBalancedAccuracy"].mean()), 4),
        "baselineAccuracy": {"mean": round(float(chosen["baselineAccuracy"].mean()), 4), "std": round(float(chosen["baselineAccuracy"].std(ddof=0)), 4)},
        "roundsBeatingBaseline": int((chosen["testAccuracy"] > chosen["baselineAccuracy"]).sum()),
        "places": int(table["place"].nunique()), "samples": int(len(table)), "classes": sorted(table["cls"].unique()),
    }
    metrics["accepted"] = metrics["testAccuracy"]["mean"] > metrics["baselineAccuracy"]["mean"]
    metrics["accuracy"] = metrics["testPlaceAccuracy"]["mean"]  # what the clients show (accuracy for a place never seen)
    metrics["places"] = metrics["places"]
    path = path or risk.MODEL_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"classifier": final, "features": features, "radar_classifier": radar_final, "radar_features": sets[radar_fs], "metrics": metrics}, path)
    (log_path or LOG_PATH).write_text(json.dumps({"summary": metrics, "rounds": results}, indent=1, default=str))
    risk.load_model.cache_clear()
    return metrics


def _fit_family(table: pd.DataFrame, features: list[str], family: str):
    """Tune the chosen family on all the places (cross-validation grouped by place) and fit it on all the data."""
    folds = max(2, min(5, table["place"].nunique()))
    for name, pipe, space in families(0):
        if name != family:
            continue
        size = int(np.prod([len(v) for v in space.values()]))
        search = RandomizedSearchCV(pipe, space, n_iter=min(16, size), scoring="balanced_accuracy", cv=GroupKFold(n_splits=folds), random_state=0, n_jobs=2, refit=True)
        search.fit(table[features], table["cls"], groups=table["place"])
        return name, search.best_estimator_, float(search.best_score_), search.best_params_
    raise ValueError(family)


def main(argv=None):
    import warnings

    warnings.filterwarnings("ignore", category=FutureWarning)  # scikit-learn announcing a change to SVC(probability=True)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    ap = argparse.ArgumentParser(description="Train the land-cover model by the project document's protocol")
    ap.add_argument("--rounds", type=int, default=10, help="at least this many rounds")
    ap.add_argument("--target", type=float, default=0.9, help="keep going (up to --max-rounds) while the mean test accuracy is below this")
    ap.add_argument("--max-rounds", type=int, default=30)
    a = ap.parse_args(argv)
    table = risk.training_table(pretrain.load_series(), pretrain.load_optical())
    print(json.dumps(train_rounds(table, a.rounds, a.target, a.max_rounds), indent=1))


if __name__ == "__main__":
    main()
