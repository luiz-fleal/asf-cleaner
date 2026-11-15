"""Monthly forecast of radar backscatter and rain up to 2030 (scikit-learn). No network here.

What it does, in plain terms
- Backscatter (dB) and rain (mm/month) are turned into one value per month.
- Several small models compete: a yearly cycle (two harmonics), the same plus a straight-line trend, and
  for backscatter also the monthly rain anomaly (with and without a gradient-boosting correction).
- The models are compared by a rolling-origin backtest: train on the past, forecast the next 12 months,
  move the origin forward, repeat. The simplest model within 5 % of the lowest error wins, and its error against the plain
  "same month last year" guess is reported as `skill` (above 0 means it beats that guess).
- The winner is refitted on all the data and projected to the end date. The 80 % band comes from the errors
  the winner made in the backtest and is widened with the distance into the future, because the backtest
  only looks 12 months ahead.

Limits that the result states instead of hiding: the future rain is unknown, so the backscatter forecast
assumes normal rain unless a `rain_scale` scenario is given; a short history gives wide bands; and nothing
here knows about land-cover change, floods or fires that have not happened yet.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Ridge

MIN_MONTHS = 36  # training months a fold needs
CV_HORIZON = 12  # months forecast in every backtest fold
CV_FOLDS = 3
HARMONICS = 2
RIDGE_ALPHA = 0.5
INTERVAL = 0.8
PARSIMONY = 1.05  # a simpler model wins unless a more complex one is more than 5 % better
WIDEN_PER_YEAR = 0.2  # band growth per year beyond the first, see module docstring
MIN_FOR_RELIABLE = 48
MIN_COVERAGE = 0.5  # share of months that must have data between the first and last observation


class InsufficientHistory(ValueError):
    """Not enough monthly history to train and validate a model."""


@dataclass(frozen=True)
class Spec:
    name: str
    trend: bool = False
    rain: bool = False
    boost: bool = False


BACKSCATTER_SPECS = (
    Spec("seasonal"),
    Spec("seasonal+trend", trend=True),
    Spec("seasonal+trend+rain", trend=True, rain=True),
    Spec("seasonal+trend+rain+boosting", trend=True, rain=True, boost=True),
)
RAIN_SPECS = (Spec("seasonal"), Spec("seasonal+trend", trend=True))


# ---- features ---------------------------------------------------------------------------------
def _ordinals(index: pd.PeriodIndex) -> np.ndarray:
    return np.array([p.ordinal for p in index], dtype=float)


def rain_anomaly(rain: pd.Series | None, upto: pd.Period) -> pd.Series | None:
    """Rain of each month as a z-score against the calendar-month mean of the months up to `upto`."""
    if rain is None or rain.empty:
        return None
    base = rain[rain.index <= upto]
    if len(base) < 12:
        return None
    clim = base.groupby(base.index.month).mean()
    spread = float(base.std()) or 1.0
    z = (rain - rain.index.month.map(clim).to_numpy()) / spread
    return z.clip(-3, 3)


def _anomaly_columns(index: pd.PeriodIndex, anom: pd.Series | None) -> np.ndarray:
    """Anomaly of the month and of the month before it (0 where unknown)."""
    if anom is None:
        return np.zeros((len(index), 2))
    now = anom.reindex(index).fillna(0.0).to_numpy()
    prev = anom.reindex(index - 1).fillna(0.0).to_numpy()
    return np.column_stack([now, prev])


def _design(index: pd.PeriodIndex, spec: Spec, ref: float, anom: pd.Series | None) -> np.ndarray:
    m = index.month.to_numpy()
    cols = []
    for h in range(1, HARMONICS + 1):
        cols += [np.sin(2 * np.pi * h * m / 12), np.cos(2 * np.pi * h * m / 12)]
    if spec.trend:
        cols.append((_ordinals(index) - ref) / 120.0)  # per decade
    X = np.column_stack(cols)
    if spec.rain:
        X = np.column_stack([X, _anomaly_columns(index, anom)])
    return X


def _boost_features(index: pd.PeriodIndex, anom: pd.Series | None) -> np.ndarray:
    m = index.month.to_numpy()
    return np.column_stack([np.sin(2 * np.pi * m / 12), np.cos(2 * np.pi * m / 12), _anomaly_columns(index, anom)])


def _fit_predict(y: pd.Series, spec: Spec, anom: pd.Series | None, future: pd.PeriodIndex,
                 future_anom: pd.Series | None) -> tuple[np.ndarray, float | None]:
    """Fit on `y` and predict `future`. Returns (prediction, trend per decade or None)."""
    ref = float(_ordinals(y.index).mean())
    X = _design(y.index, spec, ref, anom)
    ridge = Ridge(alpha=RIDGE_ALPHA).fit(X, y.to_numpy())
    pred = ridge.predict(_design(future, spec, ref, future_anom))
    if spec.boost:
        resid = y.to_numpy() - ridge.predict(X)
        gb = GradientBoostingRegressor(n_estimators=120, max_depth=2, learning_rate=0.05, subsample=0.8, random_state=0)
        gb.fit(_boost_features(y.index, anom), resid)
        pred = pred + gb.predict(_boost_features(future, future_anom))
    n_seasonal = 2 * HARMONICS
    return pred, (float(ridge.coef_[n_seasonal]) if spec.trend else None)


# ---- backtest ---------------------------------------------------------------------------------
def _naive(train: pd.Series, test_index: pd.PeriodIndex) -> np.ndarray:
    """Same calendar month of the latest earlier year that has data (else the training mean)."""
    out = []
    for p in test_index:
        v = np.nan
        for k in range(1, 6):
            q = p - 12 * k
            if q in train.index:
                v = float(train[q])
                break
        out.append(train.mean() if np.isnan(v) else v)
    return np.array(out)


def _origins(y: pd.Series) -> list[pd.Period]:
    last = y.index.max()
    out = []
    for k in range(CV_FOLDS, 0, -1):
        o = last - CV_HORIZON * k
        if (y.index <= o).sum() >= MIN_MONTHS and ((y.index > o) & (y.index <= o + CV_HORIZON)).sum() >= 3:
            out.append(o)
    return out


def backtest(y: pd.Series, rain: pd.Series | None, spec: Spec, *, log_target: bool = False) -> pd.DataFrame:
    """Rolling-origin forecasts of the last years, never using what came after each origin.

    Returns one row per forecast month: horizon (months ahead), actual, pred, naive, in the original unit.
    """
    rows = []
    for origin in _origins(y):
        train = y[y.index <= origin]
        test = y[(y.index > origin) & (y.index <= origin + CV_HORIZON)]
        # the true rain of the test months is NOT given to the model: in the real forecast it is unknown
        anom = rain_anomaly(rain[rain.index <= origin] if rain is not None else None, origin)
        pred, _ = _fit_predict(train, spec, anom, test.index, None)
        naive = _naive(train, test.index)
        actual = test.to_numpy()
        if log_target:
            pred, naive, actual = np.expm1(pred), np.expm1(naive), np.expm1(actual)
        rows.append(pd.DataFrame({
            "horizon": [(p - origin).n for p in test.index], "actual": actual, "pred": pred, "naive": naive,
        }))
    if not rows:
        raise InsufficientHistory(f"need at least {MIN_MONTHS + 3} months of data to validate a model")
    return pd.concat(rows, ignore_index=True)


def _scores(bt: pd.DataFrame) -> dict:
    err = bt["pred"] - bt["actual"]
    naive_err = bt["naive"] - bt["actual"]
    mae, nmae = float(err.abs().mean()), float(naive_err.abs().mean())
    return {
        "mae": mae, "rmse": float(np.sqrt((err**2).mean())), "naiveMae": nmae,
        "skill": (1 - mae / nmae) if nmae > 0 else 0.0, "months": int(len(bt)),
    }


# ---- one series ---------------------------------------------------------------------------------
def _check(y: pd.Series, name: str) -> pd.Series:
    y = y.dropna().sort_index()
    y = y[~y.index.duplicated(keep="last")]
    if len(y) < MIN_MONTHS + 3:
        raise InsufficientHistory(f"{name}: {len(y)} months of data, need at least {MIN_MONTHS + 3}")
    span = (y.index.max() - y.index.min()).n + 1
    if len(y) / span < MIN_COVERAGE:
        raise InsufficientHistory(f"{name}: only {len(y)} of {span} months have data")
    return y


def forecast_series(y: pd.Series, specs: tuple[Spec, ...], until: pd.Period, *, rain: pd.Series | None = None,
                    rain_scale: float = 1.0, log_target: bool = False, name: str = "series") -> dict:
    """Choose the best spec by backtest, refit on everything and forecast to `until`.

    `y` is monthly (PeriodIndex). With `log_target` the model works on log1p(y) and results are in y's unit.
    """
    y = _check(y, name)
    target = np.log1p(y.clip(lower=0)) if log_target else y
    usable = tuple(s for s in specs if not s.rain or rain is not None)

    results = {}
    for spec in usable:
        bt = backtest(target, rain, spec, log_target=log_target)
        results[spec.name] = (spec, bt, _scores(bt))
    # specs are listed from simplest to most complex: take the first one within 5 % of the lowest error,
    # because three folds are too few to trust small differences and extra terms extrapolate worse
    lowest = min(r[2]["mae"] for r in results.values())
    best_name = next(k for k, r in results.items() if r[2]["mae"] <= lowest * PARSIMONY)
    spec, bt, score = results[best_name]

    last = y.index.max()
    future = pd.period_range(last + 1, until, freq="M")
    if len(future) == 0:
        raise ValueError(f"{name}: forecast end {until} is not after the last observation {last}")
    anom = rain_anomaly(rain, last)
    future_anom = None
    if spec.rain and anom is not None and rain_scale != 1.0:
        base = rain[rain.index <= last]
        clim = base.groupby(base.index.month).mean()
        spread = float(base.std()) or 1.0
        future_anom = pd.Series(((rain_scale - 1.0) * future.month.map(clim).to_numpy() / spread), index=future).clip(-3, 3)
    pred, trend = _fit_predict(target, spec, anom, future, future_anom)

    # 80 % band: errors of the winner in the backtest, widened with the years ahead
    if log_target:
        log_err = np.abs(np.log1p(bt["pred"].clip(lower=0)) - np.log1p(bt["actual"].clip(lower=0)))
        half = float(np.quantile(log_err, INTERVAL))
    else:
        half = float(np.quantile((bt["pred"] - bt["actual"]).abs(), INTERVAL))
    years_ahead = np.array([(p - last).n / 12 for p in future])
    width = half * np.sqrt(1 + WIDEN_PER_YEAR * np.maximum(0, years_ahead - 1))
    lo, hi = pred - width, pred + width
    if log_target:
        pred, lo, hi = np.expm1(pred) * rain_scale, np.expm1(lo).clip(min=0) * rain_scale, np.expm1(hi) * rain_scale

    def stamp(p: pd.Period) -> int:
        return int(p.start_time.tz_localize("UTC").timestamp() * 1000)

    return {
        "model": spec.name,
        "reliable": bool(score["skill"] > 0 and len(y) >= MIN_FOR_RELIABLE),
        "backtest": {**score, "folds": len(_origins(target)), "candidates": {k: v[2] for k, v in results.items()}},
        "trendPerDecade": trend,
        "bandWidth": INTERVAL,
        "history": [{"t": stamp(p), "v": round(float(v), 4)} for p, v in y.items()],
        "forecast": [{"t": stamp(p), "v": round(float(v), 4), "lo": round(float(a), 4), "hi": round(float(b), 4)}
                     for p, v, a, b in zip(future, pred, lo, hi)],
        "yearly": _yearly(future, pred),
    }


def _yearly(future: pd.PeriodIndex, pred: np.ndarray) -> list[dict]:
    df = pd.DataFrame({"year": future.year, "v": pred})
    g = df.groupby("year")["v"].agg(["mean", "sum", "count"])
    return [{"year": int(y), "mean": round(float(r["mean"]), 3), "total": round(float(r["sum"]), 1), "months": int(r["count"])}
            for y, r in g.iterrows()]


# ---- from scene rows to the API result ------------------------------------------------------------
def monthly_backscatter(rows: pd.DataFrame, min_valid: float = 0.8) -> dict[str, tuple[str, pd.Series]]:
    """{pol_class: (polarization, monthly median dB)} from cleaned scene rows (`harmonize` columns)."""
    d = rows[(rows["valid_fraction"] >= min_valid) & rows["backscatter_db_median"].notna()].copy()
    d["month"] = pd.to_datetime(d["start_time"], utc=True).dt.tz_localize(None).dt.to_period("M")
    out = {}
    for cls, grp in d.groupby("pol_class"):
        pol = grp["polarization"].value_counts().idxmax()  # one polarization per class
        s = grp[grp["polarization"] == pol].groupby("month")["backscatter_db_median"].median()
        out[str(cls)] = (str(pol), s)
    return out


def run_forecast(radar: pd.DataFrame, rain_monthly: pd.Series | None, until: str = "2030-12", rain_scale: float = 1.0) -> dict:
    """Backscatter (co and cross polarization) and rain forecasts from the collected history."""
    end = pd.Period(until, freq="M")
    result: dict = {"until": str(end), "rainScale": rain_scale, "backscatter": {}, "notes": []}
    problems = []
    for cls, (pol, series) in (monthly_backscatter(radar) if radar is not None and len(radar) else {}).items():
        try:
            out = forecast_series(series, BACKSCATTER_SPECS, end, rain=rain_monthly, rain_scale=rain_scale, name=f"backscatter {pol}")
            out["pol"] = pol
            result["backscatter"][cls] = out
        except InsufficientHistory as e:
            problems.append(str(e))
    if rain_monthly is not None and len(rain_monthly):
        try:
            result["rain"] = forecast_series(rain_monthly, RAIN_SPECS, end, rain_scale=rain_scale, log_target=True, name="rain")
        except InsufficientHistory as e:
            problems.append(str(e))
    if not result["backscatter"] and "rain" not in result:
        raise InsufficientHistory("; ".join(problems) or "no history to learn from")
    result["notes"] += problems
    result["notes"].append("Future rain is unknown: the backscatter forecast assumes normal rain" + (
        f" scaled by {rain_scale:g}." if rain_scale != 1 else "."))
    result["notes"].append("The 80 % band comes from backtest errors and widens with the years ahead; it does not include events not yet seen.")
    return result
