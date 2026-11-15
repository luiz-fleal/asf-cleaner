"""Risk assessment of one place from its radar and rain history. No network here.

What is learned, and from what
- Land cover (forest, other vegetation, city, open water) is recognised by a RandomForest that was
  pre-trained on reference places with known cover (`pretrain.py`). It decides which risks make sense at a place
  (vegetation loss is meaningless over a city, drought over a lake) and how the place is described.
- Everything else is learned from the place's own history, so it is never compared with another place's numbers:
  * vegetation loss: the cross-polarization (VH) is sensitive to the volume of vegetation. After removing the
    yearly cycle, the single most likely step change is located (largest t-statistic between two segments);
    a persistent drop of several dB is the radar signature of clearing or degradation.
  * flood: how often the place already showed inundation-like backscatter, how wet the last months were against
    its own climate, and how high the forecast wet season is against its own past peaks.
  * drought and fire: a standardised 3-month rain index (like the SPI) at the last month and the forecast
    rain of the next six months against the same months of the past.
  * unusual change: an IsolationForest over the monthly [VV, VH, VV-VH, rain, 3-month rain] vectors flags months
    that do not look like the rest of the history; the share of unusual months in the last year is the signal.
- Each risk has a score from 0 to 1, a level (low below 0.3, moderate below 0.6, high from 0.6) and the evidence
  behind it, written as templates with numbers so a client can translate them.

These are screening signals, not predictions of events: a high level means "look here first".
"""
from __future__ import annotations

import functools
import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd

from . import forecast as fc

log = logging.getLogger("asf_cleaner")

MODEL_PATH = Path(os.environ.get("ASF_MODELS_DIR", Path(__file__).resolve().parents[2] / "models")) / "risk_model.joblib"
CLASSES = ("forest", "vegetation", "urban", "water")
# C-band radar alone cannot tell cleared land, savanna and floodplain from each other (the pre-training backtest,
# leaving whole places out, recognised them 0 to 60 % of the time), so they are one class: other vegetation.
MERGE = {"cleared": "vegetation", "savanna": "vegetation", "floodplain": "vegetation", "open": "vegetation"}
MODERATE, HIGH = 0.3, 0.6
MIN_MONTHS = 30
SEGMENT = 6  # months a segment needs for the change search
INUNDATION_DB = -15.0  # co-polarization below this is almost always open water
# which risks make sense over which cover
APPLICABLE = {
    "vegetation_loss": {"forest", "vegetation"},
    "flood": {"forest", "vegetation", "urban"},
    "drought_fire": {"forest", "vegetation"},
    "unusual_change": set(CLASSES),
}


def level(score: float) -> str:
    return "high" if score >= HIGH else "moderate" if score >= MODERATE else "low"


def _ev(text: str, **vars) -> dict:
    return {"text": text, "vars": {k: (round(v, 2) if isinstance(v, float) else v) for k, v in vars.items()}}


# ---- land cover ------------------------------------------------------------------------------------
FEATURES = ["vv_mean", "vh_mean", "vv_std", "vh_std", "ratio_mean", "vh_amplitude", "vv_min", "vh_p10"]
FEATURES_RICH = FEATURES + ["vv_p90", "vh_p90", "vv_amplitude", "ratio_std", "vv_vh_corr", "vh_iqr", "vv_iqr", "vh_max_minus_min"]


def cover_features(vv: pd.Series, vh: pd.Series) -> dict | None:
    """Features of one year (or the whole history) of monthly VV and VH (dB)."""
    both = pd.concat([vv.rename("vv"), vh.rename("vh")], axis=1).dropna()
    if len(both) < 8:
        return None
    clim = both["vh"].groupby(both.index.month).mean()
    clim_vv = both["vv"].groupby(both.index.month).mean()
    ratio = both["vv"] - both["vh"]
    q = lambda s, p: float(np.quantile(s, p))  # noqa: E731
    return {
        "vv_mean": both["vv"].mean(), "vh_mean": both["vh"].mean(), "vv_std": both["vv"].std(), "vh_std": both["vh"].std(),
        "ratio_mean": ratio.mean(), "vh_amplitude": float(clim.max() - clim.min()),
        "vv_min": both["vv"].min(), "vh_p10": q(both["vh"], 0.1),
        "vv_p90": q(both["vv"], 0.9), "vh_p90": q(both["vh"], 0.9), "vv_amplitude": float(clim_vv.max() - clim_vv.min()),
        "ratio_std": ratio.std(), "vv_vh_corr": float(both["vv"].corr(both["vh"])) if both["vv"].std() > 0 and both["vh"].std() > 0 else 0.0,
        "vh_iqr": q(both["vh"], 0.75) - q(both["vh"], 0.25), "vv_iqr": q(both["vv"], 0.75) - q(both["vv"], 0.25),
        "vh_max_minus_min": float(both["vh"].max() - both["vh"].min()),
    }


@functools.lru_cache(maxsize=1)
def load_model(path: str | None = None):
    """The pre-trained land-cover model, or None if it has not been trained."""
    import joblib

    p = Path(path) if path else MODEL_PATH
    if not p.exists():
        return None
    try:
        return joblib.load(p)
    except Exception as e:  # noqa: BLE001 - a broken file must not break the forecast
        log.warning("ignoring unreadable risk model %s (%s)", p, e)
        return None


def land_cover(vv: pd.Series | None, vh: pd.Series | None, model=None, optical: dict | None = None) -> dict | None:
    model = load_model() if model is None else model  # model=False means: no model, do not load one
    if not model or vv is None or vh is None:
        return None
    if not model["metrics"].get("accepted", True):  # the document's rule: a model that does not beat the simple reference is not used
        return None
    feats = cover_features(vv, vh)
    if feats is None:
        return None
    # the model that also sees the color of the ground when the place has it; else the radar-only one
    uses_optical = any(c.startswith("opt_") for c in model["features"])
    if uses_optical and optical is None:
        if "radar_classifier" not in model:
            return None
        clf, cols = model["radar_classifier"], model["radar_features"]
    else:
        clf, cols = model["classifier"], model["features"]
    X = pd.DataFrame([{**feats, **(optical or {})}])[cols]
    proba = clf.predict_proba(X)[0]
    classes = list(clf.classes_)
    best = int(np.argmax(proba))
    return {"class": classes[best], "confidence": round(float(proba[best]), 3),
            "probabilities": {c: round(float(p), 3) for c, p in zip(classes, proba)}, "source": "pretrained", "inputs": "radar + optical" if any(c.startswith("opt_") for c in cols) else "radar",
            "trainedOn": model["metrics"]["places"],
            "cvAccuracy": model["metrics"]["accuracy"] if cols == model["features"] else model["metrics"].get("radarAccuracy")}


# ---- helpers ---------------------------------------------------------------------------------------
def _stamp(p: pd.Period) -> int:
    return int(p.start_time.tz_localize("UTC").timestamp() * 1000)


def _residual(y: pd.Series) -> pd.Series:
    """The series without its yearly cycle."""
    fitted, _ = fc._fit_predict(y, fc.Spec("seasonal"), None, y.index, None)
    return y - fitted


def _step(res: pd.Series) -> dict | None:
    """The most likely single step change of a series: largest Welch t between a before and an after segment."""
    v = res.to_numpy()
    n = len(v)
    if n < 2 * SEGMENT + 2:
        return None
    best = None
    for k in range(SEGMENT, n - SEGMENT + 1):
        a, b = v[:k], v[k:]
        se = np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
        if se == 0:
            continue
        t = (b.mean() - a.mean()) / se
        if best is None or abs(t) > abs(best["t"]):
            best = {"t": float(t), "k": k, "delta": float(b.mean() - a.mean())}
    if best is None:
        return None
    best["at"] = res.index[best["k"]]
    best["months_after"] = n - best["k"]
    return best


def _spi(rain: pd.Series, window: int = 3) -> tuple[float, pd.Series] | None:
    """Standardised rain of the last `window` months against the same calendar window of the past."""
    full = rain.reindex(pd.period_range(rain.index.min(), rain.index.max(), freq="M"))
    sums = full.rolling(window, min_periods=window).sum().dropna()
    if len(sums) < 24:
        return None
    last = sums.index.max()
    same = sums[sums.index.month == last.month]
    base = same[same.index < last]
    if len(base) < 5 or base.std() == 0:
        return None
    return float((sums[last] - base.mean()) / base.std()), sums


# ---- the four risks ---------------------------------------------------------------------------------
def vegetation_loss(vh: pd.Series, forecast_vh: dict | None) -> dict:
    res = _residual(vh)
    step = _step(res)
    if step is None:
        return {"score": 0.0, "evidence": [_ev("Not enough history to look for a change.")]}
    age = (vh.index.max() - step["at"]).n
    drop = max(0.0, -step["delta"])
    strength = min(1.0, drop / 2.5) * min(1.0, abs(step["t"]) / 5.0)
    if age > 36:
        strength *= 0.4  # an old step: it already happened
    if step["months_after"] < SEGMENT:
        strength *= 0.5
    ev = []
    if drop > 0.3:
        ev.append(_ev("Cross-polarization backscatter fell {db} dB around {date} and stayed lower.", db=drop, date=str(step["at"])))
    elif step["delta"] > 0.3:
        ev.append(_ev("Cross-polarization backscatter rose {db} dB around {date}.", db=step["delta"], date=str(step["at"])))
    else:
        ev.append(_ev("No lasting change in cross-polarization backscatter."))
    trend = (forecast_vh or {}).get("trendPerDecade")
    if trend is not None and trend < -0.5:
        strength = min(1.0, strength + 0.1)
        ev.append(_ev("The forecast keeps falling by {db} dB per decade.", db=-trend))
    return {"score": float(strength), "evidence": ev, "changeAt": _stamp(step["at"]) if drop > 0.3 else None,
            "changeDb": round(step["delta"], 2)}


def flood(vv: pd.Series, rain: pd.Series | None, rain_forecast: dict | None) -> dict:
    inund = float((vv < INUNDATION_DB).mean())
    prone = min(1.0, inund * 4)
    score, ev = 0.35 * prone, []
    if inund > 0.02:
        ev.append(_ev("{pct}% of the months looked flooded in the radar.", pct=round(100 * inund)))
    if rain is not None and len(rain) >= 24:
        spi = _spi(rain)
        if spi:
            z = spi[0]
            score += 0.35 * min(1.0, max(0.0, z / 2))
            if z > 0.7:
                ev.append(_ev("Rain of the last 3 months is {z} standard deviations above normal.", z=z))
        if rain_forecast and rain_forecast.get("forecast"):
            hist_peaks = rain.groupby(rain.index.year).max()
            ref = float(np.quantile(hist_peaks, 0.9)) if len(hist_peaks) >= 4 else float(rain.max())
            nxt = max(p["v"] for p in rain_forecast["forecast"][:12])
            ratio = nxt / ref if ref > 0 else 0
            score += 0.30 * min(1.0, max(0.0, (ratio - 0.9) / 0.6))
            if ratio > 1.0:
                ev.append(_ev("The forecast wet season peaks at {pct}% of the wettest past years.", pct=round(100 * ratio)))
    if not ev:
        ev.append(_ev("No sign of flooding in the history or in the rain."))
    return {"score": float(min(1.0, score)), "evidence": ev}


def drought_fire(rain: pd.Series | None, rain_forecast: dict | None) -> dict:
    if rain is None or len(rain) < 36:
        return {"score": 0.0, "evidence": [_ev("Not enough rain history.")], "applicable": False}
    spi = _spi(rain)
    score, ev = 0.0, []
    if spi:
        z = spi[0]
        score += 0.6 * min(1.0, max(0.0, -z / 2))
        if z < -0.7:
            ev.append(_ev("Rain of the last 3 months is {z} standard deviations below normal.", z=-z))
    if rain_forecast and rain_forecast.get("forecast"):
        nxt = rain_forecast["forecast"][:6]
        months = [pd.Timestamp(p["t"], unit="ms").month for p in nxt]
        clim = rain.groupby(rain.index.month).mean()
        expected = float(sum(clim.get(m, 0.0) for m in months))
        got = float(sum(p["v"] for p in nxt))
        ratio = got / expected if expected > 0 else 1.0
        score += 0.4 * min(1.0, max(0.0, (1 - ratio) / 0.5))
        if ratio < 0.85:
            ev.append(_ev("The next 6 months are forecast {pct}% drier than the same months of the past.", pct=round(100 * (1 - ratio))))
    if not ev:
        ev.append(_ev("Rain is normal or above, now and in the forecast."))
    return {"score": float(min(1.0, score)), "evidence": ev}


def unusual_change(vv: pd.Series, vh: pd.Series, rain: pd.Series | None) -> dict:
    from sklearn.ensemble import IsolationForest
    from sklearn.preprocessing import StandardScaler

    idx = vv.index.intersection(vh.index)
    X = pd.DataFrame({"vv": vv[idx], "vh": vh[idx]})
    X["ratio"] = X["vv"] - X["vh"]
    if rain is not None and len(rain):
        r = rain.reindex(pd.period_range(rain.index.min(), rain.index.max(), freq="M"))
        X["rain"] = r.reindex(idx)
        X["rain3"] = r.rolling(3, min_periods=3).sum().reindex(idx)
    X = X.dropna()
    if len(X) < MIN_MONTHS:
        return {"score": 0.0, "evidence": [_ev("Not enough history to find unusual months.")], "anomalies": []}
    Z = StandardScaler().fit_transform(X.to_numpy())
    forest = IsolationForest(n_estimators=300, contamination=0.1, random_state=0).fit(Z)
    flag = forest.predict(Z) == -1
    raw = -forest.score_samples(Z)
    recent = flag[-12:]
    frac = float(recent.mean())
    score = min(1.0, max(0.0, (frac - 0.1) / 0.4))
    top = [{"t": _stamp(p), "score": round(float(raw[i]), 3), "vv": round(float(X["vv"].iloc[i]), 2), "vh": round(float(X["vh"].iloc[i]), 2)}
           for i, p in enumerate(X.index) if flag[i] and i >= len(X) - 24]
    ev = [_ev("{n} of the last 12 months look unlike the rest of the history.", n=int(recent.sum()))] if recent.any() else [_ev("The last 12 months look like the rest of the history.")]
    return {"score": float(score), "evidence": ev, "anomalies": top}


# ---- everything together ----------------------------------------------------------------------------
def assess(radar: dict[str, tuple[str, pd.Series]], rain: pd.Series | None, result: dict, model=None, optical: dict | None = None) -> dict:
    """Risk assessment from the monthly radar series ({class: (pol, series)}), monthly rain and the forecast result."""
    vv = radar.get("co", (None, None))[1]
    vh = radar.get("cross", (None, None))[1]
    if vv is None or vh is None or len(vv) < MIN_MONTHS or len(vh) < MIN_MONTHS:
        return {"available": False, "reason": "Both polarizations with at least 30 months of data are needed."}
    cover = land_cover(vv, vh, model, optical)
    cls = cover["class"] if cover else None
    fc_vh = result.get("backscatter", {}).get("cross")
    fc_rain = result.get("rain")

    parts = {
        "vegetation_loss": vegetation_loss(vh, fc_vh),
        "flood": flood(vv, rain, fc_rain),
        "drought_fire": drought_fire(rain, fc_rain),
        "unusual_change": unusual_change(vv, vh, rain),
    }
    risks = []
    for rid, r in parts.items():
        applicable = r.pop("applicable", True) and (cls is None or cls in APPLICABLE[rid])
        risks.append({"id": rid, "applicable": applicable, "score": round(r["score"], 3) if applicable else 0.0,
                      "level": level(r["score"]) if applicable else "n/a", **{k: v for k, v in r.items() if k != "score"}})
    live = [r for r in risks if r["applicable"]]
    top = max(live, key=lambda r: r["score"]) if live else None
    return {
        "available": True, "landCover": cover, "risks": sorted(risks, key=lambda r: -r["score"]),
        "overall": {"score": top["score"] if top else 0.0, "level": level(top["score"]) if top else "low", "main": top["id"] if top else None},
        "months": {"radar": int(min(len(vv), len(vh))), "rain": 0 if rain is None else int(len(rain))},
        "note": "Screening signals learned from this place's own history; they say where to look first, not that an event will happen.",
    }


# ---- pre-training ---------------------------------------------------------------------------------------
def training_table(series: pd.DataFrame, optical: pd.DataFrame | None = None) -> pd.DataFrame:
    """One row per (place, year) with the cover features, from the collected reference series.
    With `optical` (one row per place, see optical.py) its columns are added to every row of the place."""
    rows = []
    series = series.assign(cls=series["cls"].replace(MERGE))
    for (place, cls), g in series.groupby(["place", "cls"]):
        wide = g.pivot_table(index="month", columns="pol_class", values="db")
        if not {"co", "cross"} <= set(wide.columns):
            continue
        wide.index = pd.PeriodIndex(wide.index, freq="M")
        for year, w in wide.groupby(wide.index.year):
            f = cover_features(w["co"], w["cross"])
            if f:
                rows.append({"place": place, "cls": cls, "year": int(year), **f})
    table = pd.DataFrame(rows)
    if optical is not None and len(table):
        table = table.merge(optical, on="place", how="inner")
    return table


def train(table: pd.DataFrame, path: Path | None = None) -> dict:
    """Fit the land-cover classifier, check it by leaving whole places out, and save it."""
    import joblib
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import GroupKFold, cross_val_predict

    if table["cls"].nunique() < 2 or table["place"].nunique() < 4:
        raise ValueError("need at least 4 places of at least 2 classes to pre-train")
    X, y, groups = table[FEATURES], table["cls"], table["place"]
    make = lambda: RandomForestClassifier(n_estimators=400, min_samples_leaf=2, class_weight="balanced", random_state=0, n_jobs=1)  # noqa: E731
    folds = min(5, groups.nunique())
    pred = cross_val_predict(make(), X, y, groups=groups, cv=GroupKFold(n_splits=folds))
    acc = float((pred == y).mean())
    per_class = {c: round(float((pred[y == c] == c).mean()), 3) for c in sorted(y.unique())}
    clf = make().fit(X, y)
    metrics = {"accuracy": round(acc, 3), "perClass": per_class, "places": int(groups.nunique()), "samples": int(len(table)),
               "validation": f"leave-places-out, {folds} folds"}
    path = path or MODEL_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"classifier": clf, "features": FEATURES, "metrics": metrics}, path)
    load_model.cache_clear()
    return metrics


def train_from_file(series_path: Path, path: Path | None = None) -> dict:
    return train(training_table(pd.read_parquet(series_path)), path)
