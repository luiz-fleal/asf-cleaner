"""Ready-made forecasts for whole regions, computed ahead so nothing has to load when someone picks one.

A region is a handful of sample places spread over it (cached histories where possible). Each sample goes through the
same forecast and risk code as any clicked point, then the results are averaged into one result with the same shape,
so a client shows a region exactly like a point. The result keeps the riskiest sample of every risk (where to look
first) and lists the samples, so nothing is hidden by the averaging.

    uv run python -m asf_cleaner.regions build            # all regions (slow the first time, then cached)
    uv run python -m asf_cleaner.regions build amazon     # one region

Files go to `models/regions/<id>.json` (override with ASF_REGIONS_DIR) and the API serves them at /regions.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger("asf_cleaner")
REGIONS_DIR = Path(os.environ.get("ASF_REGIONS_DIR", Path(__file__).resolve().parents[2] / "models" / "regions"))

REGIONS = {
    "amazon": {
        "name": "Amazon", "center": [-62.0, -4.5], "zoom": 4.4, "bounds": [-74.0, -14.0, -48.0, 5.0],
        "samples": [("Reserva Ducke", -59.97, -2.93), ("Jaú", -62.45, -1.90), ("Tapajós", -54.95, -3.10), ("Novo Progresso", -55.40, -7.10),
                    ("Rondônia pasture", -61.95, -10.88), ("Yanomami", -63.80, 2.80)],
    },
    "sao-paulo": {
        "name": "São Paulo", "center": [-47.6, -22.6], "zoom": 6.2, "bounds": [-53.2, -25.4, -44.0, -19.7],
        "samples": [("São Paulo city", -46.63, -23.55), ("Serra do Mar", -45.40, -23.40), ("Campinas", -47.06, -22.91), ("Ribeirão Preto", -47.81, -21.18)],
    },
    "rio-grande-do-sul": {
        "name": "Rio Grande do Sul", "center": [-53.2, -30.2], "zoom": 5.8, "bounds": [-57.7, -33.8, -49.6, -27.0],
        "samples": [("Porto Alegre", -51.23, -30.03), ("Pampa gaúcho", -54.00, -30.50), ("Lagoa dos Patos", -51.30, -31.00), ("Caxias do Sul", -51.18, -29.17)],
    },
}


# ---- averaging ------------------------------------------------------------------------------------
def _mean_points(lists: list[list[dict]]) -> list[dict]:
    by_t: dict[int, list[dict]] = {}
    for pts in lists:
        for p in pts:
            by_t.setdefault(p["t"], []).append(p)
    out = []
    for t in sorted(by_t):
        ps = by_t[t]
        item = {"t": t, "v": round(float(np.mean([p["v"] for p in ps])), 4)}
        if all("lo" in p for p in ps):
            item["lo"] = round(float(np.mean([p["lo"] for p in ps])), 4)
            item["hi"] = round(float(np.mean([p["hi"] for p in ps])), 4)
        out.append(item)
    return out


def _yearly(forecast: list[dict]) -> list[dict]:
    df = pd.DataFrame({"year": [pd.Timestamp(p["t"], unit="ms").year for p in forecast], "v": [p["v"] for p in forecast]})
    return [{"year": int(y), "mean": round(float(g["v"].mean()), 3), "total": round(float(g["v"].sum()), 1), "months": int(len(g))} for y, g in df.groupby("year")]


def _average_series(series: list[dict], label: str) -> dict:
    trends = [s["trendPerDecade"] for s in series if s.get("trendPerDecade") is not None]
    mean = lambda key: round(float(np.mean([s["backtest"][key] for s in series])), 4)  # noqa: E731
    forecast = _mean_points([s["forecast"] for s in series])
    return {
        "model": f"average of {len(series)} places", "reliable": sum(bool(s["reliable"]) for s in series) > len(series) / 2,
        "pol": series[0].get("pol"), "trendPerDecade": round(float(np.mean(trends)), 4) if trends else None, "bandWidth": series[0].get("bandWidth", 0.8),
        "history": _mean_points([s["history"] for s in series]), "forecast": forecast, "yearly": _yearly(forecast),
        "backtest": {"mae": mean("mae"), "rmse": mean("rmse"), "naiveMae": mean("naiveMae"), "skill": mean("skill"), "folds": min(s["backtest"]["folds"] for s in series),
                     "months": int(sum(s["backtest"]["months"] for s in series)), "candidates": {}},
        "places": len(series), "label": label,
    }


def _years_of(r: dict) -> dict:
    """The forecast of one sample place for each year: rain (mm per year) and the co-polarized backscatter (dB)."""
    out: dict[str, dict] = {}
    for y in (r.get("rain") or {}).get("yearly", []):
        if y["months"] >= 12:
            out.setdefault(str(y["year"]), {})["rain"] = round(y["total"])
    for y in (r.get("backscatter", {}).get("co") or {}).get("yearly", []):
        if y["months"] >= 12:
            out.setdefault(str(y["year"]), {})["vv"] = round(y["mean"], 2)
    return out


LEVELS = {"low": 0, "moderate": 1, "high": 2}


def aggregate(results: list[dict], samples: list[tuple]) -> dict:
    """One result, shaped like a point's, from the results of the sample places (`samples` = (name, lon, lat) per result)."""
    if not results:
        raise ValueError("no sample finished")
    out: dict = {"until": results[0]["until"], "rainScale": results[0].get("rainScale", 1.0), "backscatter": {}, "notes": []}
    for cls in ("co", "cross"):
        ser = [r["backscatter"][cls] for r in results if cls in r.get("backscatter", {})]
        if ser:
            out["backscatter"][cls] = _average_series(ser, cls)
    rains = [r["rain"] for r in results if r.get("rain")]
    if rains:
        out["rain"] = _average_series(rains, "rain")

    # risks: the riskiest sample of each one, so a hot spot is never averaged away
    risk_items: dict[str, list[tuple[dict, str]]] = {}
    for r, (name, *_rest) in zip(results, samples):
        for item in (r.get("risk") or {}).get("risks", []):
            risk_items.setdefault(item["id"], []).append((item, name))
    risks = []
    for rid, items in risk_items.items():
        live = [(i, n) for i, n in items if i["applicable"]]
        if not live:
            risks.append({"id": rid, "applicable": False, "score": 0.0, "level": "n/a", "evidence": [], "places": 0})
            continue
        best, where = max(live, key=lambda x: x[0]["score"])
        risks.append({**best, "place": where, "places": len(live), "meanScore": round(float(np.mean([i["score"] for i, _ in live])), 3),
                      "evidence": best["evidence"]})
    risks.sort(key=lambda r: -r["score"])
    live_risks = [r for r in risks if r["applicable"]]
    top = live_risks[0] if live_risks else None
    covers = Counter((r.get("risk") or {}).get("landCover", {}) and r["risk"]["landCover"]["class"] for r in results if (r.get("risk") or {}).get("landCover"))
    out["risk"] = {
        "available": bool(risks), "risks": risks,
        "overall": {"score": top["score"] if top else 0.0, "level": top["level"] if top else "low", "main": top["id"] if top else None, "place": top.get("place") if top else None},
        "landCover": ({"class": covers.most_common(1)[0][0], "confidence": round(covers.most_common(1)[0][1] / sum(covers.values()), 3),
                       "mix": {k: round(v / sum(covers.values()), 3) for k, v in covers.items()}, "source": "region samples", "inputs": "radar + optical",
                       "trainedOn": None, "cvAccuracy": None} if covers else None),
        "note": "Region summary: for every risk, the riskiest of the sample places is shown (where to look first); the samples are listed below.",
    }
    out["samples"] = [{"name": name, "lon": lon, "lat": lat, "level": (r.get("risk") or {}).get("overall", {}).get("level"), "main": (r.get("risk") or {}).get("overall", {}).get("main"),
                       "cover": ((r.get("risk") or {}).get("landCover") or {}).get("class"), "years": _years_of(r)} for r, (name, lon, lat) in zip(results, samples)]
    out["notes"] = [f"Average of {len(results)} sample places spread over the region.", *results[0].get("notes", [])[-2:]]
    out["meta"] = {"radarScenes": int(sum(r["meta"].get("radarScenes", 0) for r in results)), "rainMonths": int(max(r["meta"].get("rainMonths", 0) for r in results))}
    return out


# ---- files ----------------------------------------------------------------------------------------
def path_of(region_id: str) -> Path:
    return REGIONS_DIR / f"{region_id}.json"


def load(region_id: str) -> dict | None:
    p = path_of(region_id)
    return json.loads(p.read_text()) if p.exists() else None


def listing() -> list[dict]:
    out = []
    for rid, r in REGIONS.items():
        data = load(rid)
        out.append({"id": rid, "name": r["name"], "center": r["center"], "zoom": r["zoom"], "bounds": r["bounds"], "samples": len(r["samples"]),
                    "ready": data is not None, "builtAt": data.get("builtAt") if data else None,
                    "overall": data["risk"]["overall"] if data and data.get("risk") else None})
    return out


def build(region_id: str, compute, only_missing: bool = False) -> dict:
    """Compute every sample with `compute(lon, lat) -> forecast result`, average them and save the file."""
    if only_missing and path_of(region_id).exists():
        return load(region_id)
    spec = REGIONS[region_id]
    results, used = [], []
    for name, lon, lat in spec["samples"]:
        t0 = time.perf_counter()
        try:
            results.append(compute(lon, lat))
            used.append((name, lon, lat))
            log.info("%s / %s: %.0fs", region_id, name, time.perf_counter() - t0)
        except Exception as e:  # noqa: BLE001 - one sample failing must not lose the region
            log.warning("%s / %s failed: %s", region_id, name, e)
    data = aggregate(results, used)
    data["region"] = {"id": region_id, "name": spec["name"], "center": spec["center"], "zoom": spec["zoom"], "bounds": spec["bounds"]}
    data["builtAt"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    REGIONS_DIR.mkdir(parents=True, exist_ok=True)
    tmp = path_of(region_id).with_suffix(".tmp")
    tmp.write_text(json.dumps(data, separators=(",", ":")))
    os.replace(tmp, path_of(region_id))
    return data


def main(argv=None):
    from . import api  # the same code that serves a clicked point

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    ap = argparse.ArgumentParser(description="Build the ready-made region forecasts")
    ap.add_argument("cmd", choices=["build"])
    ap.add_argument("ids", nargs="*", choices=[[], *REGIONS][1:] or None, help="regions to build (default: all)")
    ap.add_argument("--missing", action="store_true", help="skip regions that already have a file")
    a = ap.parse_args(argv)
    for rid in a.ids or REGIONS:
        build(rid, api.forecast_result, a.missing)


if __name__ == "__main__":
    main()
