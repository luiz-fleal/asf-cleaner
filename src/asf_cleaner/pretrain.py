"""Pre-training of the risk model on reference places whose land cover is known.

    uv run python -m asf_cleaner.pretrain collect    # read the radar history of the reference places (slow, cached)
    uv run python -m asf_cleaner.pretrain train      # fit the models from the collected series (fast)

`collect` writes `models/reference_series.parquet` (monthly VV and VH per place). `train` turns it into the
`models/risk_model.joblib` file that `risk.py` loads. The reference places are well known areas picked by hand
(dense forest, flooded plains, cleared land and savanna, which C-band radar cannot tell apart and are merged into one
class, cities, open water); see REFERENCE_POINTS.
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from . import forecast as fc
from . import history
from .search import search

log = logging.getLogger("asf_cleaner")
MODELS = Path(os.environ.get("ASF_MODELS_DIR", Path(__file__).resolve().parents[2] / "models"))
SERIES = MODELS / "reference_series.parquet"
START = datetime(2019, 1, 1, tzinfo=timezone.utc)

# (name, class, lon, lat). Cover picked by hand at places whose cover is not in doubt.
REFERENCE_POINTS = [
    # dense forest
    ("Reserva Ducke", "forest", -59.97, -2.93), ("Jau", "forest", -62.45, -1.90), ("Tapajos", "forest", -54.95, -3.10),
    ("Neblina", "forest", -66.50, 0.80), ("Serra do Divisor", "forest", -73.20, -8.30), ("Yanomami", "forest", -63.80, 2.80),
    ("Cachimbo", "forest", -55.50, -9.30), ("Tumucumaque", "forest", -54.50, 1.50), ("Juruena", "forest", -58.50, -10.50),
    ("Manu", "forest", -71.50, -12.00), ("Yasuni", "forest", -76.00, -1.00), ("Madidi", "forest", -68.00, -13.60),
    ("Serra do Mar", "forest", -45.40, -23.40),
    # floodplain, cleared land and savanna (merged into "vegetation", see risk.MERGE)
    ("Pantanal", "floodplain", -56.90, -17.70), ("Pantanal north", "floodplain", -56.60, -16.50), ("Careiro varzea", "floodplain", -59.90, -3.40),
    ("Bananal", "floodplain", -50.20, -11.80),
    ("Rondonia pasture", "cleared", -61.95, -10.88), ("Sinop cropland", "cleared", -55.50, -11.90), ("Novo Progresso", "cleared", -55.40, -7.10),
    ("Luis Eduardo Magalhaes", "cleared", -45.80, -12.10), ("Cascavel", "cleared", -53.45, -24.95), ("Uberaba", "cleared", -48.00, -19.00),
    ("Dourados", "cleared", -54.80, -22.20), ("Pergamino", "cleared", -60.60, -33.90), ("Pampa gaucho", "cleared", -54.00, -30.50),
    ("Emas", "savanna", -52.90, -18.00), ("Jalapao", "savanna", -46.60, -10.40), ("Chapada dos Veadeiros", "savanna", -47.65, -14.00),
    # cities (centers)
    ("Sao Paulo", "urban", -46.63, -23.55), ("Belo Horizonte", "urban", -43.94, -19.92), ("Rio de Janeiro", "urban", -43.20, -22.91), ("Curitiba", "urban", -49.27, -25.43),
    ("Buenos Aires", "urban", -58.40, -34.60), ("Santiago", "urban", -70.65, -33.45), ("Lima", "urban", -77.04, -12.05), ("Bogota", "urban", -74.08, 4.65),
    ("Caracas", "urban", -66.90, 10.50), ("Porto Alegre", "urban", -51.23, -30.03), ("Salvador", "urban", -38.50, -12.98), ("Recife", "urban", -34.88, -8.06),
    ("Fortaleza", "urban", -38.53, -3.73), ("Medellin", "urban", -75.57, 6.25), ("Montevideo", "urban", -56.19, -34.90), ("Quito", "urban", -78.50, -0.20),
    # open water
    ("Tucurui", "water", -49.65, -4.20), ("Lagoa dos Patos", "water", -51.30, -31.00), ("Mamiraua lake", "water", -64.90, -3.10),
    ("Titicaca", "water", -69.40, -15.80), ("Lagoa Mirim", "water", -53.10, -32.90), ("Maracaibo", "water", -71.60, 9.80),
    ("Lago Argentino", "water", -72.90, -50.40), ("Lago Buenos Aires", "water", -71.60, -46.60), ("Llanquihue", "water", -72.80, -41.10),
    ("Mar Chiquita", "water", -62.60, -30.70), ("Nahuel Huapi", "water", -71.50, -41.00),
]


def load_series() -> pd.DataFrame:
    """Everything collected so far: the main file plus the files of parallel shards."""
    parts = [pd.read_parquet(p) for p in sorted(MODELS.glob("reference_series*.parquet"))]
    if not parts:
        return pd.DataFrame()
    return pd.concat(parts, ignore_index=True).drop_duplicates(["place", "pol_class", "month"], keep="last")


def collect(cache_dir: str, workers: int = 8, shard: tuple[int, int] = (0, 1)) -> pd.DataFrame:
    """Read the radar history of the reference places not collected yet. `shard` (i, n) takes every n-th of the
    missing places, so several processes can work at once; each writes its own file."""
    from .config import get_token

    get_token()
    i, n = shard
    done = set(load_series().get("place", []))
    todo = [p for p in REFERENCE_POINTS if p[0] not in done][i::n]
    out = MODELS / ("reference_series.parquet" if n == 1 else f"reference_series.shard{i}.parquet")
    rows = pd.read_parquet(out).to_dict("records") if out.exists() else []
    for name, cls, lon, lat in todo:
        t0 = time.perf_counter()
        try:
            data = None
            for ev, payload in history.radar_history(lon, lat, search_fn=search, cache_dir=cache_dir, workers=workers, start=START):
                if ev == "data":
                    data = payload
            radar, _, count = data
            for pol_class, (pol, series) in fc.monthly_backscatter(radar).items():
                for m, v in series.items():
                    rows.append({"place": name, "cls": cls, "lon": lon, "lat": lat, "pol_class": pol_class, "month": str(m), "db": float(v)})
            log.info("%s (%s): %d scenes in %.0fs", name, cls, count, time.perf_counter() - t0)
        except Exception as e:  # noqa: BLE001 - one place failing must not lose the others
            log.warning("%s failed: %s", name, e)
        MODELS.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_parquet(out, index=False)
    return pd.DataFrame(rows)


OPTICAL = MODELS / "reference_optical.parquet"


def collect_optical() -> pd.DataFrame:
    """The optical (Sentinel-2 color) features of every reference place: one tile request each."""
    from . import optical

    rows = []
    for name, cls, lon, lat in REFERENCE_POINTS:
        try:
            rows.append({"place": name, **optical.features(lon, lat)})
        except Exception as e:  # noqa: BLE001 - one place failing must not lose the others
            log.warning("%s: no optical features (%s)", name, e)
    MODELS.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_parquet(OPTICAL, index=False)
    log.info("optical features for %d of %d places", len(df), len(REFERENCE_POINTS))
    return df


def load_optical() -> pd.DataFrame | None:
    return pd.read_parquet(OPTICAL) if OPTICAL.exists() else None


def main(argv=None):
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description="Pre-train the risk model")
    ap.add_argument("step", choices=["collect", "optical", "train"])
    ap.add_argument("--cache", default=os.environ.get("ASF_CACHE_DIR", "out"))
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--shard", default="0/1", help="i/n: take every n-th missing place (run n processes at once)")
    a = ap.parse_args(argv)
    if a.step == "optical":
        collect_optical()
    elif a.step == "collect":
        i, n = map(int, a.shard.split("/"))
        collect(a.cache, a.workers, (i, n))
    else:
        from . import training

        training.main([])


if __name__ == "__main__":
    sys.exit(main())
