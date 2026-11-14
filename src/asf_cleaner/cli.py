import argparse
import logging
from pathlib import Path

import pandas as pd

from . import nisar, opera
from .config import Settings, get_token, resolve_dataset, resolve_range
from .harmonize import harmonize
from .search import search
from .stream import open_remote_h5

log = logging.getLogger("asf_cleaner")
META = ("scene", "burst_id", "track", "direction", "start_time", "platform")


def run_nisar(s: Settings, token: str) -> list[dict]:
    rows = []
    for sc in search("NISAR", s.lon, s.lat, s.start, s.end, s.max_results):
        try:
            with open_remote_h5(sc["url"], token) as h5:
                for r in nisar.clean_point(h5, s.lon, s.lat, s.window_m):
                    rows.append({"sensor": "NISAR", **{k: sc[k] for k in META}, **r})
            log.info("ok  %s", sc["scene"])
        except Exception as e:  # keep going on per-scene failures
            log.warning("skip %s: %s", sc["scene"], e)
    return rows


def run_opera(s: Settings, token: str) -> list[dict]:
    rows = []
    for sc in search("OPERA", s.lon, s.lat, s.start, s.end, s.max_results):
        try:
            for r in opera.clean_point(sc["tifs"], token, s.lon, s.lat, s.window_m):
                rows.append({"sensor": "OPERA", **{k: sc[k] for k in META}, **r})
            log.info("ok  %s", sc["scene"])
        except Exception as e:
            log.warning("skip %s: %s", sc["scene"], e)
    return rows


ADAPTERS = {"NISAR": run_nisar, "OPERA": run_opera}


def main(argv=None):
    d = Settings()
    ap = argparse.ArgumentParser(description="Stream & clean ASF NISAR / OPERA-S1 data at a point")
    ap.add_argument("--dataset", default=d.dataset, choices=["auto", *ADAPTERS],
                    help="auto: NISAR if the range is inside NISAR coverage, else OPERA")
    ap.add_argument("--lon", type=float, default=d.lon)
    ap.add_argument("--lat", type=float, default=d.lat)
    ap.add_argument("--start", help="e.g. 2026-06-01 (default: 180 days before --end)")
    ap.add_argument("--end", help="e.g. 2026-11-13 (default: now)")
    ap.add_argument("--max-results", type=int, default=d.max_results,
                    help="cap on total scenes; newest are kept and a warning is logged if exceeded")
    ap.add_argument("--window-m", type=float, default=d.window_m, help="ground footprint in meters")
    ap.add_argument("--out", default=d.out)
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    start, end = resolve_range(a.start, a.end)
    dataset, why = resolve_dataset(a.dataset, start, end)
    log.info("mode: %s - %s", dataset, why)
    s = Settings(a.lon, a.lat, start, end, a.max_results, a.window_m, a.out, dataset)
    rows = ADAPTERS[dataset](s, get_token())
    if not rows:
        log.warning("no data produced (no scenes at this point/range, or all scenes failed)")
        return
    df = harmonize(pd.DataFrame(rows))
    out = Path(s.out)
    out.mkdir(parents=True, exist_ok=True)
    stem = out / f"{s.dataset.lower()}_{s.lon}_{s.lat}"
    df.to_csv(f"{stem}.csv", index=False)
    df.to_parquet(f"{stem}.parquet", index=False)
    log.info("wrote %s.{csv,parquet} (%d rows)", stem, len(df))


if __name__ == "__main__":
    main()
