import argparse
import logging
from pathlib import Path

import pandas as pd

from . import others
from .config import Settings, get_token
from .nisar import clean_point
from .search import search
from .stream import open_remote_h5

log = logging.getLogger("asf_cleaner")


def run_nisar(s: Settings) -> pd.DataFrame:
    token = get_token()
    scenes = search("NISAR", s.lon, s.lat, s.start, s.end, s.max_results)
    log.info("found %d scenes", len(scenes))
    rows = []
    for sc in scenes:
        try:
            with open_remote_h5(sc["url"], token) as h5:
                for r in clean_point(h5, s.lon, s.lat, s.window):
                    rows.append({**{k: v for k, v in sc.items() if k != "url"}, **r})
            log.info("ok  %s", sc["scene"])
        except Exception as e:  # keep going on per-scene failures
            log.warning("skip %s: %s", sc["scene"], e)
    return pd.DataFrame(rows)


ADAPTERS = {"NISAR": run_nisar, "SENTINEL-1": others.sentinel1, "OPERA": others.opera}


def main(argv=None):
    d = Settings()
    ap = argparse.ArgumentParser(description="Stream & clean ASF NISAR GCOV data at a point")
    ap.add_argument("--dataset", default=d.dataset, choices=ADAPTERS)
    ap.add_argument("--lon", type=float, default=d.lon)
    ap.add_argument("--lat", type=float, default=d.lat)
    ap.add_argument("--start", default=d.start)
    ap.add_argument("--end", default=d.end)
    ap.add_argument("--max-results", type=int, default=d.max_results)
    ap.add_argument("--window", type=int, default=d.window, help="odd pixel window size")
    ap.add_argument("--out", default=d.out)
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    s = Settings(a.lon, a.lat, a.start, a.end, a.max_results, a.window, a.out, a.dataset)
    df = ADAPTERS[s.dataset](s)
    if df.empty:
        log.warning("no data produced")
        return
    out = Path(s.out)
    out.mkdir(parents=True, exist_ok=True)
    stem = out / f"{s.dataset.lower()}_{s.lon}_{s.lat}"
    df.to_csv(f"{stem}.csv", index=False)
    df.to_parquet(f"{stem}.parquet", index=False)
    log.info("wrote %s.{csv,parquet} (%d rows)", stem, len(df))


if __name__ == "__main__":
    main()
