"""Append-only cache of processed scenes, kept apart from the output file.

A scene's cleaned rows depend only on (point, window size, processing version), so they can be
reused. Keeping the cache separate means a narrow-range run never destroys earlier work.
"""
import logging
import os
import threading
from pathlib import Path

import pandas as pd

log = logging.getLogger("asf_cleaner")
KEY = ["scene", "polarization"]


def cache_path(out, dataset: str, lon: float, lat: float, window_m: float) -> Path:
    return Path(out) / ".cache" / f"{dataset.lower()}_{lon}_{lat}_w{window_m:g}.parquet"


def load(path: Path, version: int) -> pd.DataFrame:
    """Cached rows of the current processing version (empty frame if none)."""
    if not path.exists():
        return pd.DataFrame()
    try:
        df = pd.read_parquet(path)
    except Exception as e:
        log.warning("ignoring unreadable cache %s (%s)", path, e)
        return pd.DataFrame()
    if "processing_version" not in df.columns:
        return pd.DataFrame()
    return df[df["processing_version"] == version].reset_index(drop=True)


def save(path: Path, old: pd.DataFrame, new_rows: list[dict], key: list[str] = KEY) -> None:
    """Merge new rows into the cache atomically (temp file + rename).

    The file is read again here and merged, because another job (a second API request, or the
    front-end's two streams) may have written to it since `old` was loaded; writing only `old` plus
    the new rows would silently drop what it added.
    """
    from .config import PROCESSING_VERSION

    current = load(path, PROCESSING_VERSION)  # what is on disk right now (older versions dropped)
    parts = [d for d in (old, current, pd.DataFrame(new_rows)) if len(d)]  # later parts win
    if not parts:
        return
    df = pd.concat(parts, ignore_index=True).drop_duplicates(key, keep="last")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")  # unique per writer
    df.to_parquet(tmp, index=False)
    os.replace(tmp, path)
