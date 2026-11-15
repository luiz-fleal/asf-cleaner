"""Cleaning for OPERA RTC-S1 (Sentinel-1, C-band): per-burst GeoTIFFs, gamma0 linear power."""
import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from pyproj import Transformer
from rasterio.windows import Window

from .common import stats, to_db, window_bounds, window_pixels
from .stream import open_remote_tif

# OPERA RTC-S1 mask: 0 = valid, 1 = shadow, 2 = layover, 3 = layover+shadow, 255 = invalid
VALID_MASK = 0

# Every scene needs 2-3 independent files (VV, VH, mask). Each read is dominated by network latency
# (auth redirect, GeoTIFF header, one ranged read), so they run at the same time instead of one after
# the other. A dedicated pool, separate from the one running scenes, so scene jobs waiting for their
# files can never starve the file reads (no deadlock), while still capping total file concurrency.
_FILE_THREADS = int(os.environ.get("ASF_FILE_THREADS", "48"))
_files = ThreadPoolExecutor(max_workers=_FILE_THREADS, thread_name_prefix="asf-file")


def _read_window(url: str, token: str | None, lon: float, lat: float, window_m: float):
    """Read the window around the point from one GeoTIFF: (array, (row, col) of the point inside the
    window, pixel spacing, nodata). The window is computed from this file's own grid."""
    with open_remote_tif(url, token) as ds:
        tx, ty = Transformer.from_crs(4326, ds.crs, always_xy=True).transform(lon, lat)
        r, c = ds.index(tx, ty)
        if not (0 <= r < ds.height and 0 <= c < ds.width):
            raise ValueError("point falls outside the burst")
        spacing = abs(ds.transform.a)
        n = window_pixels(window_m, spacing)
        r0, r1 = window_bounds(r, n, ds.height)
        c0, c1 = window_bounds(c, n, ds.width)
        win = Window(c0, r0, c1 - c0, r1 - r0)
        return ds.read(1, window=win), (r - r0, c - c0), spacing, ds.nodata


def clean_point(tifs: dict[str, str], token: str | None, lon: float, lat: float,
                window_m: float = 330) -> list[dict]:
    """tifs maps 'VV'/'VH'/... and 'mask' to URLs (or local paths when token is None)."""
    pols = [k for k in tifs if k != "mask"]
    futures = {k: _files.submit(_read_window, tifs[k], token, lon, lat, window_m)
               for k in [*pols, *(["mask"] if "mask" in tifs else [])]}
    try:
        got = {k: f.result() for k, f in futures.items()}
    except BaseException:
        for f in futures.values():
            f.cancel()  # reads not started yet are dropped; the first error is what the caller sees
        raise

    bad = None
    if "mask" in got:
        bad = got["mask"][0] != VALID_MASK  # drop shadow/layover/invalid

    rows = []
    for pol in pols:
        arr, centre, spacing, nodata = got[pol]
        raw = arr.astype("float64")
        if nodata is not None and not np.isnan(nodata):
            raw[raw == nodata] = np.nan
        if bad is not None:
            if bad.shape != raw.shape:
                raise ValueError(f"mask window {bad.shape} does not match the {pol} window {raw.shape}")
            raw[bad] = np.nan
        rows.append(
            {
                "polarization": pol,
                **stats(to_db(raw), centre),
                "window_m": window_m,
                "pixel_spacing_m": spacing,
                "normalization": "gamma0",
            }
        )
    return rows
