"""Cleaning for OPERA RTC-S1 (Sentinel-1, C-band): per-burst GeoTIFFs, gamma0 linear power."""
import numpy as np
import rasterio
from pyproj import Transformer
from rasterio.windows import Window

from .common import stats, to_db, window_bounds, window_pixels
from .stream import open_remote_tif

# OPERA RTC-S1 mask: 0 = valid, 1 = shadow, 2 = layover, 3 = layover+shadow, 255 = invalid
VALID_MASK = 0


def clean_point(tifs: dict[str, str], token: str | None, lon: float, lat: float,
                window_m: float = 330) -> list[dict]:
    """tifs maps 'VV'/'VH'/... and 'mask' to URLs (or local paths when token is None)."""
    mask_win = None
    rows = []
    for pol in [k for k in tifs if k != "mask"]:
        with open_remote_tif(tifs[pol], token) as ds:
            tx, ty = Transformer.from_crs(4326, ds.crs, always_xy=True).transform(lon, lat)
            r, c = ds.index(tx, ty)
            if not (0 <= r < ds.height and 0 <= c < ds.width):
                raise ValueError("point falls outside the burst")
            spacing = abs(ds.transform.a)
            n = window_pixels(window_m, spacing)
            r0, r1 = window_bounds(r, n, ds.height)
            c0, c1 = window_bounds(c, n, ds.width)
            win = Window(c0, r0, c1 - c0, r1 - r0)
            raw = ds.read(1, window=win).astype("float64")
            nodata = ds.nodata
            if nodata is not None and not np.isnan(nodata):
                raw[raw == nodata] = np.nan

        if mask_win is None and "mask" in tifs:
            with open_remote_tif(tifs["mask"], token) as m:
                mask_win = m.read(1, window=win)
        if mask_win is not None:
            raw[mask_win != VALID_MASK] = np.nan  # drop shadow/layover/invalid

        rows.append(
            {
                "polarization": pol,
                **stats(to_db(raw), (r - r0, c - c0)),
                "window_m": window_m,
                "pixel_spacing_m": spacing,
                "normalization": "gamma0",
            }
        )
    return rows
