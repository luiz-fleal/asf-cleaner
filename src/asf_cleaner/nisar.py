"""Cleaning for NISAR L2 GCOV (HDF5, L-band)."""
import logging

import numpy as np
from pyproj import Transformer

from .common import stats, to_db, window_bounds, window_pixels

log = logging.getLogger("asf_cleaner")

FREQ_GROUP = "/science/LSAR/GCOV/grids/frequencyA"
MASK_INVALID = 0
MASK_OUTSIDE = 255
POLS = {"HH", "HV", "VH", "VV"}


def _epsg(grp) -> int:
    proj = grp["projection"]
    for key in ("epsg_code", "epsg"):
        if key in proj.attrs:
            return int(proj.attrs[key])
    return int(proj[()])


def _is_power_term(name: str) -> bool:
    # diagonal covariance terms: HHHH, HVHV, VHVH, VVVV
    return len(name) == 4 and name[:2] in POLS and name[:2] == name[2:]


def _normalization(grp, ds) -> str:
    """Radiometric normalization. GCOV stores gamma0 and ships `rtcGammaToSigmaFactor` to
    convert it to sigma0; fall back to the dataset description otherwise."""
    if "rtcGammaToSigmaFactor" in grp:
        return "gamma0"
    desc = ds.attrs.get("description", b"")
    desc = desc.decode() if isinstance(desc, bytes) else str(desc)
    for word in ("gamma0", "sigma0", "beta0"):
        if word in desc.lower().replace("-", "").replace(" ", ""):
            return word
    return "unknown"


def clean_point(h5, lon: float, lat: float, window_m: float = 330) -> list[dict]:
    """Read a ~window_m square around lon/lat for each diagonal covariance term."""
    grp = h5[FREQ_GROUP]
    tx, ty = Transformer.from_crs(4326, _epsg(grp), always_xy=True).transform(lon, lat)

    # coordinates are 1-D and monotonic; read only the axes, not the grids
    xs, ys = grp["xCoordinates"][()], grp["yCoordinates"][()]
    ix, iy = int(np.abs(xs - tx).argmin()), int(np.abs(ys - ty).argmin())
    dx, dy = abs(xs[1] - xs[0]), abs(ys[1] - ys[0])
    if abs(xs[ix] - tx) > dx or abs(ys[iy] - ty) > dy:
        raise ValueError("point falls outside the product grid")

    n = window_pixels(window_m, (dx + dy) / 2)
    y0, y1 = window_bounds(iy, n, len(ys))
    x0, x1 = window_bounds(ix, n, len(xs))

    # Valid-samples mask: 0 = not fully focused (invalid), 1..N = subswath number (valid),
    # 255 = outside the acquisition. NOTE: opposite of OPERA, where 0 means valid.
    bad = None
    if "mask" in grp:
        mds = grp["mask"]
        m = mds[y0:y1, x0:x1]
        bad = (m == MASK_INVALID) | (m == mds.attrs.get("_FillValue", MASK_OUTSIDE))
    else:
        log.warning("no 'mask' dataset in product; invalid samples are not filtered")

    rows = []
    for name in grp:
        if not _is_power_term(name):
            continue
        ds = grp[name]
        raw = ds[y0:y1, x0:x1]  # ranged read: only the needed chunks
        fill = ds.attrs.get("_FillValue")
        if fill is not None and not np.isnan(fill):
            raw = np.where(raw == fill, np.nan, raw)
        if bad is not None:
            raw = np.where(bad, np.nan, raw)
        rows.append(
            {
                "polarization": name[:2],
                **stats(to_db(raw), (iy - y0, ix - x0)),
                "window_m": window_m,
                "pixel_spacing_m": float((dx + dy) / 2),
                "normalization": _normalization(grp, ds),
            }
        )
    if not rows:
        raise ValueError("no diagonal covariance terms found in product")
    return rows
