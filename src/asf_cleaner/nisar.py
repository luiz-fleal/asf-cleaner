"""Cleaning for NISAR L2 GCOV (HDF5)."""
import numpy as np
from pyproj import Transformer

FREQ_GROUP = "/science/LSAR/GCOV/grids/frequencyA"
POLS = {"HH", "HV", "VH", "VV"}


def _epsg(h5grp) -> int:
    proj = h5grp["projection"]
    for key in ("epsg_code", "epsg"):
        if key in proj.attrs:
            return int(proj.attrs[key])
    return int(proj[()])


def _is_power_term(name: str) -> bool:
    # diagonal covariance terms: HHHH, HVHV, VHVH, VVVV
    return len(name) == 4 and name[:2] in POLS and name[:2] == name[2:]


def to_db(power: np.ndarray) -> np.ndarray:
    """Mask invalid values (NaN/inf/<=0) and convert linear power to dB."""
    p = np.asarray(power, dtype="float64")
    valid = np.isfinite(p) & (p > 0)
    out = np.full(p.shape, np.nan)
    out[valid] = 10 * np.log10(p[valid])
    return out


def clean_point(h5, lon: float, lat: float, window: int = 11) -> list[dict]:
    """Read a window around lon/lat for each diagonal covariance term."""
    grp = h5[FREQ_GROUP]
    x, y = grp["xCoordinates"], grp["yCoordinates"]
    tx, ty = Transformer.from_crs(4326, _epsg(grp), always_xy=True).transform(lon, lat)

    # coordinates are 1-D and monotonic; read only the axes, not the grids
    xs, ys = x[()], y[()]
    ix, iy = int(np.abs(xs - tx).argmin()), int(np.abs(ys - ty).argmin())
    dx, dy = abs(xs[1] - xs[0]), abs(ys[1] - ys[0])
    if abs(xs[ix] - tx) > dx or abs(ys[iy] - ty) > dy:
        raise ValueError("point falls outside the product grid")

    half = window // 2
    y0, y1 = max(iy - half, 0), min(iy + half + 1, len(ys))
    x0, x1 = max(ix - half, 0), min(ix + half + 1, len(xs))

    rows = []
    for name in grp:
        if not _is_power_term(name):
            continue
        ds = grp[name]
        raw = ds[y0:y1, x0:x1]  # ranged read: only the needed chunks
        fill = ds.attrs.get("_FillValue")
        if fill is not None and not np.isnan(fill):
            raw = np.where(raw == fill, np.nan, raw)
        db = to_db(raw)
        n_valid = int(np.isfinite(db).sum())
        rows.append(
            {
                "polarization": name[:2],
                "window_px": f"{raw.shape[0]}x{raw.shape[1]}",
                "valid_fraction": n_valid / db.size,
                "sigma0_db_mean": float(np.nanmean(db)) if n_valid else np.nan,
                "sigma0_db_median": float(np.nanmedian(db)) if n_valid else np.nan,
                "sigma0_db_center": float(db[iy - y0, ix - x0]),
            }
        )
    if not rows:
        raise ValueError("no diagonal covariance terms found in product")
    return rows
