"""Cleaning helpers shared by all sensors so outputs are computed identically."""
import numpy as np


def to_db(power: np.ndarray) -> np.ndarray:
    """Mask invalid values (NaN/inf/<=0) and convert linear power to dB."""
    p = np.asarray(power, dtype="float64")
    valid = np.isfinite(p) & (p > 0)
    out = np.full(p.shape, np.nan)
    out[valid] = 10 * np.log10(p[valid])
    return out


def window_pixels(window_m: float, spacing_m: float) -> int:
    """Odd number of pixels covering ~window_m on the ground."""
    n = max(1, round(window_m / spacing_m))
    return n if n % 2 else n + 1


def window_bounds(center: int, n_pix: int, size: int) -> tuple[int, int]:
    half = n_pix // 2
    return max(center - half, 0), min(center + half + 1, size)


def stats(db: np.ndarray, center_rc: tuple[int, int]) -> dict:
    """Summary of a cleaned dB window. center_rc is the point's (row, col) in the window."""
    finite = np.isfinite(db)
    n = int(finite.sum())
    return {
        "window_px": f"{db.shape[0]}x{db.shape[1]}",
        "valid_fraction": n / db.size,
        "backscatter_db_mean": float(np.nanmean(db)) if n else np.nan,
        "backscatter_db_median": float(np.nanmedian(db)) if n else np.nan,
        "backscatter_db_center": float(db[center_rc]),
    }
