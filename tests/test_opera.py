import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import from_origin

from asf_cleaner.compare import compare
from asf_cleaner.harmonize import harmonize
from asf_cleaner.opera import clean_point
from asf_cleaner.search import search  # noqa: F401  (import check)


def write_tif(path, arr, dtype):
    from pyproj import Transformer
    tx, ty = Transformer.from_crs(4326, 32723, always_xy=True).transform(-47.9, -15.8)
    t = from_origin(tx - 15 * 30, ty + 15 * 30, 30, 30)
    with rasterio.open(path, "w", driver="GTiff", height=30, width=30, count=1, dtype=dtype,
                       crs="EPSG:32723", transform=t) as f:
        f.write(arr.astype(dtype), 1)


def test_opera_mask_and_db(tmp_path):
    vv = np.full((30, 30), 0.1, "float32")
    vv[15, 15] = np.nan
    mask = np.zeros((30, 30), "uint8")
    mask[14, 14] = 2  # layover pixel must be dropped
    write_tif(tmp_path / "vv.tif", vv, "float32")
    write_tif(tmp_path / "vh.tif", vv * 0.1, "float32")
    write_tif(tmp_path / "m.tif", mask, "uint8")
    rows = clean_point({"VV": str(tmp_path / "vv.tif"), "VH": str(tmp_path / "vh.tif"),
                        "mask": str(tmp_path / "m.tif")}, None, -47.9, -15.8, window_m=90)
    vv_row = next(r for r in rows if r["polarization"] == "VV")
    assert vv_row["window_px"] == "3x3" and np.isclose(vv_row["valid_fraction"], 7 / 9)
    assert np.isclose(vv_row["backscatter_db_median"], -10)
    assert vv_row["normalization"] == "gamma0"


def _rows(sensor, pols, vals, track=1):
    return [dict(sensor=sensor, platform=sensor, scene=f"s{i}", burst_id=None, track=track,
                 direction="A", start_time=f"2026-0{i+1}-01T00:00:00Z", polarization=p,
                 valid_fraction=1, backscatter_db_mean=v, backscatter_db_median=v,
                 backscatter_db_center=v, window_m=330, window_px="11x11",
                 pixel_spacing_m=30, normalization="gamma0")
            for p in pols for i, v in enumerate(vals)]


def test_harmonize_and_compare():
    n = harmonize(pd.DataFrame(_rows("NISAR", ["HH", "HV"], [-10, -9, -8, -9, -10])))
    o = harmonize(pd.DataFrame(_rows("OPERA", ["VV", "VH"], [-15, -14, -13, -14, -15])))
    assert set(n.pol_class) == {"co", "cross"} and set(n.band) == {"L"} and set(o.band) == {"C"}
    assert abs(n.z_median[n.polarization == "HH"].mean()) < 1e-9
    res = compare(n, o)
    assert (res.n_pairs == 5).all() and np.allclose(res.z_correlation, 1)
    assert np.allclose(res.mean_offset_db_nisar_minus_opera, 5)
