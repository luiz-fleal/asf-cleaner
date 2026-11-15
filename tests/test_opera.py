import time

import numpy as np
import pandas as pd
import pytest
import rasterio
from rasterio.transform import from_origin

from asf_cleaner.harmonize import harmonize
from asf_cleaner import opera
from asf_cleaner.opera import clean_point


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


def test_opera_files_are_read_at_the_same_time(tmp_path, monkeypatch):
    """VV, VH and the mask are independent downloads: they must overlap, not queue."""
    real = opera._read_window

    def slow_read(*a):
        time.sleep(0.3)
        return real(*a)

    monkeypatch.setattr(opera, "_read_window", slow_read)
    arr = np.full((30, 30), 0.1, "float32")
    for name in ("vv", "vh"):
        write_tif(tmp_path / f"{name}.tif", arr, "float32")
    write_tif(tmp_path / "m.tif", np.zeros((30, 30), "uint8"), "uint8")
    t = time.perf_counter()
    rows = clean_point({"VV": str(tmp_path / "vv.tif"), "VH": str(tmp_path / "vh.tif"),
                        "mask": str(tmp_path / "m.tif")}, None, -47.9, -15.8, window_m=90)
    assert len(rows) == 2 and time.perf_counter() - t < 0.8  # one after the other would be 0.9s+


def test_opera_failed_file_raises_its_error(tmp_path):
    write_tif(tmp_path / "vv.tif", np.full((30, 30), 0.1, "float32"), "float32")
    with pytest.raises(ValueError, match="outside the burst"):  # the point is not inside this raster
        clean_point({"VV": str(tmp_path / "vv.tif")}, None, 10.0, 10.0, window_m=90)


def _rows(sensor, pols, vals, track=1):
    return [dict(sensor=sensor, platform=sensor, scene=f"s{i}", burst_id=None, track=track,
                 direction="A", start_time=f"2026-0{i+1}-01T00:00:00Z", polarization=p,
                 valid_fraction=1, backscatter_db_mean=v, backscatter_db_median=v,
                 backscatter_db_center=v, window_m=330, window_px="11x11",
                 pixel_spacing_m=30, normalization="gamma0", processing_version=2)
            for p in pols for i, v in enumerate(vals)]


def test_harmonize():
    rows = _rows("NISAR", ["HH", "HV"], [-10, -9, -8]) + _rows("NISAR", ["HH"], [-10])[:1]  # duplicate
    n = harmonize(pd.DataFrame(rows))
    assert len(n) == 6  # duplicate scene/pol dropped
    assert set(n.pol_class) == {"co", "cross"} and set(n.band) == {"L"}
    assert "z_median" not in n.columns
