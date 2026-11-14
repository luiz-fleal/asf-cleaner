import h5py
import numpy as np
from pyproj import Transformer

from asf_cleaner.common import to_db
from asf_cleaner.nisar import FREQ_GROUP, clean_point


def make_h5(path):
    tx, ty = Transformer.from_crs(4326, 32723, always_xy=True).transform(-47.9, -15.8)
    xs = tx + np.arange(-500, 500) * 20.0
    ys = ty - np.arange(-500, 500) * 20.0
    with h5py.File(path, "w") as f:
        g = f.create_group(FREQ_GROUP)
        g["xCoordinates"], g["yCoordinates"] = xs, ys
        g["projection"] = np.uint32(32723)
        d = np.full((1000, 1000), 0.1, dtype="float32")
        d[500, 500] = np.nan
        d[501, 500] = 0.0
        g["HHHH"] = d
        g["HVHV"] = d * 0.1
        g["HHHH"].attrs["description"] = "Radar backscatter gamma0 power"
        g["HHHV"] = d  # off-diagonal term, must be ignored
        g["rtcGammaToSigmaFactor"] = np.ones((4, 4), "float32")
        mask = np.ones((1000, 1000), "uint8")  # subswath 1 = valid
        mask[499, 500] = 0      # not fully focused -> must be dropped
        mask[500, 501] = 255    # outside acquisition -> must be dropped
        g["mask"] = mask
        g["mask"].attrs["_FillValue"] = np.uint8(255)


def test_to_db():
    out = to_db(np.array([1.0, 0.1, 0.0, np.nan, -1, np.inf]))
    assert out[0] == 0 and np.isclose(out[1], -10)
    assert np.isnan(out[2:]).all()


def test_clean_point(tmp_path):
    p = tmp_path / "t.h5"
    make_h5(p)
    with h5py.File(p) as h5:
        rows = clean_point(h5, -47.9, -15.8, window_m=100)
    assert {r["polarization"] for r in rows} == {"HH", "HV"}
    hh = next(r for r in rows if r["polarization"] == "HH")
    # 5x5 window: 1 NaN + 1 zero power + mask 0 + mask 255 = 4 invalid of 25
    assert np.isclose(hh["valid_fraction"], 21 / 25)
    assert np.isclose(hh["backscatter_db_median"], -10)
    assert hh["normalization"] == "gamma0"


def test_missing_mask_warns(tmp_path, caplog):
    p = tmp_path / "t.h5"
    make_h5(p)
    with h5py.File(p, "a") as f:
        del f[FREQ_GROUP]["mask"]
    with h5py.File(p) as h5, caplog.at_level("WARNING", logger="asf_cleaner"):
        rows = clean_point(h5, -47.9, -15.8, window_m=100)
    assert "no 'mask'" in caplog.text
    assert np.isclose(rows[0]["valid_fraction"], 23 / 25)


def test_outside(tmp_path):
    import pytest

    p = tmp_path / "t.h5"
    make_h5(p)
    with h5py.File(p) as h5, pytest.raises(ValueError):
        clean_point(h5, 10.0, 10.0)
