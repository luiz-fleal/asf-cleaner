import h5py
import numpy as np
from pyproj import Transformer

from asf_cleaner.nisar import FREQ_GROUP, clean_point, to_db


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
        g["HHHV"] = d  # off-diagonal term, must be ignored


def test_to_db():
    out = to_db(np.array([1.0, 0.1, 0.0, np.nan, -1, np.inf]))
    assert out[0] == 0 and np.isclose(out[1], -10)
    assert np.isnan(out[2:]).all()


def test_clean_point(tmp_path):
    p = tmp_path / "t.h5"
    make_h5(p)
    with h5py.File(p) as h5:
        rows = clean_point(h5, -47.9, -15.8, window=5)
    assert {r["polarization"] for r in rows} == {"HH", "HV"}
    hh = next(r for r in rows if r["polarization"] == "HH")
    assert hh["valid_fraction"] < 1 and np.isclose(hh["sigma0_db_median"], -10)


def test_outside(tmp_path):
    import pytest

    p = tmp_path / "t.h5"
    make_h5(p)
    with h5py.File(p) as h5, pytest.raises(ValueError):
        clean_point(h5, 10.0, 10.0)
