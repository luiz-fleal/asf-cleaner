"""Optical (Sentinel-2 color) features: tile math and color statistics, with synthetic tiles (no network)."""
import numpy as np
import pytest

from asf_cleaner import optical


def test_tile_pixel_matches_the_web_mercator_grid():
    x, y, px, py = optical.tile_pixel(0.0, 0.0, 1)
    assert (x, y, px, py) == (1, 1, 0, 0)  # the origin is the corner of tiles (1,1)
    x, y, px, py = optical.tile_pixel(-60.0185, -3.116, 14)
    assert (x, y) == (5461, 8213) or abs(x - 5461) <= 1  # Manaus
    assert 0 <= px < 256 and 0 <= py < 256


def tile(color, noise=0.0, seed=0):
    rng = np.random.default_rng(seed)
    base = np.ones((256, 256, 3), "float32") * np.array(color, "float32")
    return np.clip(base + rng.normal(0, noise, base.shape), 0, 255).astype("float32")


def test_forest_water_and_city_look_different():
    forest = optical.features_from_tile(tile((19, 45, 21), 2), 128, 128)
    water = optical.features_from_tile(tile((33, 44, 26), 1), 128, 128)
    city = optical.features_from_tile(tile((112, 106, 91), 40), 128, 128)
    assert city["opt_texture"] > 5 * forest["opt_texture"] > 0 and water["opt_texture"] < forest["opt_texture"] + 1
    assert forest["opt_exg"] > 40 > city["opt_exg"] and city["opt_brightness"] > 3 * forest["opt_brightness"]
    assert set(forest) == set(optical.FEATURES)


def test_a_point_near_the_tile_edge_still_gets_features():
    f = optical.features_from_tile(tile((50, 80, 40)), 0, 255)
    assert f["opt_g"] == pytest.approx(80) and f["opt_context_brightness"] == pytest.approx((50 + 80 + 40) / 3)


def test_features_read_one_tile(monkeypatch):
    seen = []
    monkeypatch.setattr(optical, "fetch_tile", lambda x, y, z=14: seen.append((x, y, z)) or tile((19, 45, 21)))
    f = optical.features(-60.0185, -3.116)
    assert len(seen) == 1 and seen[0][2] == 14 and f["opt_g"] == pytest.approx(45)
