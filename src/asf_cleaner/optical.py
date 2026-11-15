"""Optical context of a place: the color of the Sentinel-2 cloudless mosaic (EOX, 10 m) around the point.

C-band radar alone cannot tell some covers apart (cleared land from savanna from floodplain), but the color of the
ground can: water is dark and blue, forest dark green, farmland bright green or brown, a city gray and rough. These
features come from one 256-pixel tile (zoom 14, about 10 m per pixel) read from the same mosaic the map shows, so they
are the same for any year and cost one request per place.
"""
from __future__ import annotations

import io
import math

import numpy as np
import requests

from .runner import retry

TILE_URL = "https://tiles.maps.eox.at/wmts/1.0.0/s2cloudless-2025_3857/default/GoogleMapsCompatible/{z}/{y}/{x}.jpg"
ZOOM = 14
FEATURES = ["opt_r", "opt_g", "opt_b", "opt_brightness", "opt_exg", "opt_g_over_r", "opt_b_over_r", "opt_texture", "opt_context_brightness", "opt_context_exg"]


def tile_pixel(lon: float, lat: float, z: int = ZOOM) -> tuple[int, int, int, int]:
    """(tile x, tile y, pixel x, pixel y) of a point in the Web Mercator tile grid."""
    n = 2**z
    fx = (lon + 180.0) / 360.0 * n
    fy = (1.0 - math.asinh(math.tan(math.radians(lat))) / math.pi) / 2.0 * n
    return int(fx), int(fy), int((fx % 1) * 256), int((fy % 1) * 256)


def fetch_tile(x: int, y: int, z: int = ZOOM) -> np.ndarray:
    def go():
        r = requests.get(TILE_URL.format(z=z, x=x, y=y), timeout=30)
        r.raise_for_status()
        return r.content

    from PIL import Image

    return np.asarray(Image.open(io.BytesIO(retry(go))).convert("RGB"), dtype="float32")


def features_from_tile(img: np.ndarray, px: int, py: int) -> dict:
    """Color statistics: the point (9 x 9 pixels, about 90 m), the texture there and the wider context (65 x 65, 650 m)."""
    def window(h):
        return img[max(0, py - h): py + h + 1, max(0, px - h): px + h + 1]

    near, far = window(4), window(32)
    r, g, b = (float(near[..., i].mean()) for i in range(3))
    fr, fg, fb = (float(far[..., i].mean()) for i in range(3))
    bright = near.mean(axis=2)
    return {
        "opt_r": r, "opt_g": g, "opt_b": b, "opt_brightness": (r + g + b) / 3,
        "opt_exg": 2 * g - r - b, "opt_g_over_r": g / (r + 1.0), "opt_b_over_r": b / (r + 1.0),
        "opt_texture": float(bright.std()),
        "opt_context_brightness": (fr + fg + fb) / 3, "opt_context_exg": 2 * fg - fr - fb,
    }


def features(lon: float, lat: float) -> dict:
    """Optical features of a place (one tile request)."""
    x, y, px, py = tile_pixel(lon, lat)
    return features_from_tile(fetch_tile(x, y), px, py)
