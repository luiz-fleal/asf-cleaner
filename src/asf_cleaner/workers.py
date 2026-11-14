"""Per-scene worker functions.

Kept top-level and import-light so they can run in spawned processes. Each returns the cleaned
rows for one scene and raises on failure (the runner isolates failures).
"""
from . import nisar, opera
from .config import PROCESSING_VERSION
from .runner import retry
from .stream import open_remote_h5

META = ("scene", "burst_id", "track", "direction", "start_time", "platform")


def _meta(sensor: str, sc: dict) -> dict:
    return {"sensor": sensor, **{k: sc[k] for k in META}, "processing_version": PROCESSING_VERSION}


def process_nisar(sc, *, lon, lat, window_m, token):
    def go():
        with open_remote_h5(sc["url"], token) as h5:
            return nisar.clean_point(h5, lon, lat, window_m)

    return [{**_meta("NISAR", sc), **r} for r in retry(go)]


def process_opera(sc, *, lon, lat, window_m, token):
    rows = retry(lambda: opera.clean_point(sc["tifs"], token, lon, lat, window_m))
    return [{**_meta("OPERA", sc), **r} for r in rows]

