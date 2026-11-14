"""Settings. Edit the defaults below or override with CLI flags / env vars."""
import os
from dataclasses import dataclass

# ---- USER-SETTABLE DEFAULTS -------------------------------------------------
LON = -47.9
LAT = -15.8
START = "2026-06-01T00:00:00Z"
END = "2026-11-13T23:59:59Z"
MAX_RESULTS = 50
WINDOW_PX = 11  # odd window (pixels) read around the point
# -----------------------------------------------------------------------------

ASF_SEARCH_URL = "https://api.daac.asf.alaska.edu/services/search/param"


@dataclass
class Settings:
    lon: float = LON
    lat: float = LAT
    start: str = START
    end: str = END
    max_results: int = MAX_RESULTS
    window: int = WINDOW_PX
    out: str = "out"
    dataset: str = "NISAR"


def get_token() -> str:
    token = os.environ.get("EARTHDATA_TOKEN")
    if not token:
        try:  # optional .env file in the current directory
            for line in open(".env"):
                if line.startswith("EARTHDATA_TOKEN="):
                    token = line.split("=", 1)[1].strip().strip("'\"")
        except OSError:
            pass
    if not token:
        raise RuntimeError(
            "EARTHDATA_TOKEN not set. Generate a token at "
            "https://urs.earthdata.nasa.gov (Generate Token) and export it."
        )
    return token
