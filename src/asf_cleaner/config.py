"""Settings. Edit the defaults below or override with CLI flags / env vars."""
import os
from dataclasses import dataclass

# ---- USER-SETTABLE DEFAULTS -------------------------------------------------
LON = -47.9
LAT = -15.8
MAX_RESULTS = 50
WINDOW_M = 500  # ground footprint (meters) sampled around the point, for every sensor
# default date ranges per dataset (override with --start/--end)
DATE_RANGES = {
    "NISAR": ("2026-06-01T00:00:00Z", "2026-11-13T23:59:59Z"),
    "OPERA": ("2018-01-01T00:00:00Z", "2018-02-28T23:59:59Z"),
}
# -----------------------------------------------------------------------------

ASF_SEARCH_URL = "https://api.daac.asf.alaska.edu/services/search/param"


@dataclass
class Settings:
    lon: float = LON
    lat: float = LAT
    start: str | None = None
    end: str | None = None
    max_results: int = MAX_RESULTS
    window_m: float = WINDOW_M
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
