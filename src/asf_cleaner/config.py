"""Settings. Edit the defaults below or override with CLI flags / env vars."""
import math
import os
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

# ---- USER-SETTABLE DEFAULTS -------------------------------------------------
LON = -47.9
LAT = -15.8
MAX_RESULTS = 1000  # cap on total scenes processed; a warning is logged if more exist
WINDOW_M = 500  # ground footprint (meters) sampled around the point
DEFAULT_LOOKBACK_DAYS = 180  # used when --start/--end are not given
DEFAULT_WORKERS = 8  # scenes processed in parallel (1 = sequential); keep <= ~16 to avoid throttling
# -----------------------------------------------------------------------------

# First NISAR GCOV acquisition in the ASF archive (global; a given point may start later).
NISAR_START = datetime(2025, 10, 1, tzinfo=timezone.utc)

# Bump when the cleaning logic changes: cached rows from older versions are refetched.
PROCESSING_VERSION = 2

# Pool type per dataset. GDAL (OPERA GeoTIFFs) releases the GIL, so threads work; h5py (NISAR)
# holds a global lock while reading over the network, so it needs separate processes.
EXECUTORS = {"NISAR": "process", "OPERA": "thread"}

# NASA GPM IMERG V07 daily precipitation (GES DISC): short name and CMR collection id.
# "final" is gauge-adjusted but ends 2025-09-30; "late" runs up to yesterday.
IMERG_COLLECTIONS = {
    "final": ("GPM_3IMERGDF", "C2723754864-GES_DISC"),
    "late": ("GPM_3IMERGDL", "C2723754859-GES_DISC"),
}
IMERG_LOOKBACK_DAYS = 30  # rain is fetched this many days before --start (for antecedent sums)

ASF_SEARCH_URL = "https://api.daac.asf.alaska.edu/services/search/param"
TIME_FMT = "%Y-%m-%dT%H:%M:%SZ"


@dataclass
class Settings:
    lon: float = LON
    lat: float = LAT
    start: str | None = None
    end: str | None = None
    max_results: int = MAX_RESULTS
    window_m: float = WINDOW_M
    out: str = "out"
    dataset: str = "auto"
    workers: int = DEFAULT_WORKERS
    use_cache: bool = True
    imerg: bool = True


def parse_time(s: str, end_of_day: bool = False) -> datetime:
    """Accepts 2026-06-01, 2026-06-01T00:00:00 or ...Z; always returns an aware UTC datetime."""
    dt = datetime.fromisoformat(s)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
        if end_of_day and len(s) == 10:  # bare date as an end bound = whole day
            dt += timedelta(days=1, seconds=-1)
    return dt.astimezone(timezone.utc)


def resolve_range(start, end, now=None) -> tuple[datetime, datetime]:
    now = now or datetime.now(timezone.utc)
    e = parse_time(end, True) if end else now.replace(microsecond=0)
    s = parse_time(start) if start else e - timedelta(days=DEFAULT_LOOKBACK_DAYS)
    if s >= e:
        raise ValueError(f"start ({s:%Y-%m-%d}) must be before end ({e:%Y-%m-%d})")
    return s, e


def resolve_dataset(requested: str, start: datetime, end: datetime) -> tuple[str, str]:
    """Return (dataset, reason). NISAR only when the whole range is inside NISAR's archive."""
    if requested != "auto":
        note = ""
        if requested == "NISAR" and start < NISAR_START:
            note = f" (NISAR data begins {NISAR_START:%Y-%m-%d}; earlier dates return nothing)"
        return requested, "chosen by --dataset" + note
    if start >= NISAR_START:
        return "NISAR", f"range is within NISAR coverage (from {NISAR_START:%Y-%m-%d})"
    return "OPERA", f"range starts before NISAR coverage ({NISAR_START:%Y-%m-%d}); using the historical series"


# The API snaps requested points to a grid of SNAP_M metres before searching, processing and
# caching, so clicks near each other share one cache file. The job runs on the snapped point, so
# cached rows always match their key. The point moves by at most half a step (50 m by default); the
# step is capped at window_m / 5 so a small window is never shifted by a large share of itself.
# 0 turns snapping off. The CLI uses the exact point you type.
SNAP_M = float(os.environ.get("ASF_SNAP_M", "100"))
_M_PER_DEG = 111_320.0


def snap_point(lon: float, lat: float, window_m: float = 500, step_m: float | None = None) -> tuple[float, float]:
    step = min(SNAP_M if step_m is None else step_m, window_m / 5)
    if step <= 0:
        return lon, lat
    dlat = step / _M_PER_DEG
    lat_s = min(max(round(round(lat / dlat) * dlat, 6), -90.0), 90.0)
    # a degree of longitude shrinks with latitude; use the snapped latitude so the key is stable
    dlon = step / (_M_PER_DEG * max(math.cos(math.radians(lat_s)), 0.01))
    lon_s = min(max(round(round(lon / dlon) * dlon, 6), -180.0), 180.0)
    return lon_s + 0.0, lat_s + 0.0  # + 0.0 turns -0.0 into 0.0 (one cache key)


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
