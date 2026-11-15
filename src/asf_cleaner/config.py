"""Settings. Edit the defaults below or override with CLI flags / env vars."""
import os
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

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
# First OPERA RTC-S1 acquisition we use (Sentinel-1 data in the OPERA archive starts in 2016).
OPERA_START = datetime(2016, 1, 1, tzinfo=timezone.utc)

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
    "monthly": ("GPM_3IMERGM", "C2723754851-GES_DISC"),  # final run only, ends 2025-09-30
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


# Where the token may live besides the environment: a .env file in the folder the server was started from, or in the
# project folder, so it is found wherever `asf-api` is launched from.
PROJECT_DIR = Path(__file__).resolve().parents[2]
ENV_FILES = (Path(".env"), PROJECT_DIR / ".env")


def env_value(name: str) -> str | None:
    """A setting from the environment, else from a `.env` file (read on every call, so a value added while the
    server runs is picked up without a restart)."""
    if os.environ.get(name):
        return os.environ[name]
    for path in ENV_FILES:
        try:
            lines = path.read_text().splitlines()
        except OSError:
            continue
        for line in lines:
            line = line.strip().removeprefix("export ").strip()
            if line.startswith(name + "="):
                value = line.split("=", 1)[1].strip().strip("'\"")
                if value:
                    return value
    return None


def _token_from_env_files() -> str | None:
    return env_value("EARTHDATA_TOKEN")


def get_token() -> str:
    """The Earthdata token: the EARTHDATA_TOKEN variable, else a `.env` file (read on every call, so a token added
    while the server runs is picked up without a restart)."""
    token = os.environ.get("EARTHDATA_TOKEN") or _token_from_env_files()
    if not token:
        raise RuntimeError(
            "EARTHDATA_TOKEN not set. Generate a token at https://urs.earthdata.nasa.gov (Generate Token) and put it "
            f"in {PROJECT_DIR / '.env'} as EARTHDATA_TOKEN=..., or export it."
        )
    return token
