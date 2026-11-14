"""Settings. Edit the defaults below or override with CLI flags / env vars."""
import os
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

# ---- USER-SETTABLE DEFAULTS -------------------------------------------------
LON = -47.9
LAT = -15.8
MAX_RESULTS = 1000  # cap on total scenes processed; a warning is logged if more exist
WINDOW_M = 500  # ground footprint (meters) sampled around the point
DEFAULT_LOOKBACK_DAYS = 180  # used when --start/--end are not given
# -----------------------------------------------------------------------------

# First NISAR GCOV acquisition in the ASF archive (global; a given point may start later).
NISAR_START = datetime(2025, 10, 1, tzinfo=timezone.utc)

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
