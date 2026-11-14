"""Common output schema shared by the NISAR (L-band) and OPERA/Sentinel-1 (C-band) modes."""
import pandas as pd

BAND = {"NISAR": "L", "OPERA": "C"}
POL_CLASS = {"HH": "co", "VV": "co", "HV": "cross", "VH": "cross"}

COLUMNS = [
    "sensor", "band", "platform", "scene", "burst_id", "track", "direction", "start_time",
    "polarization", "pol_class", "valid_fraction", "backscatter_db_mean",
    "backscatter_db_median", "backscatter_db_center", "window_m", "window_px",
    "pixel_spacing_m", "normalization", "processing_version",
]


def harmonize(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["band"] = df["sensor"].map(BAND)
    df["pol_class"] = df["polarization"].map(POL_CLASS)
    df["start_time"] = pd.to_datetime(df["start_time"], utc=True)
    df["burst_id"] = df["burst_id"].fillna("")
    # one row per acquisition/burst/pol (e.g. reprocessed duplicates of the same scene)
    df = df.drop_duplicates(["sensor", "burst_id", "track", "start_time", "polarization"],
                            keep="last")
    return df.sort_values(["sensor", "start_time"])[COLUMNS].reset_index(drop=True)

