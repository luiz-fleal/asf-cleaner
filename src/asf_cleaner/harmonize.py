"""Common output schema shared by the NISAR (L-band) and OPERA/Sentinel-1 (C-band) modes."""
import numpy as np
import pandas as pd

BAND = {"NISAR": "L", "OPERA": "C"}
POL_CLASS = {"HH": "co", "VV": "co", "HV": "cross", "VH": "cross"}
GROUP = ["sensor", "burst_id", "track", "direction", "polarization"]

COLUMNS = [
    "sensor", "band", "platform", "scene", "burst_id", "track", "direction", "start_time",
    "polarization", "pol_class", "valid_fraction", "backscatter_db_mean",
    "backscatter_db_median", "backscatter_db_center", "window_m", "window_px",
    "pixel_spacing_m", "normalization", "z_median",
]


def harmonize(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["band"] = df["sensor"].map(BAND)
    df["pol_class"] = df["polarization"].map(POL_CLASS)
    df["start_time"] = pd.to_datetime(df["start_time"], utc=True)
    df["burst_id"] = df["burst_id"].fillna("")  # keep NaN-free for groupby
    # one row per acquisition/burst/pol (e.g. reprocessed duplicates of the same scene)
    df = df.drop_duplicates(["sensor", "burst_id", "track", "start_time", "polarization"],
                            keep="last")

    # z-score within each independent series (same sensor/track/direction/pol/burst):
    # puts series with different absolute levels (tracks, polarizations) on one scale for plotting.
    g = df.groupby(GROUP)["backscatter_db_median"]
    std = g.transform("std")
    n = g.transform("count")
    df["z_median"] = np.where(
        (n >= 3) & (std > 0), (df["backscatter_db_median"] - g.transform("mean")) / std, np.nan
    )
    return df.sort_values(["sensor", "start_time"])[COLUMNS].reset_index(drop=True)
