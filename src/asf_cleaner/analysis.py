"""Analysis of `asf-clean` output, using Polars lazily.

Parquet/CSV files are scanned lazily, so filters on a selected series are pushed down and only
that series is loaded. Every function works on ONE independent series (same burst, track,
direction and polarization); never mix them.
"""
import argparse
from dataclasses import dataclass
from pathlib import Path

import polars as pl
from scipy.stats import kendalltau, theilslopes

SERIES_KEYS = ["sensor", "burst_id", "track", "direction", "polarization"]
VALUE = "backscatter_db_median"
# columns asf-clean adds from IMERG (mm before each acquisition); absent in older files
RAIN_COLS = ["rain_d0_mm", "rain_1d_mm", "rain_3d_mm", "rain_7d_mm", "rain_30d_mm"]
DEFAULT_RAIN_MM = 10.0
YEAR_S = 365.25 * 86400


# ---- loading / selecting ----------------------------------------------------------------
def scan(path) -> pl.LazyFrame:
    """Lazily scan a Parquet (preferred) or CSV file written by asf-clean."""
    path = Path(path)
    if path.suffix == ".parquet":
        return pl.scan_parquet(path)
    return pl.scan_csv(path, schema_overrides={"burst_id": pl.String}).with_columns(
        pl.col("start_time").str.to_datetime(time_zone="UTC"),
        pl.col("burst_id").fill_null(""),
    )


def list_series(lf: pl.LazyFrame) -> pl.DataFrame:
    """One row per independent series, so you can pick which one to analyse."""
    return (
        lf.group_by(SERIES_KEYS)
        .agg(
            n=pl.len(),
            first=pl.col("start_time").min(),
            last=pl.col("start_time").max(),
            median_db=pl.col(VALUE).median(),
        )
        .sort(SERIES_KEYS)
        .collect()
    )


def rain_cols(lf: pl.LazyFrame) -> list[str]:
    """The IMERG rain columns present in this file (none for files made before IMERG support)."""
    have = lf.collect_schema().names()
    return [c for c in RAIN_COLS if c in have]


def imerg_sibling(path) -> Path | None:
    """The daily IMERG file written next to an asf-clean output (imerg_<lon>_<lat>.*), if any."""
    path = Path(path)
    if "_" not in path.name:
        return None
    rest = path.name.split("_", 1)[1]
    stem = rest.rsplit(".", 1)[0]
    for suffix in (path.suffix, ".parquet", ".csv"):
        cand = path.with_name(f"imerg_{stem}{suffix}")
        if cand.exists():
            return cand
    return None


def load_daily_rain(path, start, end) -> pl.DataFrame:
    """Daily IMERG rows (date, precipitation_mm_day) between two datetimes, loaded lazily."""
    lf = scan_any(path)
    if lf.collect_schema()["date"] == pl.String:
        lf = lf.with_columns(pl.col("date").str.to_date())
    return (
        lf.filter(pl.col("date").is_between(start.date(), end.date()))
        .select("date", "precipitation_mm_day")
        .sort("date")
        .collect()
    )


def scan_any(path) -> pl.LazyFrame:
    path = Path(path)
    return pl.scan_parquet(path) if path.suffix == ".parquet" else pl.scan_csv(path)


def _series_filter(track, direction, min_valid, burst_id=None):
    cond = (
        (pl.col("track") == track)
        & (pl.col("direction") == direction.upper())
        & (pl.col("valid_fraction") >= min_valid)
    )
    return cond if burst_id is None else cond & (pl.col("burst_id") == burst_id)


def select_series(lf, *, track, direction, polarization, burst_id=None, min_valid=0.8) -> pl.DataFrame:
    """Load a single series, sorted by time. Raises if it still mixes several bursts."""
    df = (
        lf.filter(_series_filter(track, direction, min_valid, burst_id))
        .filter(pl.col("polarization") == polarization.upper())
        .select("start_time", "burst_id", "platform", VALUE, "valid_fraction", *rain_cols(lf))
        .sort("start_time")
        .collect()
    )
    bursts = df["burst_id"].unique().to_list()
    if len(bursts) > 1:
        raise ValueError(f"selection mixes several bursts {bursts}; pass burst_id to pick one")
    return df


def cross_pol_ratio(lf, *, track, direction, burst_id=None, min_valid=0.8) -> pl.DataFrame:
    """cross-pol minus co-pol (dB) per date (HV-HH or VH-VV). Higher = relatively more volume
    scattering (vegetation); lower = more surface scattering."""
    return (
        lf.filter(_series_filter(track, direction, min_valid, burst_id))
        .group_by("burst_id", "start_time")
        .agg(
            co=pl.col(VALUE).filter(pl.col("pol_class") == "co").first(),
            cross=pl.col(VALUE).filter(pl.col("pol_class") == "cross").first(),
        )
        .drop_nulls(["co", "cross"])
        .with_columns(ratio_db=pl.col("cross") - pl.col("co"))
        .sort("start_time")
        .collect()
    )


# ---- analysis ---------------------------------------------------------------------------
def span_years(df: pl.DataFrame) -> float:
    return (df["start_time"].max() - df["start_time"].min()).total_seconds() / YEAR_S


def deseasonalize(df: pl.DataFrame, min_years: float = 2.0) -> tuple[pl.DataFrame, bool]:
    """Add `anomaly_db` = value minus the typical level for that calendar month.

    Only done when the series spans at least `min_years` (otherwise the "season" would just be
    the data itself); then the anomaly is relative to the overall median and the returned flag
    is False. Months with a single observation get a null anomaly.
    """
    seasonal = span_years(df) >= min_years
    if seasonal:
        month = pl.col("start_time").dt.month()
        anomaly = (
            pl.when(pl.len().over(month) >= 2)
            .then(pl.col(VALUE) - pl.col(VALUE).median().over(month))
            .otherwise(None)
        )
    else:
        anomaly = pl.col(VALUE) - pl.col(VALUE).median()
    return df.with_columns(anomaly_db=anomaly), seasonal


@dataclass
class Trend:
    n: int
    slope_db_per_year: float
    ci_low: float
    ci_high: float
    p_value: float

    @property
    def significant(self) -> bool:
        """The 95% interval excludes zero AND Mann-Kendall p < 0.05."""
        return not (self.ci_low <= 0 <= self.ci_high) and self.p_value < 0.05


def trend(df: pl.DataFrame, min_n: int = 10) -> Trend | None:
    """Theil-Sen slope (robust) with 95% CI and Mann-Kendall p-value on the anomalies.

    The p-value assumes independent samples; consecutive scenes are autocorrelated, so treat it
    as optimistic and rely on the CI plus a look at the plot.
    """
    d = df.drop_nulls("anomaly_db")
    if d.height < min_n:
        return None
    t = ((d["start_time"] - d["start_time"].min()).dt.total_seconds() / YEAR_S).to_numpy()
    y = d["anomaly_db"].to_numpy()
    slope, _, lo, hi = theilslopes(y, t)
    _, p = kendalltau(t, y)
    return Trend(d.height, float(slope), float(lo), float(hi), float(p))


def find_gaps(df: pl.DataFrame, min_days: int = 45) -> pl.DataFrame:
    """Stretches with no scenes in the series (a normal revisit is 6 to 12 days). Often another
    polarization was acquired instead (e.g. VV-only periods), or the point had no valid data."""
    return (
        df.select("start_time")
        .with_columns(prev=pl.col("start_time").shift(1))
        .with_columns(gap_days=(pl.col("start_time") - pl.col("prev")).dt.total_days())
        .filter(pl.col("gap_days") > min_days)
        .select(gap_from="prev", gap_to="start_time", gap_days="gap_days")
    )


def _break_gaps(t: list, y: list, min_days: int = 45) -> tuple[list, list]:
    """Insert an empty point inside every long gap so a plotted line is not drawn across it."""
    out_t, out_y = t[:1], y[:1]
    for a, b, yb in zip(t, t[1:], y[1:]):
        if (b - a).days > min_days:
            out_t.append(a + (b - a) / 2)
            out_y.append(None)
        out_t.append(b)
        out_y.append(yb)
    return out_t, out_y


def add_rain_context(df: pl.DataFrame, rain_mm: float = DEFAULT_RAIN_MM) -> pl.DataFrame:
    """`rain_context`: wet / dry / unknown from IMERG rain on the acquisition day and the day
    before (>= rain_mm counts as wet). 'dry' only means IMERG saw little rain: its cells are ~11 km
    wide, so it can miss a local storm. 'unknown' = no rain columns (file made before IMERG)."""
    if "rain_d0_mm" not in df.columns or "rain_1d_mm" not in df.columns:
        return df.with_columns(rain_context=pl.lit("unknown"))
    peak = pl.max_horizontal(pl.col("rain_d0_mm").cast(pl.Float64).fill_nan(None),
                             pl.col("rain_1d_mm").cast(pl.Float64).fill_nan(None))
    return df.with_columns(
        rain_context=pl.when(peak.is_null()).then(pl.lit("unknown"))
        .when(peak >= rain_mm).then(pl.lit("wet")).otherwise(pl.lit("dry"))
    )


def _runs(df: pl.DataFrame, threshold_db: float, rain_mm: float) -> pl.DataFrame:
    """Runs of consecutive scenes beyond +/-threshold_db in the same direction, with the rain
    seen at the first scene (the onset) of each run."""
    d = add_rain_context(df.drop_nulls("anomaly_db"), rain_mm).with_columns(
        key=pl.when(pl.col("anomaly_db").abs() > threshold_db)
        .then(pl.col("anomaly_db").sign())
        .otherwise(0)
        .cast(pl.Int8)
    )
    d = d.with_columns(run=pl.col("key").rle_id())
    onset = [pl.col(c).first().alias(f"onset_{c}") for c in RAIN_COLS if c in d.columns]
    return (
        d.filter(pl.col("key") != 0)
        .group_by("run", maintain_order=True)
        .agg(
            *onset,
            start=pl.col("start_time").min(),
            end=pl.col("start_time").max(),
            n_scenes=pl.len(),
            mean_anomaly_db=pl.col("anomaly_db").mean(),
            value_db=pl.col(VALUE).mean(),
            rain_context=pl.col("rain_context").first(),
        )
        .with_columns(direction=pl.when(pl.col("mean_anomaly_db") > 0).then(pl.lit("brighter")).otherwise(pl.lit("darker")))
        .drop("run")
        .select("start", "end", "n_scenes", "direction", "mean_anomaly_db", "value_db", "rain_context",
                *[f"onset_{c}" for c in RAIN_COLS if c in d.columns])
    )


def detect_events(df, threshold_db: float = 2.5, min_run: int = 3, rain_mm: float = DEFAULT_RAIN_MM):
    """Sustained departures: >= min_run consecutive scenes beyond +/-threshold_db in the same
    direction, with the rain at their onset."""
    return _runs(df, threshold_db, rain_mm).filter(pl.col("n_scenes") >= min_run)


def detect_spikes(df, threshold_db: float = 2.5, min_run: int = 3, rain_mm: float = DEFAULT_RAIN_MM):
    """Isolated outliers: fewer than min_run consecutive scenes beyond +/-threshold_db. Not
    sustained, so not events, but worth a look: rain, a processing problem or a brief change."""
    return _runs(df, threshold_db, rain_mm).filter(pl.col("n_scenes") < min_run)


# ---- plotting ---------------------------------------------------------------------------
CONTEXT_COLOR = {"wet": "tab:blue", "dry": "tab:red", "unknown": "tab:gray"}


def plot(df, events, spikes, threshold_db, ratio, rain_daily, out, title="") -> int:
    """Write the figure; returns the number of panels (backscatter, anomaly, [cross/co], [rain])."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    per_scene = "rain_3d_mm" in df.columns and df["rain_3d_mm"].drop_nulls().len() > 0
    has_ratio = ratio is not None and ratio.height > 0
    has_rain = (rain_daily is not None and rain_daily.height > 0) or per_scene
    panels = 2 + has_ratio + has_rain
    fig, ax = plt.subplots(panels, 1, figsize=(10, 3 * panels), sharex=True)
    t = df["start_time"].to_list()
    ax[0].plot(*_break_gaps(t, df[VALUE].to_list()), "o-", ms=3)
    ax[0].set(ylabel="backscatter median (dB)", title=title)
    ax[1].plot(*_break_gaps(t, df["anomaly_db"].to_list()), "o-", ms=3)
    ax[1].axhline(0, c="k", lw=0.5)
    for lvl in (threshold_db, -threshold_db):
        ax[1].axhline(lvl, c="r", ls="--", lw=0.8)
    for e in events.iter_rows(named=True):
        ax[1].axvspan(e["start"], e["end"], color="orange", alpha=0.3)
    for ctx in ("wet", "dry", "unknown"):
        sp = spikes.filter(pl.col("rain_context") == ctx)
        if sp.height:
            ax[1].scatter(sp["start"].to_list(), sp["mean_anomaly_db"].to_list(), marker="X", s=70,
                          c=CONTEXT_COLOR[ctx], zorder=3, label=f"isolated outlier, rain: {ctx}")
    if spikes.height:
        ax[1].legend(loc="best", fontsize=8)
    ax[1].set(ylabel="anomaly (dB)")
    k = 2
    if has_ratio:
        ax[k].plot(*_break_gaps(ratio["start_time"].to_list(), ratio["ratio_db"].to_list()), "o-", ms=3)
        ax[k].set(ylabel="cross - co (dB)")
        k += 1
    if has_rain:
        if rain_daily is not None and rain_daily.height > 0:
            ax[k].bar(rain_daily["date"].to_list(), rain_daily["precipitation_mm_day"].to_list(),
                      width=1.0, color="tab:blue")
            ax[k].set(ylabel="IMERG rain (mm/day)")
        else:
            ax[k].bar(t, df["rain_3d_mm"].to_list(), width=2.0, color="tab:blue")
            ax[k].set(ylabel="rain, 3 days before (mm)")
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return panels


# ---- CLI --------------------------------------------------------------------------------
def main(argv=None):
    ap = argparse.ArgumentParser(description="Analyse one series from an asf-clean output file")
    ap.add_argument("file", help="parquet (lazy, preferred) or csv written by asf-clean")
    ap.add_argument("--track", type=int)
    ap.add_argument("--direction", help="ASCENDING or DESCENDING")
    ap.add_argument("--polarization", help="HH, HV, VV or VH")
    ap.add_argument("--burst-id", help="OPERA burst; needed only if several bursts match")
    ap.add_argument("--min-valid", type=float, default=0.8, help="min valid_fraction per scene")
    ap.add_argument("--threshold", type=float, default=2.5, help="event threshold in dB")
    ap.add_argument("--min-run", type=int, default=3, help="consecutive scenes needed for an event")
    ap.add_argument("--rain-mm", type=float, default=DEFAULT_RAIN_MM,
                    help="IMERG rain (mm, acquisition day or day before) counted as wet")
    ap.add_argument("--rain", help="daily IMERG file for the plot (default: imerg_* next to the input)")
    ap.add_argument("--plot", help="write a PNG here")
    a = ap.parse_args(argv)

    lf = scan(a.file)
    if a.track is None or not a.direction or not a.polarization:
        with pl.Config(tbl_rows=200, tbl_cols=20, tbl_width_chars=160):
            print("Available series (pass --track --direction --polarization to analyse one):\n")
            print(list_series(lf))
        return

    df = select_series(lf, track=a.track, direction=a.direction, polarization=a.polarization,
                       burst_id=a.burst_id, min_valid=a.min_valid)
    if df.height == 0:
        raise SystemExit("no scenes match this selection (check the series list and --min-valid)")
    df, seasonal = deseasonalize(df)

    print(f"Series: track {a.track} {a.direction.upper()} {a.polarization.upper()}  "
          f"({df.height} scenes, {df['start_time'].min():%Y-%m-%d} .. {df['start_time'].max():%Y-%m-%d})")
    print(f"Median {df[VALUE].median():.2f} dB, std {df[VALUE].std():.2f} dB")
    if not seasonal:
        print("WARNING: series spans < 2 years, so seasonality was NOT removed. Anomalies are "
              "relative to the overall median and trends/events are unreliable.")

    tr = trend(df) if seasonal else None
    if not seasonal:
        print("Trend: skipped (needs a series spanning at least 2 years).")
    elif tr is None:
        print("Trend: not enough scenes (need >= 10).")
    else:
        verdict = "significant" if tr.significant else "NOT significant"
        print(f"Trend: {tr.slope_db_per_year:+.2f} dB/year (95% CI {tr.ci_low:+.2f} to "
              f"{tr.ci_high:+.2f}, Mann-Kendall p={tr.p_value:.3f}) -> {verdict}")

    gaps = find_gaps(df)
    if gaps.height:
        print(f"\nGaps longer than 45 days: {gaps.height} (no scenes in the series; another polarization "
              "may exist for those dates, or the point had no valid data)")
        with pl.Config(tbl_rows=20, tbl_width_chars=160):
            print(gaps)

    rcols = rain_cols(lf)
    if not rcols:
        print("\nRain: this file has no rain columns (made before IMERG support); rerun asf-clean "
              "to add them. Events below are reported without rain context.")

    ev = detect_events(df, a.threshold, a.min_run, a.rain_mm)
    spikes = detect_spikes(df, a.threshold, a.min_run, a.rain_mm)
    with pl.Config(tbl_rows=50, tbl_cols=20, tbl_width_chars=200):
        print(f"\nSustained events (>= {a.min_run} consecutive scenes beyond +/-{a.threshold} dB): {ev.height}")
        if ev.height:
            print(ev)
        print(f"\nIsolated outliers (< {a.min_run} scenes beyond +/-{a.threshold} dB, not sustained): {spikes.height}")
        if spikes.height:
            print(spikes)
    if rcols and (ev.height or spikes.height):
        print(f"(rain_context: wet = IMERG >= {a.rain_mm:g} mm on the acquisition day or the day before; "
              "dry only means IMERG saw little rain, its ~11 km cells can miss a local storm. "
              "onset_* = rain at the first scene of the run.)")

    ratio = cross_pol_ratio(lf, track=a.track, direction=a.direction, burst_id=a.burst_id,
                            min_valid=a.min_valid)
    if ratio.height:
        print(f"\nCross-pol minus co-pol: mean {ratio['ratio_db'].mean():.2f} dB, "
              f"std {ratio['ratio_db'].std():.2f} dB over {ratio.height} dates")
    if a.plot:
        rain_file = a.rain or imerg_sibling(a.file)
        rain_daily = None
        if rain_file:
            rain_daily = load_daily_rain(rain_file, df["start_time"].min(), df["start_time"].max())
        panels = plot(df, ev, spikes, a.threshold, ratio, rain_daily, a.plot,
                      f"track {a.track} {a.direction.upper()} {a.polarization.upper()}")
        rain_note = "with daily IMERG rain" if rain_daily is not None and rain_daily.height else (
            "with per-scene rain" if rcols else "without rain (no IMERG data found)")
        print(f"\nplot written to {a.plot} ({panels} panels, {rain_note})")
    print("\nNote: an anomaly says something changed, not what. Rain context helps rule rain in or "
          "out, but confirm with optical imagery or known event dates.")


if __name__ == "__main__":
    main()
