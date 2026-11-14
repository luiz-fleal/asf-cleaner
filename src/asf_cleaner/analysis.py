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
        .select("start_time", "burst_id", "platform", VALUE, "valid_fraction")
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


def detect_events(df: pl.DataFrame, threshold_db: float = 2.5, min_run: int = 3) -> pl.DataFrame:
    """Sustained departures: >= min_run consecutive scenes beyond +/-threshold_db in the same
    direction. A single spike (rain, speckle) is deliberately not reported."""
    d = df.drop_nulls("anomaly_db").with_columns(
        key=pl.when(pl.col("anomaly_db").abs() > threshold_db)
        .then(pl.col("anomaly_db").sign())
        .otherwise(0)
        .cast(pl.Int8)
    )
    d = d.with_columns(run=pl.col("key").rle_id())
    return (
        d.filter(pl.col("key") != 0)
        .group_by("run", maintain_order=True)
        .agg(
            start=pl.col("start_time").min(),
            end=pl.col("start_time").max(),
            n_scenes=pl.len(),
            mean_anomaly_db=pl.col("anomaly_db").mean(),
        )
        .filter(pl.col("n_scenes") >= min_run)
        .with_columns(direction=pl.when(pl.col("mean_anomaly_db") > 0).then(pl.lit("brighter")).otherwise(pl.lit("darker")))
        .drop("run")
    )


# ---- plotting ---------------------------------------------------------------------------
def plot(df, events, threshold_db, ratio, out, title=""):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    panels = 3 if ratio is not None and ratio.height else 2
    fig, ax = plt.subplots(panels, 1, figsize=(10, 3 * panels), sharex=True)
    t = df["start_time"].to_list()
    ax[0].plot(t, df[VALUE].to_list(), "o-", ms=3)
    ax[0].set(ylabel="backscatter median (dB)", title=title)
    a = df["anomaly_db"].to_list()
    ax[1].plot(t, a, "o-", ms=3)
    ax[1].axhline(0, c="k", lw=0.5)
    for s in (threshold_db, -threshold_db):
        ax[1].axhline(s, c="r", ls="--", lw=0.8)
    for e in events.iter_rows(named=True):
        ax[1].axvspan(e["start"], e["end"], color="orange", alpha=0.3)
    ax[1].set(ylabel="anomaly (dB)")
    if panels == 3:
        ax[2].plot(ratio["start_time"].to_list(), ratio["ratio_db"].to_list(), "o-", ms=3)
        ax[2].set(ylabel="cross - co (dB)")
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


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

    ev = detect_events(df, a.threshold, a.min_run)
    print(f"\nSustained events (>= {a.min_run} consecutive scenes beyond +/-{a.threshold} dB): {ev.height}")
    if ev.height:
        print(ev)

    ratio = cross_pol_ratio(lf, track=a.track, direction=a.direction, burst_id=a.burst_id,
                            min_valid=a.min_valid)
    if ratio.height:
        print(f"\nCross-pol minus co-pol: mean {ratio['ratio_db'].mean():.2f} dB, "
              f"std {ratio['ratio_db'].std():.2f} dB over {ratio.height} dates")
    if a.plot:
        plot(df, ev, a.threshold, ratio, a.plot,
             f"track {a.track} {a.direction.upper()} {a.polarization.upper()}")
        print(f"\nplot written to {a.plot}")
    print("\nNote: an anomaly says something changed, not what. Confirm with rainfall, optical "
          "imagery or known event dates.")


if __name__ == "__main__":
    main()
