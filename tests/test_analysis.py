from datetime import datetime, timedelta, timezone

import numpy as np
import polars as pl
import pytest

from asf_cleaner import analysis as A

UTC = timezone.utc


def make_frame(years=4, slope=0.5, event=True, burst="", seed=0):
    rng = np.random.default_rng(seed)
    n = int(years * 365.25 / 12)
    t0 = datetime(2019, 1, 1, tzinfo=UTC)
    times = [t0 + timedelta(days=12 * i) for i in range(n)]
    yrs = np.array([(t - t0).days / 365.25 for t in times])
    hv = -16 + 3 * np.sin(2 * np.pi * yrs) + slope * yrs + rng.normal(0, 0.3, n)
    if event:
        hv[70:74] += 5.0  # sustained brightening, 4 scenes
    rows = []
    for pol, pc, off in (("HV", "cross", 0.0), ("HH", "co", 6.0)):
        for t, v in zip(times, hv + off):
            rows.append({"sensor": "OPERA", "burst_id": burst, "track": 24, "direction": "DESCENDING",
                         "platform": "S1A", "start_time": t, "polarization": pol, "pol_class": pc,
                         "valid_fraction": 1.0, "backscatter_db_median": float(v)})
    return pl.DataFrame(rows)


@pytest.fixture
def parquet(tmp_path):
    p = tmp_path / "x.parquet"
    make_frame().write_parquet(p)
    return p


def test_lazy_scan_and_list(parquet):
    lf = A.scan(parquet)
    assert isinstance(lf, pl.LazyFrame)
    s = A.list_series(lf)
    assert s.height == 2 and set(s["polarization"]) == {"HH", "HV"}


def test_csv_roundtrip(tmp_path):
    p = tmp_path / "x.csv"
    make_frame().to_pandas().to_csv(p, index=False)
    df = A.select_series(A.scan(p), track=24, direction="descending", polarization="hv")
    assert df.height > 100 and df["start_time"].dtype == pl.Datetime("us", "UTC")


def test_trend_recovered_after_deseasonalizing(parquet):
    df = A.select_series(A.scan(parquet), track=24, direction="DESCENDING", polarization="HV")
    df, seasonal = A.deseasonalize(df)
    tr = A.trend(df)
    assert seasonal and tr.significant
    assert abs(tr.slope_db_per_year - 0.5) < 0.2 and tr.ci_low < 0.5 < tr.ci_high


def test_no_trend_in_flat_series(tmp_path):
    p = tmp_path / "f.parquet"
    make_frame(slope=0.0, event=False, seed=3).write_parquet(p)
    df, _ = A.deseasonalize(A.select_series(A.scan(p), track=24, direction="DESCENDING", polarization="HV"))
    assert not A.trend(df).significant
    assert A.detect_events(df).height == 0


def test_event_detected(parquet):
    df, _ = A.deseasonalize(A.select_series(A.scan(parquet), track=24, direction="DESCENDING", polarization="HV"))
    ev = A.detect_events(df, threshold_db=2.5, min_run=3)
    assert ev.height == 1 and ev["direction"][0] == "brighter" and ev["n_scenes"][0] >= 3


def test_single_spike_ignored():
    df = make_frame(event=False).filter(pl.col("polarization") == "HV")
    df = df.with_columns(pl.when(pl.int_range(pl.len()) == 60).then(pl.col("backscatter_db_median") + 8)
                         .otherwise(pl.col("backscatter_db_median")).alias("backscatter_db_median"))
    df, _ = A.deseasonalize(df)
    assert A.detect_events(df).height == 0


def test_short_series_skips_seasonality():
    df = make_frame(years=1).filter(pl.col("polarization") == "HV")
    _, seasonal = A.deseasonalize(df)
    assert seasonal is False


def test_multiple_bursts_rejected(tmp_path):
    p = tmp_path / "b.parquet"
    pl.concat([make_frame(burst="A"), make_frame(burst="B")]).write_parquet(p)
    with pytest.raises(ValueError, match="burst"):
        A.select_series(A.scan(p), track=24, direction="DESCENDING", polarization="HV")
    assert A.select_series(A.scan(p), track=24, direction="DESCENDING", polarization="HV", burst_id="A").height > 0


def test_cross_pol_ratio(parquet):
    r = A.cross_pol_ratio(A.scan(parquet), track=24, direction="DESCENDING")
    assert np.allclose(r["ratio_db"].to_numpy(), -6.0)


def test_cli_list_and_analyze(parquet, tmp_path, capsys):
    A.main([str(parquet)])
    assert "Available series" in capsys.readouterr().out
    png = tmp_path / "p.png"
    A.main([str(parquet), "--track", "24", "--direction", "DESCENDING", "--polarization", "HV", "--plot", str(png)])
    out = capsys.readouterr().out
    assert "Trend:" in out and "significant" in out and png.stat().st_size > 1000
