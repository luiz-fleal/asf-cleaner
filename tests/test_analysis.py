from datetime import date, datetime, timedelta, timezone

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


# ---- rain (IMERG) in the analysis ----------------------------------------------------------
def make_rain_frame(**kw):
    """4-year HV/HH series: +5 dB event at scenes 70-73 (no rain), one -8 dB dip at scene 20 (30 mm)."""
    idx = pl.int_range(pl.len()).over("polarization")
    df = make_frame(**kw)
    return df.with_columns(
        backscatter_db_median=pl.when(idx == 20).then(pl.col("backscatter_db_median") - 8)
        .otherwise(pl.col("backscatter_db_median")),
        rain_d0_mm=pl.when(idx == 20).then(30.0).otherwise(0.0),
        rain_1d_mm=pl.when(idx == 20).then(5.0).otherwise(0.0),
        rain_3d_mm=pl.when(idx == 20).then(35.0).otherwise(0.0),
        rain_7d_mm=pl.lit(0.0),
        rain_30d_mm=pl.lit(0.0),
    )


def series_of(path):
    df = A.select_series(A.scan(path), track=24, direction="DESCENDING", polarization="HV")
    return A.deseasonalize(df)[0]


@pytest.fixture
def rain_file(tmp_path):
    p = tmp_path / "opera_-60.0_-1.0.parquet"
    make_rain_frame().write_parquet(p)
    return p


def test_rain_columns_are_loaded_with_the_series(rain_file, tmp_path):
    df = A.select_series(A.scan(rain_file), track=24, direction="DESCENDING", polarization="HV")
    assert set(A.RAIN_COLS) <= set(df.columns)
    old = tmp_path / "old.parquet"
    make_frame().write_parquet(old)  # file made before IMERG support
    assert not set(A.RAIN_COLS) & set(A.select_series(A.scan(old), track=24, direction="DESCENDING", polarization="HV").columns)


def test_isolated_outlier_is_reported_with_its_rain(rain_file):
    df = series_of(rain_file)
    sp = A.detect_spikes(df)
    assert sp.height == 1 and sp["n_scenes"][0] == 1 and sp["direction"][0] == "darker"
    assert sp["rain_context"][0] == "wet" and sp["onset_rain_d0_mm"][0] == 30.0
    assert sp["mean_anomaly_db"][0] < -2.5
    ev = A.detect_events(df)  # the sustained event is separate, and the spike is not an event
    assert ev.height == 1 and ev["direction"][0] == "brighter"
    assert ev["rain_context"][0] == "dry" and ev["onset_rain_d0_mm"][0] == 0.0


def test_rain_threshold_changes_the_context(rain_file):
    df = series_of(rain_file)
    assert A.detect_spikes(df, rain_mm=25)["rain_context"][0] == "wet"
    assert A.detect_spikes(df, rain_mm=40)["rain_context"][0] == "dry"  # peak was 30 mm


def test_no_rain_columns_gives_unknown_not_an_error(tmp_path):
    p = tmp_path / "old.parquet"
    make_frame().write_parquet(p)
    df = series_of(p)
    ev = A.detect_events(df)
    assert ev.height == 1 and ev["rain_context"][0] == "unknown"
    assert not [c for c in ev.columns if c.startswith("onset_")]


def test_missing_rain_is_unknown(rain_file):
    df = series_of(rain_file).with_columns(rain_d0_mm=pl.lit(None, dtype=pl.Float64),
                                           rain_1d_mm=pl.lit(float("nan")))
    assert A.detect_spikes(df)["rain_context"][0] == "unknown"


def test_imerg_sibling_and_daily_loading(tmp_path):
    main_file = tmp_path / "opera_-60.0_-1.0.parquet"
    assert A.imerg_sibling(main_file) is None
    d = pl.DataFrame({"date": ["2026-01-01", "2026-01-02", "2026-01-03"],
                                 "precipitation_mm_day": [1.0, 2.0, 3.0]})
    d.write_csv(tmp_path / "imerg_-60.0_-1.0.csv")
    assert A.imerg_sibling(main_file).name == "imerg_-60.0_-1.0.csv"
    d.write_parquet(tmp_path / "imerg_-60.0_-1.0.parquet")
    assert A.imerg_sibling(main_file).name == "imerg_-60.0_-1.0.parquet"
    for f in ("imerg_-60.0_-1.0.csv", "imerg_-60.0_-1.0.parquet"):
        got = A.load_daily_rain(tmp_path / f, datetime(2026, 1, 2, tzinfo=UTC), datetime(2026, 1, 3, tzinfo=UTC))
        assert got["precipitation_mm_day"].to_list() == [2.0, 3.0] and got["date"].dtype == pl.Date


def test_plot_gets_a_rain_panel(rain_file, tmp_path):
    df = series_of(rain_file)
    ev, sp = A.detect_events(df), A.detect_spikes(df)
    ratio = A.cross_pol_ratio(A.scan(rain_file), track=24, direction="DESCENDING")
    daily = pl.DataFrame({"date": [date(2022, 1, 1), date(2022, 1, 2)], "precipitation_mm_day": [4.0, 0.0]})
    png = tmp_path / "p.png"
    assert A.plot(df, ev, sp, 2.5, ratio, daily, png) == 4          # backscatter, anomaly, ratio, daily rain
    assert A.plot(df, ev, sp, 2.5, ratio, None, png) == 4           # falls back to per-scene rain
    old = tmp_path / "old.parquet"
    make_frame().write_parquet(old)
    d2 = series_of(old)
    assert A.plot(d2, A.detect_events(d2), A.detect_spikes(d2), 2.5, ratio, None, png) == 3  # no rain at all
    assert png.stat().st_size > 1000


def test_cli_reports_rain_and_plots_it(rain_file, tmp_path, capsys):
    pl.DataFrame({"date": ["2022-01-01", "2022-01-02"], "precipitation_mm_day": [3.0, 5.0]}).write_parquet(
        tmp_path / "imerg_-60.0_-1.0.parquet")  # picked up automatically next to the input
    png = tmp_path / "p.png"
    A.main([str(rain_file), "--track", "24", "--direction", "DESCENDING", "--polarization", "HV", "--plot", str(png)])
    out = capsys.readouterr().out
    assert "Isolated outliers" in out and "wet" in out and "onset_rain_d0_mm" in out
    assert "4 panels, with daily IMERG rain" in out and png.stat().st_size > 1000


def test_cli_warns_when_file_has_no_rain(tmp_path, capsys):
    p = tmp_path / "old.parquet"
    make_frame().write_parquet(p)
    A.main([str(p), "--track", "24", "--direction", "DESCENDING", "--polarization", "HV"])
    out = capsys.readouterr().out
    assert "no rain columns" in out and "unknown" in out


# ---- gaps ----------------------------------------------------------------------------------
def gappy(tmp_path):
    p = tmp_path / "g.parquet"
    df = make_frame(event=False)
    keep = ~((pl.col("start_time") > datetime(2020, 3, 1, tzinfo=UTC)) & (pl.col("start_time") < datetime(2020, 9, 1, tzinfo=UTC)))
    df.filter(keep).write_parquet(p)
    return p


def test_find_gaps(tmp_path):
    df = A.select_series(A.scan(gappy(tmp_path)), track=24, direction="DESCENDING", polarization="HV")
    gaps = A.find_gaps(df)
    assert gaps.height == 1 and 150 < gaps["gap_days"][0] < 200
    assert A.find_gaps(df, min_days=400).height == 0


def test_lines_are_broken_across_gaps():
    t = [datetime(2020, 1, 1, tzinfo=UTC) + timedelta(days=d) for d in (0, 12, 24, 300, 312)]
    tt, yy = A._break_gaps(t, [1.0, 2.0, 3.0, 4.0, 5.0])
    assert len(tt) == 6 and yy == [1.0, 2.0, 3.0, None, 4.0, 5.0] and tt[3] > t[2] and tt[3] < t[3]
    assert A._break_gaps(t[:1], [1.0]) == (t[:1], [1.0])


def test_cli_reports_gaps(tmp_path, capsys):
    A.main([str(gappy(tmp_path)), "--track", "24", "--direction", "DESCENDING", "--polarization", "HV"])
    assert "Gaps longer than 45 days: 1" in capsys.readouterr().out
