"""IMERG rain is retrieved with any SAR mode and never costs the SAR data."""
import numpy as np
import pandas as pd
import pytest

from asf_cleaner import cli, imerg


def scene(n):
    return {"scene": f"s{n}", "burst_id": None, "track": 68, "direction": "DESCENDING",
            "start_time": f"2026-07-{n:02d}T22:24:00Z", "platform": "NISAR"}


def sar_rows(sc, sensor):
    return [{"sensor": sensor, **{k: sc[k] for k in ("scene", "burst_id", "track", "direction", "start_time", "platform")},
             "polarization": p, "valid_fraction": 1.0, "backscatter_db_mean": -10.0,
             "backscatter_db_median": -10.0, "backscatter_db_center": -10.0, "window_m": 500,
             "window_px": "3x3", "pixel_spacing_m": 20.0, "normalization": "gamma0",
             "processing_version": 2} for p in ("HH", "HV")]


def rain_df():
    d = pd.date_range("2026-05-01", "2026-08-01")
    return pd.DataFrame({"date": d.strftime("%Y-%m-%d"), "precipitation_mm_day": 2.0, "imerg_run": "late"})


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setenv("EARTHDATA_TOKEN", "x")
    st = {"searched": [], "rain_calls": 0, "rain": rain_df}

    def search(kind, *a, **k):
        st["searched"].append(kind)
        return [scene(1), scene(2)]

    monkeypatch.setattr(cli, "search", search)
    monkeypatch.setitem(cli.WORKERS, "NISAR", lambda sc, **kw: sar_rows(sc, "NISAR"))
    monkeypatch.setitem(cli.WORKERS, "OPERA", lambda sc, **kw: sar_rows(sc, "OPERA"))

    def fetch_daily(*a, **k):
        st["rain_calls"] += 1
        r = st["rain"]
        if isinstance(r, Exception):
            raise r
        return r()

    monkeypatch.setattr(cli.imerg, "fetch_daily", fetch_daily)
    st["run"] = lambda *extra: cli.main(["--lon", "-60", "--lat", "-1", "--start", "2026-06-01",
                                         "--end", "2026-07-30", "--workers", "1", "--out", str(tmp_path), *extra])
    st["files"] = lambda: sorted(p.name for p in tmp_path.glob("*.parquet"))
    st["read"] = lambda name: pd.read_parquet(tmp_path / f"{name}_-60.0_-1.0.parquet")
    return st


@pytest.mark.parametrize("mode", ["NISAR", "OPERA"])
def test_rain_is_retrieved_with_either_sar_mode(env, mode):
    env["run"]("--dataset", mode)
    assert env["searched"] == [mode]
    assert env["files"]() == ["imerg_-60.0_-1.0.parquet", f"{mode.lower()}_-60.0_-1.0.parquet"]
    df = env["read"](mode.lower())
    assert {"rain_d0_mm", "rain_1d_mm", "rain_7d_mm", "rain_30d_mm"} <= set(df.columns)
    assert np.allclose(df["rain_3d_mm"], 6.0)


def test_no_imerg_flag(env):
    env["run"]("--dataset", "NISAR", "--no-imerg")
    assert env["rain_calls"] == 0 and "imerg_-60.0_-1.0.parquet" not in env["files"]()
    assert "rain_1d_mm" not in env["read"]("nisar").columns


@pytest.mark.parametrize("err", [imerg.ImergAccessError("authorize the app"), RuntimeError("cmr down")])
def test_rain_failure_never_costs_the_sar_data(env, err, caplog):
    env["rain"] = err
    with caplog.at_level("WARNING", logger="asf_cleaner"):
        env["run"]("--dataset", "NISAR")
    assert env["files"]() == ["nisar_-60.0_-1.0.parquet"]
    assert "rain_1d_mm" not in env["read"]("nisar").columns
    assert "authorize the app" in caplog.text or "cmr down" in caplog.text


def test_auto_mode_with_recent_range_uses_nisar_and_rain(env):
    env["run"]()
    assert env["searched"] == ["NISAR"] and env["rain_calls"] == 1


def test_sme2_flag_is_gone(env):
    with pytest.raises(SystemExit):
        env["run"]("--no-sme2")
