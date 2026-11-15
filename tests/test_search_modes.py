import logging
from datetime import datetime, timedelta, timezone

import pytest

from asf_cleaner import search as S
from asf_cleaner.config import NISAR_START, parse_time, resolve_dataset, resolve_range

UTC = timezone.utc


def fake_asf(monkeypatch, times, page=10):
    """Fake ASF: one NISAR-like feature per timestamp; newest first; no offset support."""
    monkeypatch.setattr(S, "PAGE_SIZE", page)
    calls = {"features": 0}

    def inrange(t0, t1):
        return sorted((t for t in times if t0 <= t <= t1), reverse=True)

    monkeypatch.setattr(S, "_count", lambda k, lo, la, t0, t1: len(inrange(t0, t1)))

    def feats(k, lo, la, t0, t1, limit):
        calls["features"] += 1
        assert limit <= page
        return [{"properties": {"sceneName": f"s{t:%Y%m%d%H%M%S}", "startTime": t.isoformat(),
                                "url": "x.h5"}} for t in inrange(t0, t1)[:limit]]

    monkeypatch.setattr(S, "_features", feats)
    return calls


def series(n):
    t = datetime(2020, 1, 1, tzinfo=UTC)
    return [t + timedelta(days=12 * i) for i in range(n)]


def test_pagination_gets_everything(monkeypatch):
    times = series(95)  # 9.5 pages
    calls = fake_asf(monkeypatch, times)
    out = S.search("NISAR", 0, 0, times[0] - timedelta(days=1), times[-1], max_results=1000)
    assert len(out) == 95 and len({s["scene"] for s in out}) == 95
    assert calls["features"] > 1  # bisected, not one request


def test_cap_keeps_newest_and_warns(monkeypatch, caplog):
    times = series(95)
    fake_asf(monkeypatch, times)
    with caplog.at_level(logging.WARNING, logger="asf_cleaner"):
        out = S.search("NISAR", 0, 0, times[0] - timedelta(days=1), times[-1], max_results=30)
    assert len(out) == 30
    assert {s["scene"] for s in out} == {f"s{t:%Y%m%d%H%M%S}" for t in times[-30:]}
    assert "max-results=30" in caplog.text and "95" in caplog.text


def test_no_warning_when_enough(monkeypatch, caplog):
    times = series(20)
    fake_asf(monkeypatch, times)
    with caplog.at_level(logging.WARNING, logger="asf_cleaner"):
        S.search("NISAR", 0, 0, times[0] - timedelta(days=1), times[-1], max_results=50)
    assert "max-results" not in caplog.text


def test_mode_selection():
    after = NISAR_START + timedelta(days=30)
    assert resolve_dataset("auto", after, after + timedelta(days=60))[0] == "NISAR"
    assert resolve_dataset("auto", datetime(2018, 1, 1, tzinfo=UTC), after)[0] == "OPERA"  # spans
    assert resolve_dataset("auto", datetime(2018, 1, 1, tzinfo=UTC), datetime(2019, 1, 1, tzinfo=UTC))[0] == "OPERA"
    assert resolve_dataset("OPERA", after, after + timedelta(days=5))[0] == "OPERA"
    assert "begins" in resolve_dataset("NISAR", datetime(2018, 1, 1, tzinfo=UTC), after)[1]


def test_range_parsing():
    assert parse_time("2026-06-01") == datetime(2026, 6, 1, tzinfo=UTC)
    assert parse_time("2026-06-01T00:00:00Z") == datetime(2026, 6, 1, tzinfo=UTC)
    assert parse_time("2026-06-01", end_of_day=True) == datetime(2026, 6, 1, 23, 59, 59, tzinfo=UTC)
    now = datetime(2026, 11, 14, 12, tzinfo=UTC)
    s, e = resolve_range(None, None, now)
    assert e == now and (e - s).days == 180
    with pytest.raises(ValueError):
        resolve_range("2026-06-02", "2026-06-01")


# ---- legacy datapool URLs ------------------------------------------------------------------
NAME = "OPERA_L2_RTC-S1_T010-020420-IW3_20250704T093925Z_20250710T173632Z_S1A_30_v1.0"
S3 = [f"s3://asf-cumulus-prod-opera-browse/OPERA_L2_RTC-S1/{NAME}/{NAME}_BROWSE.png",
      f"s3://asf-cumulus-prod-opera-products/OPERA_L2_RTC-S1/{NAME}/{NAME}_VV.tif.md5",
      f"s3://asf-cumulus-prod-opera-products/OPERA_L2_RTC-S1/{NAME}/{NAME}_VV.tif",
      f"s3://asf-cumulus-prod-opera-products/OPERA_L2_RTC-S1/{NAME}/{NAME}_VH.tif",
      f"s3://asf-cumulus-prod-opera-products/OPERA_L2_RTC-S1/{NAME}/{NAME}_mask.tif"]
CLOUD = f"https://cumulus.asf.earthdatacloud.nasa.gov/OPERA/OPERA_L2_RTC-S1/{NAME}/{NAME}"


def test_legacy_datapool_urls_rewritten():
    legacy = f"https://datapool.asf.alaska.edu/RTC/OPERA-S1/{NAME}_VV.tif"
    assert S._cloud_url(legacy, S3) == CLOUD + "_VV.tif"  # not the .md5 neighbour
    assert S._cloud_url(CLOUD + "_VV.tif", S3) == CLOUD + "_VV.tif"  # already fine: untouched
    assert S._cloud_url(legacy, []) == legacy  # no s3 info: fall back to the original


def test_search_returns_cloud_urls_for_legacy_granules(monkeypatch):
    t = datetime(2025, 7, 4, tzinfo=UTC)
    feat = {"properties": {
        "sceneName": NAME, "startTime": "2025-07-04T09:39:25Z", "flightDirection": "DESCENDING",
        "pathNumber": 10, "platform": "Sentinel-1A", "operaBurstID": "T010_020420_IW3",
        "url": f"https://datapool.asf.alaska.edu/RTC/OPERA-S1/{NAME}.h5", "s3Urls": S3,
        "additionalUrls": [f"https://datapool.asf.alaska.edu/RTC/OPERA-S1/{NAME}_{x}.tif" for x in ("VH", "VV", "mask")]}}
    monkeypatch.setattr(S, "_count", lambda *a: 1)
    monkeypatch.setattr(S, "_features", lambda *a: [feat])
    sc = S.search("OPERA", 0, 0, t, t + timedelta(days=1))[0]
    assert sc["tifs"] == {x: f"{CLOUD}_{x}.tif" for x in ("VH", "VV", "mask")}


def test_search_counts_once_when_one_page_is_enough(monkeypatch):
    """The total is already known from the warning check; asking ASF for it again wastes a request."""
    times = series(20)
    fake_asf(monkeypatch, times, page=50)  # one page is enough: no bisecting
    counts = []
    real = S._count
    monkeypatch.setattr(S, "_count", lambda *a: counts.append(a) or real(*a))
    out = S.search("NISAR", 0, 0, times[0] - timedelta(days=1), times[-1], max_results=50)
    assert len(out) == 20 and len(counts) == 1
