"""What a person is told: the sensor chosen, too little data, failures. No network."""
from datetime import datetime, timedelta, timezone

import requests

from asf_cleaner import notices as N
from asf_cleaner.config import NISAR_START, OPERA_START

UTC = timezone.utc
NOW = datetime(2026, 11, 15, tzinfo=UTC)


def every(n, step=12, first=datetime(2026, 1, 1, tzinfo=UTC)):
    return [first + timedelta(days=step * i) for i in range(n)]


def codes(notes):
    return [n["code"] for n in notes]


def test_render_fills_the_template():
    n = N.notice(N.WARNING, "x", "{a} of {b} scene(s)", a=2, b=5)
    assert N.render(n) == "2 of 5 scene(s)"


def test_recent_period_picks_nisar_and_says_why():
    notes = N.dataset_notices("auto", "NISAR", NISAR_START + timedelta(days=30), NOW, NOW)
    assert codes(notes) == ["dataset-nisar"]


def test_period_across_the_nisar_start_uses_opera_and_says_nisar_was_left_out():
    notes = N.dataset_notices("auto", "OPERA", datetime(2025, 1, 1, tzinfo=UTC), NOW, NOW)
    assert codes(notes) == ["dataset-opera", "nisar-partial"]
    assert notes[1]["vars"]["from"] == "2025-10-01"


def test_period_before_the_archives_is_flagged():
    before = N.dataset_notices("auto", "OPERA", datetime(2010, 1, 1, tzinfo=UTC), datetime(2012, 1, 1, tzinfo=UTC), NOW)
    assert before[-1]["code"] == "before-opera" and before[-1]["level"] == N.WARNING
    partly = N.dataset_notices("auto", "OPERA", datetime(2015, 6, 1, tzinfo=UTC), datetime(2017, 1, 1, tzinfo=UTC), NOW)
    assert partly[-1]["code"] == "before-opera" and partly[-1]["level"] == N.INFO
    forced = N.dataset_notices("NISAR", "NISAR", datetime(2025, 1, 1, tzinfo=UTC), NOW, NOW)
    assert "before-nisar" in codes(forced)


def test_period_in_the_future_is_flagged():
    notes = N.dataset_notices("auto", "NISAR", NISAR_START + timedelta(days=30), NOW + timedelta(days=90), NOW)
    assert "future-end" in codes(notes)


def test_no_scenes_is_insufficient():
    a, notes = N.assess_series("NISAR", NISAR_START, NOW, [])
    assert a["confidence"] == "insufficient" and a["scenes"] == 0
    assert "no-scenes" in codes(notes)


def test_two_scenes_are_not_enough_to_see_a_pattern():
    a, notes = N.assess_series("OPERA", datetime(2026, 1, 1, tzinfo=UTC), NOW, every(2))
    assert a["confidence"] == "insufficient"
    assert "too-few-scenes" in codes(notes)


def test_few_scenes_give_a_limited_analysis():
    a, notes = N.assess_series("OPERA", datetime(2026, 1, 1, tzinfo=UTC), datetime(2026, 4, 1, tzinfo=UTC), every(6, 15))
    assert a["confidence"] == "limited"
    assert "few-scenes" in codes(notes)


def test_a_long_dense_series_is_high_confidence():
    a, notes = N.assess_series("OPERA", datetime(2024, 1, 1, tzinfo=UTC), datetime(2026, 1, 1, tzinfo=UTC),
                               every(60, 12, datetime(2024, 1, 1, tzinfo=UTC)))
    assert a["confidence"] == "high" and a["longestGapDays"] == 12
    assert [n for n in notes if n["level"] != N.INFO] == []


def test_a_gap_and_partial_coverage_are_reported():
    times = every(8, 12, datetime(2025, 1, 1, tzinfo=UTC))  # only the first ~3 months of two years
    a, notes = N.assess_series("OPERA", datetime(2025, 1, 1, tzinfo=UTC), datetime(2027, 1, 1, tzinfo=UTC), times)
    assert a["confidence"] == "limited" and "partial-coverage" in codes(notes)
    times = every(6, 12, datetime(2025, 1, 1, tzinfo=UTC)) + every(6, 12, datetime(2025, 9, 1, tzinfo=UTC))
    a, notes = N.assess_series("OPERA", datetime(2025, 1, 1, tzinfo=UTC), datetime(2025, 12, 31, tzinfo=UTC), times)
    gap = next(n for n in notes if n["code"] == "gaps")
    assert gap["vars"]["days"] > N.MAX_GAP_DAYS and a["confidence"] == "limited"


def test_cut_short_period_and_failed_scenes_are_reported():
    a, notes = N.assess_series("NISAR", datetime(2025, 10, 1, tzinfo=UTC), NOW, every(12), failed=5, total=17, truncated_to=17, matches=40)
    assert {"scenes-failed", "truncated"} <= set(codes(notes))
    assert a["confidence"] == "limited"
    assert next(n for n in notes if n["code"] == "truncated")["vars"] == {"matches": 40, "kept": 17}


def test_rain_notices():
    notes = N.rain_notices([datetime(2026, 11, 14).date()], 3, 20)
    assert codes(notes) == ["rain-missing", "rain-failed"]
    assert N.rain_notices([], 0, 0) == []


def test_errors_are_put_in_words():
    assert N.describe_error(RuntimeError("EARTHDATA_TOKEN not set. Generate a token"))["code"] == "token-missing"
    assert N.describe_error(PermissionError("Earthdata auth failed (401)"))["code"] == "auth-failed"
    assert N.describe_error(requests.ConnectTimeout("x"))["code"] == "archive-unreachable"
    http = requests.HTTPError(response=type("R", (), {"status_code": 502})())
    n = N.describe_error(http)
    assert n["code"] == "archive-error" and n["vars"] == {"status": 502}
    n = N.describe_error(ValueError("date range is longer than 12 years"))
    assert n["code"] == "failed" and n["message"] == "date range is longer than 12 years"
    assert N.describe_error(KeyError("k"))["message"].startswith("KeyError")
    assert "detail" in n and n["level"] == N.ERROR


def test_opera_start_is_the_history_start():
    from asf_cleaner import history
    assert history.RADAR_START == OPERA_START
