"""Messages for the person using the map, not only for the server log.

Anything the backend used to say only on stdout (a range cut short, scenes that could not be read, too little
data to trust a pattern, a failure) is also sent to the client as a `notice` event. A notice is a plain dict:

    {"level": "info" | "warning" | "error", "code": "few-scenes", "message": "{n} scenes ...", "vars": {"n": 4}}

`message` is an English template with `{name}` placeholders and `vars` fills them, so a client can translate
the template and still fill in the numbers. `render` does the English version (logs, the blocking endpoints).

`assess_series` is the judgement of "is this enough time to trust an analysis?" with the reasons behind it.
"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone

import requests

from .config import NISAR_START, OPERA_START

log = logging.getLogger("asf_cleaner")

INFO, WARNING, ERROR = "info", "warning", "error"

# What "enough time" means. The scene counts and the gap size are the bars the trend analysis already uses
# (analysis.py: at least 10 scenes for a trend, gaps longer than 45 days).
MIN_SCENES = 3  # fewer scenes: nothing can be said about change over time
GOOD_SCENES = 10  # from here the picture is steady
MAX_GAP_DAYS = 45
MIN_COVERAGE = 0.5  # share of the asked period that lies between the first and the last scene
MAX_FAILED_SHARE = 0.25
SEASON_DAYS = 365  # a shorter period cannot tell a season from a real change

BAND = {"NISAR": "L-band", "OPERA": "C-band"}


def notice(level: str, code: str, message: str, **vars) -> dict:
    return {"level": level, "code": code, "message": message, "vars": vars}


def render(n: dict) -> str:
    out = n["message"]
    for k, v in n.get("vars", {}).items():
        out = out.replace("{" + k + "}", str(v))
    return out


def _day(t: datetime) -> str:
    return f"{t:%Y-%m-%d}"


# ---- which sensor, and what the period leaves out ---------------------------------------------
def dataset_notices(requested: str, dataset: str, start: datetime, end: datetime, now: datetime | None = None) -> list[dict]:
    """Which sensor was used and why, plus what the asked period cannot give."""
    now = now or datetime.now(timezone.utc)
    out: list[dict] = []
    if requested != "auto":
        out.append(notice(INFO, "dataset-forced", "Sensor chosen by the request: {dataset} ({band}).", dataset=dataset, band=BAND[dataset]))
    elif dataset == "NISAR":
        out.append(notice(INFO, "dataset-nisar", "Sensor chosen automatically: NISAR (L-band), because the whole period is inside its coverage (from {from}).", **{"from": _day(NISAR_START)}))
    else:
        out.append(notice(INFO, "dataset-opera", "Sensor chosen automatically: OPERA Sentinel-1 (C-band), because the period starts before NISAR coverage ({from}).", **{"from": _day(NISAR_START)}))
        if end >= NISAR_START:
            out.append(notice(INFO, "nisar-partial", "The period also holds {days} day(s) of NISAR coverage, but an analysis uses one sensor and the two are never merged. To use NISAR, start the period on {from} or later.", days=(end - NISAR_START).days + 1, **{"from": _day(NISAR_START)}))
    if dataset == "NISAR" and start < NISAR_START:
        out.append(notice(WARNING, "before-nisar", "NISAR data begins {from}; the part of the period before it has nothing.", **{"from": _day(NISAR_START)}))
    if dataset == "OPERA" and end < OPERA_START:
        out.append(notice(WARNING, "before-opera", "OPERA data begins {from}; this period ends before that.", **{"from": _day(OPERA_START)}))
    elif dataset == "OPERA" and start < OPERA_START:
        out.append(notice(INFO, "before-opera", "OPERA data begins {from}; the part of the period before it has nothing.", **{"from": _day(OPERA_START)}))
    if end > now + timedelta(days=1):
        out.append(notice(INFO, "future-end", "The period ends in the future; there is data only up to today."))
    return out


# ---- is it enough? ------------------------------------------------------------------------------
def assess_series(dataset: str, start: datetime, end: datetime, times: list[datetime], *, failed: int = 0, total: int = 0,
                  truncated_to: int | None = None, matches: int | None = None) -> tuple[dict, list[dict]]:
    """Judge a radar series by the time it covers.

    `times` are the start times of the scenes that were read. Returns ({confidence, ...numbers}, notices) with
    confidence "high", "limited" (usable, with caveats) or "insufficient" (too little to say anything over time).
    """
    times = sorted(times)
    n = len(times)
    days_asked = max(1, (end - start).days)
    notes: list[dict] = []
    reasons = 0  # what pulls confidence down from high to limited
    gap = (0, None, None)
    for a, b in zip(times, times[1:]):
        if (b - a).days > gap[0]:
            gap = ((b - a).days, a, b)
    coverage = ((times[-1] - times[0]).days / days_asked) if n >= 2 else 0.0

    if failed:
        share = failed / total if total else 1.0
        notes.append(notice(WARNING, "scenes-failed", "{failed} of {total} scene(s) could not be read; the series has those gaps.", failed=failed, total=total))
        if share > MAX_FAILED_SHARE:
            reasons += 1
    if truncated_to is not None and matches is not None and matches > truncated_to:
        notes.append(notice(WARNING, "truncated", "The period holds {matches} scenes; only the newest {kept} were read. Choose a shorter period to read the rest.", matches=matches, kept=truncated_to))
        reasons += 1

    if n == 0:
        notes.append(notice(WARNING, "no-scenes", "No {dataset} scenes cover this place in this period, so there is nothing to analyse. Try a longer period or another place.", dataset=dataset))
        confidence = "insufficient"
    elif n < MIN_SCENES:
        notes.append(notice(WARNING, "too-few-scenes", "Only {n} scene(s) were read, and at least {min} are needed to see a pattern over time. Do not draw conclusions from this period.", n=n, min=MIN_SCENES))
        confidence = "insufficient"
    else:
        if n < GOOD_SCENES:
            notes.append(notice(WARNING, "few-scenes", "{n} scenes are enough for a first look, but {good} or more give a steadier picture.", n=n, good=GOOD_SCENES))
            reasons += 1
        if coverage < MIN_COVERAGE and days_asked > 2 * MAX_GAP_DAYS:
            notes.append(notice(WARNING, "partial-coverage", "The scenes cover only {pct}% of the chosen period ({first} to {last}).", pct=round(100 * coverage), first=_day(times[0]), last=_day(times[-1])))
            reasons += 1
        if gap[0] > MAX_GAP_DAYS:
            notes.append(notice(WARNING, "gaps", "Longest stretch without scenes: {days} days ({first} to {last}).", days=gap[0], first=_day(gap[1]), last=_day(gap[2])))
            reasons += 1
        if days_asked < SEASON_DAYS:
            notes.append(notice(INFO, "short-period", "A period shorter than a year cannot tell a season from a real change."))
        confidence = "limited" if reasons else "high"

    assessment = {"confidence": confidence, "scenes": n, "failed": failed, "periodDays": days_asked,
                  "coverage": round(coverage, 2), "longestGapDays": gap[0]}
    return assessment, notes


def rain_notices(missing: list, failed: int, total: int) -> list[dict]:
    """What the rain stream could not give: days NASA has not published yet and days that failed."""
    out = []
    if missing:
        out.append(notice(INFO, "rain-missing", "NASA has no rain data for {n} day(s) of this period (the latest is {last}); recent days are published with a delay.", n=len(missing), last=str(missing[-1])))
    if failed:
        out.append(notice(WARNING, "rain-failed", "{n} day(s) of rain could not be read ({total} asked).", n=failed, total=total))
    return out


# ---- failures ------------------------------------------------------------------------------------
def describe_error(err: BaseException) -> dict:
    """A failure in words a person can act on. The technical text stays in `detail`."""
    detail = f"{type(err).__name__}: {err}"
    text = str(err)
    if isinstance(err, RuntimeError) and "EARTHDATA_TOKEN" in text:
        n = notice(ERROR, "token-missing", "The server has no Earthdata token configured, so it cannot read NASA data.")
    elif isinstance(err, PermissionError):
        n = notice(ERROR, "auth-failed", "Earthdata refused the token (expired, or not authorized for this data). The server needs a new one.")
    elif isinstance(err, (requests.Timeout, requests.ConnectionError)):
        n = notice(ERROR, "archive-unreachable", "The NASA archive did not answer. Try again in a moment.")
    elif isinstance(err, requests.HTTPError):
        status = getattr(err.response, "status_code", None)
        n = notice(ERROR, "archive-error", "The NASA archive answered with an error ({status}). Try again in a moment.", status=status if status is not None else "?")
    elif isinstance(err, RuntimeError) and "GES DISC" in text:
        n = notice(ERROR, "rain-access", text)  # already says which authorization is missing and where to give it
    else:
        # Errors the code raised on purpose (bad range, rain access not authorized...) are already plain language.
        n = notice(ERROR, "failed", text if isinstance(err, (ValueError, RuntimeError)) else detail)
    n["detail"] = detail
    return n
