# asf-cleaner

Pulls SAR backscatter at a point from ASF, streaming only the pixels it needs (HTTP range reads,
no full downloads), masks invalid pixels, converts to dB, and writes a CSV and Parquet file.

It has **two independent modes**, chosen from the time range you ask for:

| Mode | Use it for | Sensor | Polarizations | Coverage |
|---|---|---|---|---|
| **NISAR** (GCOV) | Recent, high-accuracy data | L-band | HH, HV | from 2025-10 |
| **OPERA** (RTC-S1, Sentinel-1) | Historical series | C-band | VV, VH | 2016 to now |

The two datasets are **not** compared, calibrated against each other or merged. Analyse each one on
its own.

**IMERG** daily rain (NASA GPM) is retrieved alongside the backscatter with **either** SAR mode and
written to `imerg_<lon>_<lat>`; rain columns (`rain_1d_mm`, ...) are also added to the SAR table.

## Setup

    uv sync
    export EARTHDATA_TOKEN=...        # or put it in a .env file (keep .env out of git)

Create the token at https://urs.earthdata.nasa.gov ("Generate Token").

**IMERG needs one extra step.** Rain data is hosted by NASA GES DISC, which is authorized separately
from ASF. Once, in Earthdata Login: Applications > Authorized Apps > Approve More Applications, and
approve **NASA GESDISC DATA ARCHIVE** (plus an OPeNDAP application if one is listed). Without it the
run still finishes: the SAR data is written and IMERG is skipped with a message saying so. Test
access with `uv run python -m asf_cleaner.imerg -60.025 -1.74118`.

## Usage

    # Recent data -> NISAR (auto-selected)
    uv run asf-clean --lon -47.9 --lat -15.8 --start 2026-06-01 --end 2026-11-13

    # Historical series -> OPERA (auto-selected)
    uv run asf-clean --lon -47.9 --lat -15.8 --start 2018-01-01 --end 2023-12-31

    # No dates: the last 180 days (-> NISAR)
    uv run asf-clean --lon -47.9 --lat -15.8

    # Force a mode
    uv run asf-clean --dataset OPERA --start 2026-06-01 --end 2026-11-13

Longitude is negative west of Greenwich (`-47.9` is Brazil, `47.9` is Madagascar).

| Flag | Default | Meaning |
|---|---|---|
| `--lon`, `--lat` | see `config.py` | Point of interest |
| `--start`, `--end` | last 180 days | `2026-06-01` or `2026-06-01T00:00:00Z`. A bare `--end` date includes that whole day |
| `--dataset` | `auto` | `auto`, `NISAR` or `OPERA` |
| `--max-results` | 1000 | Cap on the total number of scenes processed (see below) |
| `--window-m` | 500 | Ground footprint in meters read around the point |
| `--workers` | 8 | Scenes processed in parallel (`1` = sequential) |
| `--no-cache` | off | Refetch everything and ignore the cache |
| `--no-imerg` | off | Skip the IMERG rain data |
| `--out` | `out` | Output folder |

Defaults can also be edited in `src/asf_cleaner/config.py`.

### How the mode is chosen (`--dataset auto`)
- The whole range is on or after 2025-10-01 (the start of the NISAR archive): **NISAR**.
- Anything else, including a range that starts before and ends after that date: **OPERA**.
  OPERA also covers the recent period, so a long range stays one continuous series.
- The chosen mode and the reason are logged at startup.

NISAR coverage depends on location. The archive starts in 2025-10 worldwide, but your point may
have its first scene later (the point `-47.9, -15.8` starts in 2026-06). If a NISAR run returns
nothing, check the range.

## Pagination and `--max-results`
ASF has no offset parameter, so large ranges are paginated automatically: the tool asks ASF for the
scene count, splits the time range in halves until each piece fits in one request, and collects
the pieces newest first. You don't need to do anything for long ranges.

`--max-results` caps the total. If more scenes match than the cap, the **newest** `--max-results`
are processed and a warning is logged:

    WARNING max-results=50 is smaller than the 318 scenes available: only the NEWEST 50 will be
    processed and the older ones are skipped. Raise --max-results to get them all.

A truncated series biases any trend analysis, so don't ignore that warning. Each scene costs a
few network requests, so raise the cap deliberately for very long ranges.

## Speed: parallel scenes and the cache
Fetching is network-bound (several small requests per scene), so scenes are processed in parallel
and already-processed scenes are never fetched again.

- **`--workers N`** (default 8). OPERA uses a thread pool (GDAL releases the GIL); NISAR uses
  separate processes, because h5py holds a global lock while reading over the network and threads
  would just queue behind it. The pool type per dataset is `EXECUTORS` in `config.py`. More workers
  is not always faster and ASF may throttle (HTTP 429): stay at 16 or below. The log reports
  scenes/second so you can compare `--workers 1` against higher values on your connection.
- **Retries:** transient network errors are retried with backoff (HTTP 429/5xx at the connection
  level, and up to 3 attempts per scene). Auth failures (bad token) and "point outside the scene"
  are not retried. A scene that still fails is logged and skipped, not cached, and fetched again on
  the next run.
- **Cache:** processed scenes are stored in `<out>/.cache/<dataset>_<lon>_<lat>_w<window_m>.parquet`.
  A re-run, or a run with an extended date range, only fetches the scenes it doesn't have. A fully
  cached run needs no token and no downloads. The output file contains only the scenes of the
  range you asked for, while the cache keeps everything ever processed, so a narrow run never
  destroys earlier work. Rows are reused only if the point, `--window-m` and the cleaning logic
  (`PROCESSING_VERSION` in `config.py`) are unchanged; cached rows from an older version are
  refetched automatically. Use `--no-cache` to force a refetch, or delete `<out>/.cache`.
- **Ctrl-C** cancels pending scenes and saves the ones already finished to the cache.

## Output
`out/<dataset>_<lon>_<lat>.csv` and `.parquet`, one row per scene, burst and polarization.

| Column | Meaning |
|---|---|
| `sensor`, `band`, `platform` | NISAR/OPERA, L/C band, satellite |
| `scene`, `start_time` | Granule name and acquisition time (UTC) |
| `burst_id` | OPERA burst (empty for NISAR) |
| `track`, `direction` | Orbit track and ASCENDING/DESCENDING |
| `polarization`, `pol_class` | Channel (HH, HV, VV, VH) and `co` or `cross` |
| `valid_fraction` | Share of window pixels that were usable (0 to 1) |
| `backscatter_db_mean`, `_median`, `_center` | Backscatter in dB over the window, and the point pixel |
| `window_m`, `window_px`, `pixel_spacing_m` | Footprint actually read |
| `normalization` | Radiometric normalization (gamma0 for both; see caveats) |
| `processing_version` | Version of the cleaning logic that produced the row (used by the cache) |

## Rain: IMERG (`imerg_<lon>_<lat>`)
NASA GPM IMERG V07 **daily** precipitation (0.1 degree grid, mm/day, total of the UTC day), at the
grid cell containing the point. Fetched with OPeNDAP, one tiny request per day (no files are
downloaded), in parallel and cached like the SAR scenes.

- **Range:** `--start - 30 days` to `--end`, so the first scenes still have a full 30-day history.
- **Final, then Late:** per day, the Final run (gauge-adjusted) when it exists, otherwise the Late
  run. Final currently ends 2025-09-30, so NISAR-era days always come from Late. `imerg_run` says
  which one each day used; a day cached from Late is refetched once Final appears for it. The
  newest day or two are not published yet and are simply absent.
- **Columns:** `date`, `precipitation_mm_day`, `imerg_run`, `granule`, `grid_lon`, `grid_lat`
  (the cell centre actually used), `units`, `processing_version`.
- **Rain before each acquisition** is added to the SAR table:
  `rain_d0_mm` (the acquisition's own UTC day; daily totals can't be split at the overpass time, so
  this may include rain after the pass) and `rain_1d_mm`, `rain_3d_mm`, `rain_7d_mm`, `rain_30d_mm`
  (totals of the N days **before** the acquisition day). A value is empty if any day it needs is
  missing.
- **A rain problem never costs you the SAR data.** If IMERG access fails, the SAR file is still
  written without the rain columns.

## Analysis (`asf-analyze`)
Reads the Parquet (or CSV) output **lazily with Polars**, so only the series you select is loaded.

    uv run asf-analyze out/opera_-47.9_-15.8.parquet        # list the available series
    uv run asf-analyze out/opera_-47.9_-15.8.parquet \
        --track 24 --direction DESCENDING --polarization VH --plot series.png

For one series (same burst, track, direction, polarization; scenes below `--min-valid`, default
0.8, are dropped) it reports:

1. **Median level and spread** of `backscatter_db_median`.
2. **Deseasonalized anomaly:** value minus the median for that calendar month. Only done when the
   series spans at least 2 years; otherwise it warns and uses the overall median.
3. **Trend:** Theil-Sen slope in dB/year with a 95% CI and a Mann-Kendall p-value. Needs at least
   2 years and 10 scenes. The p-value assumes independent samples, so it is optimistic; prefer
   the CI and look at the plot.
4. **Sustained events:** runs of at least `--min-run` (3) consecutive scenes more than
   `--threshold` (2.5) dB from the baseline, in the same direction.
5. **Isolated outliers:** the same test but shorter than `--min-run` scenes (a single dip, say).
   They are not events, but they are listed separately so they are not silently dropped.
6. **Rain context** for every event and outlier, from the IMERG columns `asf-clean` adds:
   `rain_context` is `wet` if IMERG showed at least `--rain-mm` (10) mm on the acquisition day or the
   day before, `dry` if less, `unknown` if there is no rain data. The rain at the first scene of
   the run is shown too (`onset_rain_d0_mm`, `_1d`, `_3d`, `_7d`, `_30d`). Files made before IMERG
   support have no rain columns: the analysis says so and reports `unknown`.
7. **Cross-pol minus co-pol (dB)** per date: higher means relatively more volume scattering
   (vegetation), lower means more surface scattering.
8. **Gaps** longer than 45 days (no scenes in the series). Often another polarization was
   acquired instead (some periods are VV-only), or the point had no valid data.
9. **A PNG** (`--plot`): backscatter, the anomaly (event spans shaded, outliers marked and coloured
   by rain context), the cross/co ratio, and a **daily IMERG rain panel**. The rain comes from the
   `imerg_*` file next to the input (or `--rain FILE`); without it the panel shows the rain of the
   3 days before each scene, and without any rain data there is no rain panel. Lines are broken
   across gaps instead of bridging them.

The same functions can be imported (`from asf_cleaner import analysis`): `scan`, `list_series`,
`select_series`, `deseasonalize`, `trend`, `detect_events`, `detect_spikes`, `add_rain_context`,
`find_gaps`, `load_daily_rain`, `cross_pol_ratio`.

Practical notes:
- **Prefer the median** over the centre pixel (speckle).
- **Cross-pol** (HV/VH) is the better channel for vegetation change; co-pol (HH/VV) responds
  more to soil moisture and surface changes.
- **An anomaly says something changed, not what.** `wet` makes rain a plausible cause and `dry`
  makes it a less likely one, but IMERG cells are about 11 km wide: `dry` can miss a local storm and
  `wet` can be rain that fell elsewhere or after the overpass (daily totals can't be split at the
  pass time). Confirm with optical imagery or known event dates before calling it an event.
- **Series with `valid_fraction` near 0** mean the point has no valid data in those scenes (for
  example outside the swath); they are excluded automatically.
- Seasonality and trend need years of data, so they suit OPERA. NISAR currently has a few months.

## HTTP API (`asf-api`)
Serves the same cleaning pipeline over HTTP, **on demand**: nothing is precomputed. A request
searches ASF, reads only the needed pixels of each scene and streams the cleaned rows back as
Server-Sent Events while the scenes finish, so a client can draw the series as it is built.

    export EARTHDATA_TOKEN=...          # stays on the server, never sent to clients
    uv run asf-api                       # http://127.0.0.1:8000, interactive docs at /docs

| Endpoint | Returns |
|---|---|
| `GET /health` | `{status, tokenConfigured, maxJobs}` |
| `GET /series/stream` | SSE: `meta`, then `rows` (cache first, then one event per finished scene), `scene_error`, `done` (or `fatal`) |
| `GET /series` | The same data as one JSON document, once every scene is done |

Query parameters (both series endpoints): `lon`, `lat` (required), `start`, `end`, `dataset`
(`auto`, `NISAR`, `OPERA`), `window_m` (30 to 5000), `max_results` (default 400, max 1000),
`workers` (1 to 16), `cache` (`false` ignores the cache).

    curl -N 'http://127.0.0.1:8000/series/stream?lon=-60.0217&lat=-3.119&start=2026-06-01'

Each `rows` event holds `rows: [{scene, sensor, band, platform, t (epoch ms), burst, track, direction,
pol, polClass, db (median dB), dbMean, dbCenter, valid}]`, plus `source` (`cache` or `live`), `done`
and `total`. Closing the connection cancels the scenes not yet started. Scenes already processed
are served from the same cache as the CLI (`ASF_CACHE_DIR`, default `out`).

Environment: `ASF_CACHE_DIR` (cache folder), `ASF_API_CORS` (comma-separated allowed origins,
default the Vite dev server), `ASF_API_MAX_JOBS` (concurrent requests, default 4; more get HTTP 503).
The API serves the SAR backscatter only; IMERG rain is a CLI output for now.
Timing is network-bound: a NISAR scene takes about a minute and a half, an OPERA scene a few
seconds, so the first request for a new point is slow and repeats are instant.

## Caveats
- **Masks differ by dataset.** OPERA: shadow/layover/invalid pixels are removed (mask 0 = valid).
  NISAR: pixels are removed when the "valid samples" mask is 0 (not fully focused) or 255
  (outside the acquisition). NISAR's mask says nothing about shadow or layover, so those pixels
  are not filtered in NISAR mode, which matters on steep terrain.
- **`normalization`:** OPERA is gamma0. For NISAR it is reported as gamma0 because GCOV ships
  `rtcGammaToSigmaFactor` (a gamma0-to-sigma0 conversion), which implies gamma0 is stored.
- **OPERA uses GeoTIFFs**, not the `.h5` (which only holds metadata). NISAR is read through the `.h5`.
- **Burst overlaps:** near a burst edge a point can appear in two bursts on one date; they are
  separate rows.
- **Platform and product versions:** OPERA spans Sentinel-1A, 1B and 1C (`platform` column) and
  was reprocessed in 2025; NISAR is early in its mission, so calibration may change. Watch for
  steps in a long series at those boundaries.
- **Single patch:** results describe the window around the point (`--window-m`), not the surrounding area.
- **Rain is coarse.** An IMERG cell is about 11 km wide, far larger than the SAR window, so it is
  the regional rain, not the rain on the patch. Convective storms can miss the point entirely.
- **Mixed IMERG runs:** Final (gauge-adjusted) and Late (not) can differ systematically. Check
  `imerg_run` before comparing rain across 2025-09-30.

## Development

    uv run pytest      # no network or token needed
