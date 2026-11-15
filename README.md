# asf-cleaner

**Backscatter and rain time series for any point on Earth, straight from NASA's archives.**

Give it a longitude and latitude and a date range. It finds the satellite radar scenes that cover
that point, reads **only the few pixels around it** from each scene (no multi-gigabyte downloads),
cleans them, and hands back a tidy time series of radar backscatter in dB together with the rain
that fell before each acquisition. It is meant for watching how land surfaces change over time:
vegetation, soil moisture, flooding, clearing, fire.

| Data | Source | What you get |
|---|---|---|
| **NISAR GCOV** (L-band) | ASF | Recent, high-quality backscatter (HH, HV), from 2025-10 |
| **OPERA RTC-S1** (Sentinel-1, C-band) | ASF | A long historical series (VV, VH), 2016 to now |
| **IMERG** (NASA GPM) | NASA GES DISC | Daily rain at the point, fetched with every run |

## Two ways to use it

The same pipeline is available in two forms, so you can test locally and then plug it into an app:

| | **CLI tool** (`asf-clean`, `asf-analyze`) | **HTTP API** (`asf-api`) |
|---|---|---|
| For | Local testing, scripts, notebooks, one-off analysis | Connecting to a **front-end** (web app, dashboard) |
| You run | `uv run asf-clean --lon ... --lat ...` | `uv run asf-api`, then call it from the browser |
| Output | CSV and Parquet files in `out/`, plus a plot | JSON, streamed live as each scene finishes (Server-Sent Events) |
| Rain (IMERG) | Yes | Not yet (SAR backscatter only) |
| Analysis (trend, events, plots) | Yes (`asf-analyze`) | No, done by the front-end |
| Shared | The same cleaning code and the **same cache**: a scene fetched by one is instant for the other | |

> **Typical workflow:** try a point with the CLI first (`asf-clean`, then `asf-analyze`) to see what
> the data looks like, then run `asf-api` and let your front-end request the same points.

## Quick start

### 1. Get the code

    git clone https://github.com/luiz-fleal/asf-cleaner.git
    cd asf-cleaner

You need [uv](https://docs.astral.sh/uv/) (it installs the right Python, 3.12, by itself):

    curl -LsSf https://astral.sh/uv/install.sh | sh     # Linux, macOS, WSL
    uv sync                                              # installs all dependencies

### 2. Add your NASA Earthdata token

ASF and NASA require a free [Earthdata Login](https://urs.earthdata.nasa.gov) account. Generate a
token there ("Generate Token", valid for up to 60 days) and give it to the tool in either way:

    export EARTHDATA_TOKEN=your_token_here
    # or put the line  EARTHDATA_TOKEN=your_token_here  in a file called .env in the folder
    # you run the commands from (.env is git-ignored)

The token stays on the machine running the tool; the API never sends it to clients.

**Rain (IMERG) needs one extra, one-time step.** IMERG is hosted by NASA GES DISC, which is
authorized separately from ASF. In Earthdata Login go to *Applications > Authorized Apps > Approve
More Applications* and approve **NASA GESDISC DATA ARCHIVE** (and an OPeNDAP application if one is
listed). Without this the run still works: the SAR data is written and rain is skipped with a
message saying why. Check access with:

    uv run python -m asf_cleaner.imerg -60.025 -1.74118

### 3a. Use it as a CLI tool (local testing)

    # Recent data -> NISAR is picked automatically
    uv run asf-clean --lon -60.025 --lat -1.74118 --start 2026-06-01 --end 2026-11-13

    # A long historical series -> OPERA is picked automatically
    uv run asf-clean --lon -60.025 --lat -1.74118 --start 2018-01-01

    # Look at the result: list the series, then analyse one and draw a plot
    uv run asf-analyze out/opera_-60.025_-1.74118.parquet
    uv run asf-analyze out/opera_-60.025_-1.74118.parquet \
        --track 10 --direction DESCENDING --polarization VH --plot series.png

Results land in `out/` (see [Output](#output)). Full options are in [CLI reference](#cli-reference).

### 3b. Use it as an API (for a front-end)

    uv run asf-api                  # http://127.0.0.1:8000, interactive docs at /docs

    curl -N 'http://127.0.0.1:8000/series/stream?lon=-60.0217&lat=-3.119&start=2026-06-01'

Rows stream back as each scene finishes, so a chart can be drawn while the series is still being
built. Details, parameters and a JavaScript example are in [HTTP API](#http-api).

### Check the install

    uv run pytest          # needs no network and no token

## Choosing the dataset: NISAR or OPERA

The two SAR datasets are **independent**: they are never compared, calibrated against each other or
merged. Analyse each on its own. By default (`--dataset auto`) the time range you ask for decides:

| Mode | Use it for | Sensor | Polarizations | Coverage |
|---|---|---|---|---|
| **NISAR** (GCOV) | Recent, high-accuracy data | L-band | HH, HV | from 2025-10 |
| **OPERA** (RTC-S1, Sentinel-1) | Historical series | C-band | VV, VH | 2016 to now |

- The whole range is on or after 2025-10-01 (the start of the NISAR archive): **NISAR**.
- Anything else, including a range that starts before and ends after that date: **OPERA**.
  OPERA also covers the recent period, so a long range stays one continuous series.
- The chosen mode and the reason are logged at startup; force one with `--dataset`.

NISAR coverage depends on location. The archive starts in 2025-10 worldwide, but your point may
have its first scene later.

## CLI reference

    uv run asf-clean [options]

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

Defaults can also be edited in `src/asf_cleaner/config.py`. Analysis options are under
[Analysis](#analysis-asf-analyze).

### Pagination and `--max-results`
ASF has no offset parameter, so large ranges are paginated automatically: the tool asks ASF for the
scene count, splits the time range in halves until each piece fits in one request, and collects
the pieces newest first. You don't need to do anything for long ranges.

`--max-results` caps the total. If more scenes match than the cap, the **newest** `--max-results`
are processed and a warning is logged:

    WARNING max-results=50 is smaller than the 318 scenes available: only the NEWEST 50 will be
    processed and the older ones are skipped. Raise --max-results to get them all.

A truncated series biases any trend analysis, so don't ignore that warning. Each scene costs a
few network requests, so raise the cap deliberately for very long ranges.

### Speed: parallel scenes and the cache
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
`out/<dataset>_<lon>_<lat>.csv` and `.parquet` (for example `out/opera_-60.025_-1.74118.parquet`),
one row per scene, burst and polarization. The daily rain goes to `out/imerg_<lon>_<lat>.*`.

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
| `rain_d0_mm`, `rain_1d_mm`, `rain_3d_mm`, `rain_7d_mm`, `rain_30d_mm` | IMERG rain before the acquisition (see below). Present when rain data was retrieved |

## Rain: IMERG (`imerg_<lon>_<lat>`)
NASA GPM IMERG V07 **daily** precipitation (0.1 degree grid, mm/day, total of the UTC day), at the
grid cell containing the point. Fetched with OPeNDAP, one tiny request per day (no files are
downloaded), in parallel and cached like the SAR scenes. It is retrieved with **either** SAR mode.

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
   across gaps instead of bridging them. The image is written to the path you give `--plot`,
   relative to the folder you run the command from.

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

## HTTP API

`asf-api` serves the same cleaning pipeline over HTTP so a **front-end** can use it. It works **on
demand**: nothing is precomputed. A request searches ASF, reads only the needed pixels of each scene
and streams the cleaned rows back as **Server-Sent Events** while the scenes finish, so a client
can draw the series as it is built. Scenes already processed (by the API or by the CLI) are served
instantly from the shared cache.

    export EARTHDATA_TOKEN=...          # stays on the server, never sent to clients
    uv run asf-api                       # http://127.0.0.1:8000, interactive docs at /docs
    uv run asf-api --host 0.0.0.0 --port 9000      # to listen elsewhere

| Endpoint | Returns |
|---|---|
| `GET /health` | `{status, tokenConfigured, maxJobs}`: use it to check the server and its token |
| `GET /series/stream` | SSE: `meta`, then `rows` (cache first, then one event per finished scene), `scene_error`, `done` (or `fatal`) |
| `GET /series` | The same data as one JSON document, once every scene is done |

Query parameters (both series endpoints):

| Parameter | Default | Meaning |
|---|---|---|
| `lon`, `lat` | required | Point (longitude negative west of Greenwich) |
| `start`, `end` | last 180 days | Dates like `2026-06-01` |
| `dataset` | `auto` | `auto`, `NISAR` or `OPERA` (same rule as the CLI) |
| `window_m` | 500 | Ground footprint, 30 to 5000 |
| `max_results` | 400 | Cap on scenes, up to 1000 |
| `workers` | 8 | Parallel scenes, 1 to 16 |
| `cache` | `true` | `false` ignores the cache |

    curl -N 'http://127.0.0.1:8000/series/stream?lon=-60.0217&lat=-3.119&start=2026-06-01'

A stream looks like this (illustrative values, shortened):

    event: meta
    data: {"dataset":"NISAR","reason":"range is within NISAR coverage ...","total":9,"cached":0,"toFetch":9,...}

    event: rows
    data: {"source":"live","done":1,"total":9,"rows":[{"scene":"NISAR_L2_...","t":1782856800000,"pol":"HH","db":-10.2,...}]}

    event: scene_error
    data: {"scene":"NISAR_L2_...","message":"ValueError: point falls outside the product grid","done":2}

    event: done
    data: {"fetched":8,"failed":1,"seconds":84.3}

Each `rows` event holds `rows: [{scene, sensor, band, platform, t (epoch ms), burst, track, direction,
pol, polClass, db (median dB), dbMean, dbCenter, valid}]`, plus `source` (`cache` or `live`), `done`
and `total`. A `fatal` event (`{message}`) means the whole job failed.

From a browser front-end:

```js
const url = "http://127.0.0.1:8000/series/stream?lon=-60.0217&lat=-3.119&start=2018-01-01";
const es = new EventSource(url);
const points = [];
es.addEventListener("meta", (e) => console.log("scenes:", JSON.parse(e.data).total));
es.addEventListener("rows", (e) => {
  points.push(...JSON.parse(e.data).rows);      // update the chart here as rows arrive
});
es.addEventListener("scene_error", (e) => console.warn(JSON.parse(e.data)));
es.addEventListener("done", () => es.close()); // close, or EventSource will reconnect
es.addEventListener("fatal", (e) => { console.error(JSON.parse(e.data).message); es.close(); });
```

Notes:
- **Errors:** bad dates (for example `start` after `end`) return HTTP 400; a missing or out-of-range
  parameter returns 422; when `ASF_API_MAX_JOBS` requests are already running the server returns 503
  (try again shortly); `/series` returns 502 if the upstream fails.
- **Cancelling:** closing the connection cancels the scenes not yet started.
- **CORS:** by default only the Vite dev server (`http://localhost:5173`) may call the API from a
  browser. Set `ASF_API_CORS` to your front-end's origin(s) for anything else.
- **Environment:** `EARTHDATA_TOKEN` (required), `ASF_CACHE_DIR` (cache folder, default `out`),
  `ASF_API_CORS` (comma-separated allowed origins), `ASF_API_MAX_JOBS` (concurrent requests,
  default 4).
- **Speed:** timing is network-bound. A NISAR scene takes about a minute and a half, an OPERA scene
  a few seconds, so the first request for a new point is slow and repeats are instant.
- **Scope:** the API serves the SAR backscatter only. IMERG rain and the analysis are CLI features
  for now.

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
- **Platform and product versions:** OPERA spans Sentinel-1A, 1B, 1C and 1D (`platform` column)
  and was reprocessed in 2025; NISAR is early in its mission, so calibration may change. Watch for
  steps in a long series at those boundaries.
- **Single patch:** results describe the window around the point (`--window-m`), not the surrounding area.
- **Rain is coarse.** An IMERG cell is about 11 km wide, far larger than the SAR window, so it is
  the regional rain, not the rain on the patch. Convective storms can miss the point entirely.
- **Mixed IMERG runs:** Final (gauge-adjusted) and Late (not) can differ systematically. Check
  `imerg_run` before comparing rain across 2025-09-30.

## Project layout

| Module (`src/asf_cleaner/`) | Role |
|---|---|
| `cli.py` | `asf-clean`: the command-line entry point |
| `api.py` | `asf-api`: the FastAPI / Server-Sent Events server |
| `analysis.py` | `asf-analyze`: trend, events, outliers, rain context, plots (Polars, lazy) |
| `search.py` | Finds scenes through the ASF search API, with automatic pagination |
| `stream.py` | Authenticated range reads of remote HDF5 and GeoTIFF files |
| `nisar.py`, `opera.py` | Cleaning of NISAR GCOV and OPERA RTC-S1 scenes |
| `imerg.py` | IMERG daily rain at a point (OPeNDAP) and the rain-before-acquisition columns |
| `workers.py`, `runner.py` | Per-scene work, parallel execution, retries |
| `cache.py` | The append-only scene cache shared by the CLI and the API |
| `harmonize.py`, `common.py` | The common output schema and shared cleaning helpers |
| `config.py` | Defaults, dataset selection, token handling |

## Development

    uv sync
    uv run pytest      # no network or token needed
