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

## Setup

    uv sync
    export EARTHDATA_TOKEN=...        # or put it in a .env file (keep .env out of git)

Create the token at https://urs.earthdata.nasa.gov ("Generate Token").

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

## Using the output
- **One series at a time:** filter to a single burst, track, direction and polarization before
  plotting or testing trends. Different tracks view the site from different angles and have
  different levels.
- **Prefer the median** over the centre pixel (speckle). Drop low `valid_fraction` rows.
- **Cross-pol** (HV/VH) is the better channel for vegetation change; co-pol (HH/VV) responds more
  to soil moisture and surface changes.
- **Deseasonalize** before testing for a trend (Theil-Sen / Mann-Kendall work well). Slow drifts
  under about 1 dB are hard to separate from noise and seasonal changes.

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

## Development

    uv run pytest      # no network or token needed
