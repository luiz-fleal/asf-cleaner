# asf-cleaner

Streams ASF NISAR GCOV (L-band) and OPERA-S1 RTC (Sentinel-1, C-band) at a point, masks invalid
pixels, converts to dB and writes a CSV/Parquet in one common schema.

    uv sync
    export EARTHDATA_TOKEN=...            # or put it in .env
    uv run asf-clean --dataset NISAR --lon -47.9 --lat -15.8
    uv run asf-clean --dataset OPERA --lon -47.9 --lat -15.8          # 2018 default range
    uv run asf-clean --dataset OPERA --start 2026-06-01T00:00:00Z --end 2026-11-13T23:59:59Z
    uv run asf-compare out/nisar_-47.9_-15.8.parquet out/opera_-47.9_-15.8.parquet

NISAR is read through the `.h5` (HTTP range reads). For OPERA the `.h5` is metadata only, so the
`_VV/_VH/_mask.tif` files are range-read instead. Defaults live in `src/asf_cleaner/config.py`.

## Reading the series: comparability caveats
- **C vs L band:** absolute dB differ (C saturates over dense canopy, L penetrates). Compare trends
  via `z_median` (z-score per sensor/burst/track/direction/polarization), not raw dB.
- **Polarization:** VV/VH are not HH/HV. `pol_class` maps both to `co` / `cross`; nothing is converted.
- **Geometry:** never mix tracks/directions in one series; `track` and `direction` are provided.
- **Footprint:** `--window-m` is a ground size; OPERA is 30 m, NISAR ~20 m (see `pixel_spacing_m`).
- **Normalization:** check the `normalization` column (OPERA = gamma0; NISAR read from the file, may be `unknown`).
- **Burst overlaps:** a point can fall in several bursts on one date; they are separate rows (`burst_id`).
- **2018 vs 2026:** no coincident data, so validate with the overlap run above (Sentinel-1 is still operating).
