# asf-cleaner

Searches ASF for NISAR GCOV scenes at a point, streams each `.h5` via HTTP range
requests (no full download), masks invalid pixels, converts to dB, and writes a CSV/Parquet.

    uv sync
    export EARTHDATA_TOKEN=...   # or put it in .env
    uv run asf-clean --lon -47.9 --lat -15.8

Defaults live in `src/asf_cleaner/config.py`; every one has a CLI flag (`--help`).
Sentinel-1 / OPERA are stubs in `others.py`.
