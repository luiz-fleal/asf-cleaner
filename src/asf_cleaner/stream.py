from contextlib import contextmanager

import fsspec
import h5py
import rasterio
import requests


def resolve_url(url: str, token: str) -> str:
    """Follow ASF's redirect (Earthdata auth) to the signed CloudFront URL.

    requests drops the Authorization header on cross-host redirects, so the
    token is never forwarded to the CDN.
    """
    r = requests.get(
        url,
        headers={"Authorization": f"Bearer {token}", "Range": "bytes=0-0"},
        stream=True,
        timeout=60,
    )
    if r.status_code in (401, 403):
        raise PermissionError(f"Earthdata auth failed ({r.status_code}); check EARTHDATA_TOKEN")
    r.raise_for_status()
    final = r.url
    r.close()
    return final


@contextmanager
def open_remote_h5(url: str, token: str, block_size: int = 4 * 1024 * 1024):
    """Open a remote HDF5 via HTTP range requests (nothing is fully downloaded)."""
    signed = resolve_url(url, token)
    with fsspec.open(signed, "rb", block_size=block_size, cache_type="readahead") as f:
        with h5py.File(f, "r") as h5:
            yield h5


@contextmanager
def open_remote_tif(url: str, token: str | None):
    """Open a GeoTIFF with GDAL range reads. token=None means a local path (tests)."""
    if token is None:
        with rasterio.open(url) as ds:
            yield ds
        return
    signed = resolve_url(url, token)
    with rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR", GDAL_HTTP_MAX_RETRY="3"):
        with rasterio.open(f"/vsicurl/{signed}") as ds:
            yield ds
