import threading
from contextlib import contextmanager

import fsspec
import h5py
import rasterio
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

_local = threading.local()


def get_session() -> requests.Session:
    """One keep-alive session per thread/process, retrying 429/5xx with backoff.

    Sessions keep requests' behaviour of dropping the Authorization header on cross-host
    redirects, so the token is never forwarded to the CDN.
    """
    s = getattr(_local, "session", None)
    if s is None:
        s = requests.Session()
        retry = Retry(
            total=4,
            connect=1,  # connection failures are retried by the outer runner.retry as well
            read=1,
            status=4,
            backoff_factor=0.5,
            status_forcelist=(429, 500, 502, 503, 504),
            allowed_methods=frozenset({"GET", "HEAD"}),
            respect_retry_after_header=True,
        )
        adapter = HTTPAdapter(max_retries=retry, pool_connections=32, pool_maxsize=32)
        s.mount("https://", adapter)
        s.mount("http://", adapter)
        _local.session = s
    return s


def resolve_url(url: str, token: str) -> str:
    """Follow ASF's redirect (Earthdata auth) to the signed CloudFront URL."""
    r = get_session().get(
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
