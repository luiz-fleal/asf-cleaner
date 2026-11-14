"""Extension points for other ASF datasets (not implemented yet)."""


def sentinel1(*_a, **_k):
    raise NotImplementedError(
        "Sentinel-1: products are SAFE zips (not HDF5); needs a zip/range-based or "
        "COG-style reader. Implement here and register in cli.ADAPTERS."
    )


def opera(*_a, **_k):
    raise NotImplementedError(
        "OPERA (e.g. RTC-S1 / DISP-S1): per-burst GeoTIFF/HDF5 products; implement a "
        "reader here (h5py+fsspec for .h5, rasterio for .tif) and register in cli.ADAPTERS."
    )
