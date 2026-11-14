import requests

from .config import ASF_SEARCH_URL

DATASETS = {
    "NISAR": {"dataset": "NISAR", "processingLevel": "GCOV"},
    "OPERA": {"dataset": "OPERA-S1", "processingLevel": "RTC"},
}


def _query(kind, lon, lat, start, end, max_results):
    params = {
        **DATASETS[kind],
        "intersectsWith": f"POINT({lon} {lat})",
        "start": start,
        "end": end,
        "output": "geojson",
        "maxResults": max_results,
    }
    r = requests.get(ASF_SEARCH_URL, params=params, timeout=60)
    r.raise_for_status()
    return r.json().get("features", [])


def search(kind, lon, lat, start, end, max_results=50):
    scenes = []
    for f in _query(kind, lon, lat, start, end, max_results):
        p = f["properties"]
        base = {
            "scene": p["sceneName"],
            "start_time": p["startTime"],
            "direction": p.get("flightDirection"),
            "track": p.get("pathNumber"),
            "platform": p.get("platform") or kind,
        }
        if kind == "NISAR":
            if str(p.get("url", "")).endswith(".h5"):
                scenes.append({**base, "burst_id": None, "url": p["url"]})
        else:
            # the .h5 is metadata only; the backscatter lives in per-pol GeoTIFFs
            tifs = {}
            for u in p.get("additionalUrls", []):
                tag = u.rsplit("_", 1)[-1].removesuffix(".tif")
                if u.endswith(".tif") and tag in ("VV", "VH", "HH", "HV", "mask"):
                    tifs[tag] = u
            if any(k != "mask" for k in tifs):
                scenes.append({**base, "burst_id": p.get("operaBurstID"), "tifs": tifs})
    return scenes
