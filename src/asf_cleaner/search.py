import requests

from .config import ASF_SEARCH_URL


def search(dataset, lon, lat, start, end, max_results=50, processing_level="GCOV"):
    params = {
        "dataset": dataset,
        "processingLevel": processing_level,
        "intersectsWith": f"POINT({lon} {lat})",
        "start": start,
        "end": end,
        "output": "geojson",
        "maxResults": max_results,
    }
    r = requests.get(ASF_SEARCH_URL, params=params, timeout=60)
    r.raise_for_status()
    scenes = []
    for f in r.json().get("features", []):
        p = f["properties"]
        if not str(p.get("url", "")).endswith(".h5"):
            continue
        scenes.append(
            {
                "scene": p["sceneName"],
                "start_time": p["startTime"],
                "flight_direction": p.get("flightDirection"),
                "path": p.get("pathNumber"),
                "url": p["url"],
            }
        )
    return scenes
