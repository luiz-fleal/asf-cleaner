"""A small thread-safe cache whose entries expire, for slow lookups that clients repeat."""
import threading
import time


class TTLCache:
    def __init__(self, maxsize: int = 256):
        self.maxsize = maxsize
        self._data: dict = {}
        self._lock = threading.Lock()

    def get(self, key, ttl: float):
        with self._lock:
            hit = self._data.get(key)
            if hit and time.monotonic() - hit[0] < ttl:
                return hit[1]
        return None

    def put(self, key, value) -> None:
        with self._lock:
            self._data[key] = (time.monotonic(), value)
            while len(self._data) > self.maxsize:
                self._data.pop(next(iter(self._data)))

    def clear(self) -> None:
        with self._lock:
            self._data.clear()
