import os
import time
import uuid
import threading
from pathlib import Path

DETECTION_CACHE: dict[str, dict] = {}
DETECTION_CACHE_TTL = 3600  # 1 ora
_cache_lock = threading.Lock()


def _store_detections(detections: list, raw_video_path: Path | None) -> str:
    detection_id = uuid.uuid4().hex
    with _cache_lock:
        DETECTION_CACHE[detection_id] = {
            "detections": detections,
            "raw_video": str(raw_video_path) if raw_video_path else None,
            "created": time.time(),
        }
    return detection_id


def _get_detections(detection_id: str) -> list | None:
    with _cache_lock:
        entry = DETECTION_CACHE.get(detection_id)
        if not entry:
            return None
        if time.time() - entry["created"] > DETECTION_CACHE_TTL:
            DETECTION_CACHE.pop(detection_id, None)
            _cleanup_cached_video(entry)
            return None
        return entry["detections"]


def _get_cached_raw_video(detection_id: str) -> Path | None:
    with _cache_lock:
        entry = DETECTION_CACHE.get(detection_id)
        if not entry or not entry.get("raw_video"):
            return None
        p = Path(entry["raw_video"])
        return p if p.exists() else None


def _cleanup_cached_video(entry: dict):
    raw = entry.get("raw_video")
    if raw and Path(raw).exists():
        try:
            os.remove(raw)
        except Exception:
            pass