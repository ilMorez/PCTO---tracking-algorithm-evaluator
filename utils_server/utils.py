import os
import io
import csv
import json
import zipfile
import subprocess
from pathlib import Path
from collections import defaultdict

TMP_DIR = Path("server_tmp_videos")
TMP_DIR.mkdir(exist_ok=True)


def _to_h264(input_path: Path, output_path: Path) -> bool:
    result = subprocess.run(
        ["ffmpeg", "-y", "-i", str(input_path),
         "-vcodec", "libx264", "-an", "-crf", "23", "-preset", "fast",
         str(output_path)],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    return result.returncode == 0


def _csv_bytes(results: list, tracker_name: str, params: dict = None) -> bytes:
    buf = io.StringIO()
    if params:
        buf.write(f"# params: {json.dumps(params)}\n")
    writer = csv.DictWriter(buf, fieldnames=["frame", "track_id", "x1", "y1", "x2", "y2", "time"])
    writer.writeheader()
    writer.writerows(results)
    return buf.getvalue().encode()


def _make_zip(video_path: Path, csv_data: bytes, tracker_name: str) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(video_path, arcname=f"{tracker_name}_tracked.mp4")
        zf.writestr(f"{tracker_name}_tracking.csv", csv_data)
    buf.seek(0)
    return buf.read()


def _build_id_map(results_before: list, results_after: list) -> dict:
    """Costruisce {old_track_id: new_track_id} confrontando i risultati prima e dopo il remap.
    Usa il primo frame in cui appare ogni track_id per associare old → new tramite bbox.
    """
    before_by_frame = defaultdict(dict)
    for r in results_before:
        before_by_frame[r["frame"]][r["track_id"]] = (r["x1"], r["y1"], r["x2"], r["y2"])

    after_by_frame = defaultdict(dict)
    for r in results_after:
        after_by_frame[r["frame"]][r["track_id"]] = (r["x1"], r["y1"], r["x2"], r["y2"])

    id_map = {}
    for frame, before_tracks in before_by_frame.items():
        after_tracks = after_by_frame.get(frame, {})
        for old_id, bbox in before_tracks.items():
            if old_id in id_map:
                continue
            for new_id, bbox_after in after_tracks.items():
                if bbox == bbox_after:
                    id_map[old_id] = new_id
                    break

    return id_map