import os
import io
import csv
import json
import shutil
import zipfile
import subprocess
from pathlib import Path

from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse

from custom_trackers import TRACKER_REGISTRY
from detection import Detector
from visualization import Visualizer

app = FastAPI(title="Tracking Evaluator API Service")

TMP_DIR = Path("server_tmp_videos")
TMP_DIR.mkdir(exist_ok=True)

def _to_h264(input_path: Path, output_path: Path) -> bool:
    """Converte un video in H264 via ffmpeg. Ritorna True se OK."""
    result = subprocess.run(
        [
            "ffmpeg", "-y",
            "-i", str(input_path),
            "-vcodec", "libx264",
            "-an",               # niente audio
            "-crf", "23",
            "-preset", "fast",
            str(output_path),
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return result.returncode == 0


def _csv_bytes(results: list, tracker_name: str, params: dict = None) -> bytes:
    """Serializza i risultati di tracking in bytes CSV."""
    buf = io.StringIO()
    if params:
        buf.write(f"# params: {json.dumps(params)}\n")
    writer = csv.DictWriter(buf, fieldnames=["frame", "track_id", "x1", "y1", "x2", "y2", "time"])
    writer.writeheader()
    writer.writerows(results)
    return buf.getvalue().encode()


def _make_zip(video_path: Path, csv_data: bytes, tracker_name: str) -> bytes:
    """Crea uno zip in memoria con video H264 + CSV."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(video_path, arcname=f"{tracker_name}_tracked.mp4")
        zf.writestr(f"{tracker_name}_tracking.csv", csv_data)
    buf.seek(0)
    return buf.read()


@app.post("/api/detect")
async def process_detection(
    model_name:       str        = Form(...),
    yolo_params_json: str        = Form("{}"),
    video:            UploadFile = File(...),
):
    try:
        yolo_params = json.loads(yolo_params_json)
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Formato JSON non valido in 'yolo_params_json'.")

    saved_video_path = TMP_DIR / video.filename
    tmp_json         = TMP_DIR / f"{video.filename}.detections.json"
    raw_video_path   = TMP_DIR / f"{video.filename}.raw.mp4"
    h264_video_path  = TMP_DIR / f"{video.filename}.raw_h264.mp4"

    try:
        with saved_video_path.open("wb") as buffer:
            shutil.copyfileobj(video.file, buffer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Errore nel salvataggio del video: {e}")

    try:
        print(f"=== Detection: modello={model_name} video={video.filename} ===")

        detector   = Detector()
        detections = detector.run_detection(
            par_video_path=str(saved_video_path),
            model_name=model_name,
            yolo_params=yolo_params,
            par_output_json=str(tmp_json),
        )

        # --- video YOLO annotato ---
        visualizer = Visualizer(par_output_dir=str(TMP_DIR))
        visualizer.draw_raw_detections(
            detections,
            raw_video_path.name,
            str(saved_video_path),
        )

        converted = _to_h264(raw_video_path, h264_video_path)
        video_b64 = None
        if converted and h264_video_path.exists():
            import base64
            video_b64 = base64.b64encode(h264_video_path.read_bytes()).decode()

        return JSONResponse({
            "status":     "success",
            "detections": detections,
            "yolo_video": video_b64,   # base64 del video H264, None se errore
        })

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Errore durante la detection: {e}")

    finally:
        for p in (saved_video_path, tmp_json, raw_video_path, h264_video_path):
            if p.exists():
                os.remove(p)

@app.post("/api/track")
async def process_tracking(
    tracker_name:        str        = Form(...),
    detections_json:     str        = Form(...),
    tracker_params_json: str        = Form("{}"),
    video:               UploadFile = File(...),
):
    if tracker_name not in TRACKER_REGISTRY:
        raise HTTPException(status_code=400, detail=f"Tracker '{tracker_name}' non supportato.")

    try:
        detections_data = json.loads(detections_json)
        tracker_params  = json.loads(tracker_params_json)
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Formato JSON non valido.")

    saved_video_path   = TMP_DIR / video.filename
    tracked_video_path = TMP_DIR / f"{video.filename}.{tracker_name}.mp4"
    h264_video_path    = TMP_DIR / f"{video.filename}.{tracker_name}_h264.mp4"

    try:
        with saved_video_path.open("wb") as buffer:
            shutil.copyfileobj(video.file, buffer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Errore nel salvataggio del video: {e}")

    try:
        print(f"=== Tracker: {tracker_name}  Video: {video.filename} ===")

        TrackerClass     = TRACKER_REGISTRY[tracker_name]
        tracker_instance = TrackerClass(**tracker_params)

        results = tracker_instance.run(detections_data, par_video_path=str(saved_video_path))

        # --- CSV ---
        csv_data = _csv_bytes(results, tracker_name, params=tracker_params)

        # --- video tracciato ---
        # Salva CSV temporaneo per Visualizer (che legge da file)
        tmp_csv = TMP_DIR / f"{tracker_name}_tmp.csv"
        tmp_csv.write_bytes(csv_data)

        visualizer = Visualizer(par_output_dir=str(TMP_DIR))
        visualizer.draw_tracks(
            str(tmp_csv),
            tracked_video_path.name,
            str(saved_video_path),
        )

        _to_h264(tracked_video_path, h264_video_path)

        video_src = h264_video_path if h264_video_path.exists() else tracked_video_path
        if not video_src.exists():
            raise RuntimeError("Generazione video tracciato fallita.")

        zip_bytes = _make_zip(video_src, csv_data, tracker_name.lower())

        return StreamingResponse(
            io.BytesIO(zip_bytes),
            media_type="application/zip",
            headers={"Content-Disposition": f'attachment; filename="{tracker_name}_results.zip"'},
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Errore durante il tracking: {e}")

    finally:
        for p in (saved_video_path, tracked_video_path, h264_video_path, tmp_csv if 'tmp_csv' in dir() else Path("/dev/null")):
            if p.exists():
                try:
                    os.remove(p)
                except Exception:
                    pass
