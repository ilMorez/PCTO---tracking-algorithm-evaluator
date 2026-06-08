import os
import json
import shutil
from pathlib import Path
from fastapi import FastAPI, File, UploadFile, Form, HTTPException

from custom_trackers import TRACKER_REGISTRY
from detection import Detector

app = FastAPI(title="Tracking Evaluator API Service")

TRACKERS_REQUIRING_VIDEO = {"DeepSORT", "BoT-SORT"}

TMP_DIR = Path("server_tmp_videos")
TMP_DIR.mkdir(exist_ok=True)


# ---------------------------------------------------------------------------
# /api/detect  —  YOLO detection su un video
# ---------------------------------------------------------------------------

@app.post("/api/detect")
async def process_detection(
    model_name:       str        = Form(...),          # es. "yolov8n.pt"
    yolo_params_json: str        = Form("{}"),          # JSON con conf, iou, imgsz, …
    video:            UploadFile = File(...),
):
    try:
        yolo_params = json.loads(yolo_params_json)
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Formato JSON non valido in 'yolo_params_json'.")

    saved_video_path = TMP_DIR / video.filename
    try:
        with saved_video_path.open("wb") as buffer:
            shutil.copyfileobj(video.file, buffer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Errore nel salvataggio del video: {str(e)}")

    tmp_json = TMP_DIR / f"{video.filename}.detections.json"

    try:
        print("===================================")
        print(f"Detection: modello={model_name} video={video.filename}")
        print("===================================")

        detector = Detector()
        detections = detector.run_detection(
            par_video_path=str(saved_video_path),
            model_name=model_name,
            yolo_params=yolo_params,
            par_output_json=str(tmp_json),
        )

        return {"status": "success", "detections": detections}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Errore durante la detection: {str(e)}")

    finally:
        if saved_video_path.exists():
            os.remove(saved_video_path)
        if tmp_json.exists():
            os.remove(tmp_json)


# ---------------------------------------------------------------------------
# /api/track  —  tracking su detections già calcolate
# ---------------------------------------------------------------------------

@app.post("/api/track")
async def process_tracking(
    tracker_name:        str        = Form(...),
    detections_json:     str        = Form(...),
    tracker_params_json: str        = Form("{}"),
    video:               UploadFile = File(...),
):
    if tracker_name not in TRACKER_REGISTRY:
        raise HTTPException(status_code=400, detail=f"Tracker '{tracker_name}' non supportato dal server.")

    try:
        detections_data = json.loads(detections_json)
        tracker_params  = json.loads(tracker_params_json)
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Formato JSON non valido in 'detections_json' o 'tracker_params_json'.")

    saved_video_path = TMP_DIR / video.filename
    try:
        with saved_video_path.open("wb") as buffer:
            shutil.copyfileobj(video.file, buffer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Errore nel salvataggio del video: {str(e)}")

    try:
        TrackerClass     = TRACKER_REGISTRY[tracker_name]
        tracker_instance = TrackerClass(**tracker_params)

        print("===================================")
        print(f"Tracker: {tracker_name}  Video: {video.filename}")
        print("===================================")

        results = tracker_instance.run(detections_data, par_video_path=str(saved_video_path))

        return {"status": "success", "tracking_results": results}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Errore durante l'elaborazione del tracker: {str(e)}")

    finally:
        if saved_video_path.exists():
            os.remove(saved_video_path)