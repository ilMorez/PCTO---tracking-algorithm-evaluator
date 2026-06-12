import os
import io
import csv
import json
import shutil
import zipfile
import subprocess
from pathlib import Path
from collections import defaultdict

from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse

from custom_trackers import TRACKER_REGISTRY
from detection import Detector, filter_detections_by_class
from visualization import Visualizer, get_class_color

# Comando di avvio: uvicorn server:app --host 0.0.0.0 --port 8000

app = FastAPI(title="Tracking Evaluator API Service")

TMP_DIR = Path("server_tmp_videos")
TMP_DIR.mkdir(exist_ok=True)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

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


def extract_track_crops(video_path: str, results: list, output_base_dir: Path, num_crops=5, margin=5, default_class="unknown"):
    """
    Estrae i crop delle bbox dal video originale per ogni track_id unico.
    Scarta le bbox che toccano i bordi dell'inquadratura ed estrae ~4 frame equispaziati.
    Struttura finale: output_base_dir / nome_classe / track_id / frame_X.jpg
    """
    import cv2

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return
    
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # 1. Filtra ed evita le bbox che toccano i bordi dell'inquadratura (con un piccolo margine)
    valid_results = []
    for r in results:
        x1, y1, x2, y2 = r['x1'], r['y1'], r['x2'], r['y2']
        if x1 > margin and y1 > margin and x2 < (W - margin) and y2 < (H - margin):
            valid_results.append(r)
            
    # 2. Raggruppa per Classe e poi per Track ID
    tracks_by_class = defaultdict(lambda: defaultdict(list))
    for r in valid_results:
        cls_name = r.get('class_name') or r.get('label') or default_class
        tracks_by_class[cls_name][r['track_id']].append(r)
        
    # 3. Seleziona fino a `num_crops` frame equispaziati nel tempo per ogni oggetto
    frames_to_extract = defaultdict(list)
    for cls_name, tracks in tracks_by_class.items():
        for t_id, records in tracks.items():
            records = sorted(records, key=lambda x: x['frame'])
            if not records:
                continue
                
            step = max(1, len(records) // num_crops)
            selected_records = records[::step][:num_crops] 
            
            for rec in selected_records:
                frames_to_extract[rec['frame']].append((cls_name, t_id, rec))
                
    if not frames_to_extract:
        cap.release()
        return

    # 4. Scorri il video originale ed estrai i crop puliti
    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        if frame_idx in frames_to_extract:
            for cls_name, t_id, det in frames_to_extract[frame_idx]:
                x1, y1, x2, y2 = int(det['x1']), int(det['y1']), int(det['x2']), int(det['y2'])
                
                # Clip per sicurezza coordinate
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(W, x2), min(H, y2)
                
                crop = frame[y1:y2, x1:x2]
                if crop.size > 0:
                    folder = output_base_dir / str(cls_name) / str(t_id)
                    folder.mkdir(parents=True, exist_ok=True)
                    cv2.imwrite(str(folder / f"frame_{frame_idx}.jpg"), crop)
                    
        frame_idx += 1
        
    cap.release()


# ---------------------------------------------------------------------------
# /api/detect
# ---------------------------------------------------------------------------

@app.post("/api/detect")
async def process_detection(
    model_name:          str        = Form(...),
    yolo_params_json:    str        = Form("{}"),
    target_classes_json: str        = Form('["car"]'),   
    video:               UploadFile = File(...),
):
    try:
        yolo_params    = json.loads(yolo_params_json)
        target_classes = json.loads(target_classes_json)
        if not isinstance(target_classes, list) or not target_classes:
            raise ValueError
    except Exception:
        raise HTTPException(status_code=400, detail="Parametri JSON non validi.")

    saved_video_path = TMP_DIR / video.filename
    tmp_json         = TMP_DIR / f"{video.filename}.detections.json"
    raw_video_path   = TMP_DIR / f"{video.filename}.raw.mp4"
    h264_video_path  = TMP_DIR / f"{video.filename}.raw_h264.mp4"

    try:
        with saved_video_path.open("wb") as buffer:
            shutil.copyfileobj(video.file, buffer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Errore salvataggio video: {e}")

    try:
        print(f"=== Detection: modello={model_name} classi={target_classes} video={video.filename} ===")

        detector   = Detector()
        detections = detector.run_detection(
            par_video_path=str(saved_video_path),
            model_name=model_name,
            yolo_params=yolo_params,
            target_classes=target_classes,
            par_output_json=str(tmp_json),
        )

        visualizer = Visualizer(par_output_dir=str(TMP_DIR))
        visualizer.draw_raw_detections(detections, raw_video_path.name, str(saved_video_path))

        converted = _to_h264(raw_video_path, h264_video_path)
        video_b64 = None
        if converted and h264_video_path.exists():
            import base64
            video_b64 = base64.b64encode(h264_video_path.read_bytes()).decode()

        return JSONResponse({
            "status":          "success",
            "detections":       detections,
            "target_classes":  target_classes,
            "yolo_video":      video_b64,
        })

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Errore durante la detection: {e}")

    finally:
        for p in (saved_video_path, tmp_json, raw_video_path, h264_video_path):
            if p.exists():
                os.remove(p)


# ---------------------------------------------------------------------------
# /api/track — tracker singolo su una classe
# ---------------------------------------------------------------------------

@app.post("/api/track")
async def process_tracking(
    tracker_name:        str        = Form(...),
    detections_json:     str        = Form(...),
    tracker_params_json: str        = Form("{}"),
    target_class:        str        = Form(""),       
    extract_crops:       bool       = Form(False), 
    video:               UploadFile = File(...),
):
    if tracker_name not in TRACKER_REGISTRY:
        raise HTTPException(status_code=400, detail=f"Tracker '{tracker_name}' non supportato.")

    try:
        detections_data = json.loads(detections_json)
        tracker_params  = json.loads(tracker_params_json)
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Formato JSON non valido.")

    if target_class:
        detections_data = filter_detections_by_class(detections_data, target_class)

    saved_video_path   = TMP_DIR / video.filename
    tracked_video_path = TMP_DIR / f"{video.filename}.{tracker_name}.mp4"
    h264_video_path    = TMP_DIR / f"{video.filename}.{tracker_name}_h264.mp4"
    tmp_csv            = TMP_DIR / f"{tracker_name}_tmp.csv"
    crops_dir          = TMP_DIR / f"{video.filename}_crops"

    try:
        with saved_video_path.open("wb") as buffer:
            shutil.copyfileobj(video.file, buffer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Errore salvataggio video: {e}")

    try:
        print(f"=== Tracker: {tracker_name} | Classe: {target_class or 'all'} | Video: {video.filename} ===")

        TrackerClass     = TRACKER_REGISTRY[tracker_name]
        tracker_instance = TrackerClass(**tracker_params)
        results          = tracker_instance.run(detections_data, par_video_path=str(saved_video_path))
        results = tracker_instance._remap_track_ids(results)

        csv_data = _csv_bytes(results, tracker_name, params=tracker_params)
        tmp_csv.write_bytes(csv_data)

        visualizer = Visualizer(par_output_dir=str(TMP_DIR))
        visualizer.draw_tracks(
            str(tmp_csv),
            tracked_video_path.name,
            str(saved_video_path),
            label=target_class,
        )

        _to_h264(tracked_video_path, h264_video_path)
        video_src = h264_video_path if h264_video_path.exists() else tracked_video_path
        if not video_src.exists():
            raise RuntimeError("Generazione video tracciato fallita.")

        # Esegui l'estrazione dei crop se richiesto (dal video sorgente originale, pulito!)
        if extract_crops:
            extract_track_crops(
                video_path=str(saved_video_path),
                results=results,
                output_base_dir=crops_dir,
                default_class=target_class or "unknown"
            )

        # Costruzione dinamica dello ZIP per supportare la cartella dei crop
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.write(video_src, arcname=f"{tracker_name.lower()}_tracked.mp4")
            zf.writestr(f"{tracker_name.lower()}_tracking.csv", csv_data)
            
            if extract_crops and crops_dir.exists():
                for root, _, files in os.walk(crops_dir):
                    for file in files:
                        file_path = Path(root) / file
                        arc_name = f"crops/{file_path.relative_to(crops_dir)}"
                        zf.write(file_path, arcname=arc_name)
        buf.seek(0)
        zip_bytes = buf.read()

        return StreamingResponse(
            io.BytesIO(zip_bytes),
            media_type="application/zip",
            headers={"Content-Disposition": f'attachment; filename="{tracker_name}_results.zip"'},
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Errore durante il tracking: {e}")

    finally:
        for p in (saved_video_path, tracked_video_path, h264_video_path, tmp_csv):
            if p.exists():
                try:
                    os.remove(p)
                except Exception:
                    pass
        if crops_dir.exists():
            shutil.rmtree(crops_dir, ignore_errors=True)


# ---------------------------------------------------------------------------
# /api/track_multi
# ---------------------------------------------------------------------------

@app.post("/api/track_multi")
async def process_tracking_multi(
    assignments_json:    str        = Form(...),
    detections_json:     str        = Form(...),
    video:               UploadFile = File(...),
):
    try:
        assignments     = json.loads(assignments_json)
        detections_data = json.loads(detections_json)
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Formato JSON non valido.")

    if not assignments:
        raise HTTPException(status_code=400, detail="assignments_json non può essere vuota.")

    for a in assignments:
        if a.get("tracker_name") not in TRACKER_REGISTRY:
            raise HTTPException(status_code=400, detail=f"Tracker '{a.get('tracker_name')}' non supportato.")

    saved_video_path = TMP_DIR / video.filename
    merged_video_path = TMP_DIR / f"{video.filename}.multi.mp4"
    h264_merged_path  = TMP_DIR / f"{video.filename}.multi_h264.mp4"

    try:
        with saved_video_path.open("wb") as buffer:
            shutil.copyfileobj(video.file, buffer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Errore salvataggio video: {e}")

    tmp_files = [saved_video_path, merged_video_path, h264_merged_path]

    try:
        track_layers = []   
        csv_entries  = {}   

        for idx, assignment in enumerate(assignments):
            tracker_name   = assignment["tracker_name"]
            target_class   = assignment["target_class"]
            tracker_params = assignment.get("tracker_params", {})

            print(f"=== Multi-track [{idx+1}/{len(assignments)}]: {tracker_name} → {target_class} ===")

            filtered_dets = filter_detections_by_class(detections_data, target_class)

            TrackerClass     = TRACKER_REGISTRY[tracker_name]
            tracker_instance = TrackerClass(**tracker_params)
            results          = tracker_instance.run(filtered_dets, par_video_path=str(saved_video_path))
            results = tracker_instance._remap_track_ids(results)

            csv_data = _csv_bytes(results, tracker_name, params=tracker_params)
            csv_entries[f"{target_class}_{tracker_name.lower()}"] = csv_data

            tmp_csv = TMP_DIR / f"multi_{idx}_{tracker_name}.csv"
            tmp_csv.write_bytes(csv_data)
            tmp_files.append(tmp_csv)

            color = get_class_color(idx)
            track_layers.append({
                "csv_path": str(tmp_csv),
                "color":    color,
                "label":    target_class,
            })

        visualizer = Visualizer(par_output_dir=str(TMP_DIR))
        visualizer.draw_multi_class_tracks(
            track_layers=track_layers,
            par_output_video_name=merged_video_path.name,
            par_video_path=str(saved_video_path),
        )

        _to_h264(merged_video_path, h264_merged_path)
        video_src = h264_merged_path if h264_merged_path.exists() else merged_video_path
        if not video_src.exists():
            raise RuntimeError("Generazione video merged fallita.")

        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.write(video_src, arcname="multi_tracked.mp4")
            for name, data in csv_entries.items():
                zf.writestr(f"{name}_tracking.csv", data)
        buf.seek(0)

        return StreamingResponse(
            buf,
            media_type="application/zip",
            headers={"Content-Disposition": 'attachment; filename="multi_tracking_results.zip"'},
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Errore durante il multi-tracking: {e}")

    finally:
        for p in tmp_files:
            if p.exists():
                try:
                    os.remove(p)
                except Exception:
                    pass
