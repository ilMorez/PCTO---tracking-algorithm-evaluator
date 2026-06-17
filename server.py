import os
import io
import csv
import json
import time
import uuid
import shutil
import zipfile
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import subprocess
from pathlib import Path
from collections import defaultdict

from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse, FileResponse

from custom_trackers import TRACKER_REGISTRY
from detection import Detector, filter_detections_by_class
from visualization import Visualizer, get_class_color

# Comando di avvio: uvicorn server:app --host 0.0.0.0 --port 8000

app = FastAPI(title="Tracking Evaluator API Service")

TMP_DIR = Path("server_tmp_videos")
TMP_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# Cache delle detection per evitare di ritrasmettere detections_json a ogni
# chiamata di tracking. Le voci scadono dopo DETECTION_CACHE_TTL secondi.
# ---------------------------------------------------------------------------
DETECTION_CACHE: dict[str, dict] = {}
DETECTION_CACHE_TTL = 3600  # 1 ora
_cache_lock = threading.Lock()


def _store_detections(detections: list, raw_video_path: Path | None) -> str:
    detection_id = uuid.uuid4().hex
    with _cache_lock:
        DETECTION_CACHE[detection_id] = {
            "detections": detections,
            "raw_video": str(raw_video_path) if raw_video_path else None,
            "created":    time.time(),
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


def _compute_crops_to_extract(results: list, W: int, H: int, tracker_name: str,
                               num_crops=5, margin=5, default_class="unknown"):
    """Calcola, per un tracker, quali (frame, classe, track_id, bbox) estrarre. Non apre il video."""
    valid_results = []
    for r in results:
        x1, y1, x2, y2 = r['x1'], r['y1'], r['x2'], r['y2']
        if x1 > margin and y1 > margin and x2 < (W - margin) and y2 < (H - margin):
            valid_results.append(r)

    tracks_by_class = defaultdict(lambda: defaultdict(list))
    for r in valid_results:
        cls_name = r.get('class_name') or r.get('label') or default_class
        tracks_by_class[cls_name][r['track_id']].append(r)

    frames_to_extract = defaultdict(list)
    for cls_name, tracks in tracks_by_class.items():
        for t_id, records in tracks.items():
            records = sorted(records, key=lambda x: x['frame'])
            if not records:
                continue
            step = max(1, len(records) // num_crops)
            selected_records = records[::step][:num_crops]
            for rec in selected_records:
                frames_to_extract[rec['frame']].append((tracker_name, cls_name, t_id, rec))

    return frames_to_extract


def _write_crops_single_pass(video_path: str, output_base_dir: Path, frames_to_extract: dict, W: int, H: int):
    """Una singola passata sul video: scrive i crop richiesti da uno o più tracker."""
    import cv2

    if not frames_to_extract:
        return

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return

    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx in frames_to_extract:
            for tracker_name, cls_name, t_id, det in frames_to_extract[frame_idx]:
                x1, y1, x2, y2 = int(det['x1']), int(det['y1']), int(det['x2']), int(det['y2'])
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(W, x2), min(H, y2)
                crop = frame[y1:y2, x1:x2]
                if crop.size > 0:
                    folder = output_base_dir / str(tracker_name) / str(cls_name) / str(t_id)
                    folder.mkdir(parents=True, exist_ok=True)
                    import cv2 as _cv2
                    _cv2.imwrite(str(folder / f"frame_{frame_idx}.jpg"), crop)

        frame_idx += 1

    cap.release()


def extract_track_crops(video_path: str, results: list, output_base_dir: Path, tracker_name: str,
                         num_crops=5, margin=5, default_class="unknown"):
    """
    Estrae i crop delle bbox dal video originale per ogni track_id unico (singolo tracker).
    Struttura finale: output_base_dir / tracker_name / nome_classe / track_id / frame_X.jpg
    """
    import cv2

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    frames_to_extract = _compute_crops_to_extract(
        results, W, H, tracker_name, num_crops=num_crops, margin=margin, default_class=default_class
    )
    _write_crops_single_pass(video_path, output_base_dir, frames_to_extract, W, H)


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
    h264_video_path  = TMP_DIR / f"{uuid.uuid4().hex}_raw_h264.mp4"

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
        cached_video_path = h264_video_path if (converted and h264_video_path.exists()) else None

        detection_id = _store_detections(detections, cached_video_path)

        return JSONResponse({
            "status":         "success",
            "detections":     detections,
            "detection_id":   detection_id,
            "target_classes": target_classes,
            "has_raw_video":  cached_video_path is not None,
        })

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Errore durante la detection: {e}")

    finally:
        for p in (saved_video_path, tmp_json, raw_video_path):
            if p.exists():
                os.remove(p)


@app.get("/api/detect/{detection_id}/video")
async def get_detect_video(detection_id: str):
    """Scarica in streaming il video con i rilevamenti grezzi associato a un detection_id."""
    video_path = _get_cached_raw_video(detection_id)
    if video_path is None:
        raise HTTPException(status_code=404, detail="Video non trovato o scaduto.")
    return FileResponse(str(video_path), media_type="video/mp4", filename="raw_detections_h264.mp4")


# ---------------------------------------------------------------------------
# /api/track — tracker singolo su una classe
# ---------------------------------------------------------------------------

@app.post("/api/track")
async def process_tracking(
    tracker_name:        str        = Form(...),
    detections_json:     str | None = Form(None),
    detection_id:        str | None = Form(None),
    tracker_params_json: str        = Form("{}"),
    target_class:        str        = Form(""),
    extract_crops:       bool       = Form(False),
    video:               UploadFile = File(...),
):
    if tracker_name not in TRACKER_REGISTRY:
        raise HTTPException(status_code=400, detail=f"Tracker '{tracker_name}' non supportato.")

    try:
        tracker_params = json.loads(tracker_params_json)
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Formato JSON non valido (tracker_params_json).")

    detections_data = None
    if detection_id:
        detections_data = _get_detections(detection_id)
        if detections_data is None:
            raise HTTPException(status_code=404, detail="detection_id non trovato o scaduto. Rifare la detection.")
    elif detections_json:
        try:
            detections_data = json.loads(detections_json)
        except json.JSONDecodeError:
            raise HTTPException(status_code=400, detail="Formato JSON non valido (detections_json).")
    else:
        raise HTTPException(status_code=400, detail="Specificare detection_id o detections_json.")

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
        results_before   = tracker_instance.run(detections_data, par_video_path=str(saved_video_path))
        results          = tracker_instance._remap_track_ids(results_before)

        embeddings_path = None
        if hasattr(tracker_instance, 'save_embeddings'):
            video_id = Path(video.filename).stem
            # Costruisce il mapping old_id → new_id per allineare gli embedding al remap
            id_map = _build_id_map(results_before, results)
            if hasattr(tracker_instance, 'remap_embeddings'):
                tracker_instance.remap_embeddings(id_map)
            embeddings_npz = TMP_DIR / f"{tracker_name}_{video_id}_embeddings.npz"
            tracker_instance.save_embeddings(
                video_id=f"{video_id}__{tracker_name}__{target_class or 'all'}",
                out_path=str(embeddings_npz)
            )
            embeddings_path = embeddings_npz

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

        if extract_crops:
            extract_track_crops(
                video_path=str(saved_video_path),
                results=results,
                output_base_dir=crops_dir,
                tracker_name=tracker_name,
                default_class=target_class or "unknown"
            )

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

            if embeddings_path and embeddings_path.exists():
                zf.write(embeddings_path, arcname=f"{tracker_name}_{target_class or 'all'}_embeddings.npz")

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
        for p in (saved_video_path, tracked_video_path, h264_video_path, tmp_csv, embeddings_path):
            if p and p.exists():
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
    detections_json:     str | None = Form(None),
    detection_id:        str | None = Form(None),
    extract_crops:       bool       = Form(False),
    video:               UploadFile = File(...),
):
    try:
        assignments = json.loads(assignments_json)
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Formato JSON non valido (assignments_json).")

    detections_data = None
    if detection_id:
        detections_data = _get_detections(detection_id)
        if detections_data is None:
            raise HTTPException(status_code=404, detail="detection_id non trovato o scaduto. Rifare la detection.")
    elif detections_json:
        try:
            detections_data = json.loads(detections_json)
        except json.JSONDecodeError:
            raise HTTPException(status_code=400, detail="Formato JSON non valido (detections_json).")
    else:
        raise HTTPException(status_code=400, detail="Specificare detection_id o detections_json.")

    if not assignments:
        raise HTTPException(status_code=400, detail="assignments_json non può essere vuota.")

    for a in assignments:
        if a.get("tracker_name") not in TRACKER_REGISTRY:
            raise HTTPException(status_code=400, detail=f"Tracker '{a.get('tracker_name')}' non supportato.")

    saved_video_path  = TMP_DIR / video.filename
    merged_video_path = TMP_DIR / f"{video.filename}.multi.mp4"
    h264_merged_path  = TMP_DIR / f"{video.filename}.multi_h264.mp4"
    crops_dir         = TMP_DIR / f"{video.filename}_multi_crops"

    try:
        with saved_video_path.open("wb") as buffer:
            shutil.copyfileobj(video.file, buffer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Errore salvataggio video: {e}")

    tmp_files = [saved_video_path, merged_video_path, h264_merged_path]

    try:
        track_layers = []
        csv_entries  = {}
        all_frames_to_extract = defaultdict(list)
        cap_w = cap_h = None
        if extract_crops:
            import cv2
            _cap = cv2.VideoCapture(str(saved_video_path))
            cap_w = int(_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            cap_h = int(_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            _cap.release()

        def _run_single_assignment(idx, assignment):
            tracker_name   = assignment["tracker_name"]
            target_class   = assignment["target_class"]
            tracker_params = assignment.get("tracker_params", {})

            print(f"=== Multi-track [{idx+1}/{len(assignments)}]: {tracker_name} → {target_class} ===")

            filtered_dets    = filter_detections_by_class(detections_data, target_class)
            TrackerClass     = TRACKER_REGISTRY[tracker_name]
            tracker_instance = TrackerClass(**tracker_params)
            results_before   = tracker_instance.run(filtered_dets, par_video_path=str(saved_video_path))
            results          = tracker_instance._remap_track_ids(results_before)

            embeddings_npz = None
            if hasattr(tracker_instance, 'save_embeddings'):
                video_id = Path(saved_video_path.name).stem
                id_map = _build_id_map(results_before, results)
                if hasattr(tracker_instance, 'remap_embeddings'):
                    tracker_instance.remap_embeddings(id_map)
                embeddings_npz = TMP_DIR / f"multi_{idx}_{tracker_name}_embeddings.npz"
                tracker_instance.save_embeddings(
                    video_id=f"{video_id}__{tracker_name}__{target_class}",
                    out_path=str(embeddings_npz)
                )

            csv_data = _csv_bytes(results, tracker_name, params=tracker_params)
            tmp_csv  = TMP_DIR / f"multi_{idx}_{tracker_name}.csv"
            tmp_csv.write_bytes(csv_data)

            fte = None
            if extract_crops:
                fte = _compute_crops_to_extract(
                    results, cap_w, cap_h, tracker_name,
                    default_class=target_class or "unknown"
                )

            return {
                "idx": idx, "tracker_name": tracker_name, "target_class": target_class,
                "csv_key": f"{target_class}_{tracker_name.lower()}", "csv_data": csv_data,
                "tmp_csv": tmp_csv, "fte": fte,
                "embeddings_npz": embeddings_npz,
            }

        max_workers = min(len(assignments), os.cpu_count() or 4)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(_run_single_assignment, idx, a): idx
                       for idx, a in enumerate(assignments)}
            results_by_idx = {}
            for future in as_completed(futures):
                r = future.result()
                results_by_idx[r["idx"]] = r

        for idx in range(len(assignments)):
            r = results_by_idx[idx]
            csv_entries[r["csv_key"]] = r["csv_data"]
            tmp_files.append(r["tmp_csv"])

            if r.get("embeddings_npz"):
                tmp_files.append(r["embeddings_npz"])

            if extract_crops and r["fte"]:
                for frame_idx, items in r["fte"].items():
                    all_frames_to_extract[frame_idx].extend(items)

            color = get_class_color(idx)
            track_layers.append({
                "csv_path": str(r["tmp_csv"]),
                "color":    color,
                "label":    r["target_class"],
            })

        if extract_crops and all_frames_to_extract:
            _write_crops_single_pass(str(saved_video_path), crops_dir, all_frames_to_extract, cap_w, cap_h)

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

            if extract_crops and crops_dir.exists():
                for root, _, files in os.walk(crops_dir):
                    for file in files:
                        file_path = Path(root) / file
                        arc_name = f"crops/{file_path.relative_to(crops_dir)}"
                        zf.write(file_path, arcname=arc_name)

            for idx in range(len(assignments)):
                r   = results_by_idx[idx]
                npz = r.get("embeddings_npz")
                if npz and npz.exists():
                    zf.write(npz, arcname=f"{r['tracker_name']}_{r['target_class']}_embeddings.npz")

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
        if crops_dir.exists():
            shutil.rmtree(crops_dir, ignore_errors=True)
