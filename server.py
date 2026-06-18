import os
import io
import json
import uuid
import shutil
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from collections import defaultdict

from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse, FileResponse

from custom_trackers import TRACKER_REGISTRY
from detection import Detector, filter_detections_by_class
from visualization import Visualizer, get_class_color

# Moduli interni
from utils_server.cache_manager import _store_detections, _get_detections, _get_cached_raw_video
from utils_server.utils import TMP_DIR, _to_h264, _csv_bytes, _build_id_map
from utils_server.crop_processor import (
    extract_track_crops,
    _compute_crops_to_extract,
    _write_crops_single_pass,
    _classify_color_hsv,
    _run_yolo_on_crop,
)

app = FastAPI(title="Tracking Evaluator API Service")


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


# ---------------------------------------------------------------------------
# /api/analyze_crops  — post-processing YOLO-E sui crop estratti dal tracker
# ---------------------------------------------------------------------------

@app.post("/api/analyze_crops")
async def analyze_crops(
    crops_zip:        UploadFile = File(...),
    model_name:       str        = Form("yoloe-26s-seg.pt"),
    attributes_json:  str        = Form('["color", "vehicle_type"]'),
    conf_threshold:   float      = Form(0.25),
    top_n_crops:      int        = Form(5),
):
    try:
        attributes = json.loads(attributes_json)
        if not isinstance(attributes, list) or not attributes:
            raise ValueError
    except Exception:
        raise HTTPException(status_code=400, detail="attributes_json non valido.")

    tmp_zip_path  = TMP_DIR / f"{uuid.uuid4().hex}_crops.zip"
    tmp_crops_dir = TMP_DIR / f"{uuid.uuid4().hex}_crops_extracted"

    try:
        with tmp_zip_path.open("wb") as f:
            shutil.copyfileobj(crops_zip.file, f)
        with zipfile.ZipFile(tmp_zip_path, "r") as zf:
            zf.extractall(tmp_crops_dir)

        import cv2
        import numpy as np

        track_crops: dict[str, list[Path]] = {}
        track_class: dict[str, str]        = {}

        for img_path in sorted(tmp_crops_dir.rglob("*.jpg")):
            parts = img_path.parts
            if len(parts) >= 3:
                track_id   = parts[-2]
                class_name = parts[-3]
                key        = f"{class_name}__{track_id}"
                track_crops.setdefault(key, []).append(img_path)
                track_class[key] = class_name

        if not track_crops:
            raise HTTPException(status_code=400, detail="Nessun crop trovato nello ZIP.")

        model = None
        yolo_attrs = [a for a in attributes if a != "color"]
        
        if yolo_attrs:
            from ultralytics import YOLO
            model = YOLO(model_name)
            
            target_classes = []
            for attr in yolo_attrs:
                target_classes.append(attr)
            
            target_classes = list(set(target_classes))
            if hasattr(model, 'set_classes'):
                model.set_classes(target_classes)

        results_out = {}

        for key, img_paths in track_crops.items():
            class_name = track_class[key]
            track_id   = key.split("__", -1)[-1]
            selected   = img_paths[:top_n_crops]

            per_frame_results = []
            aggregated: dict[str, list] = {attr: [] for attr in attributes}

            for img_path in selected:
                frame = cv2.imread(str(img_path))
                if frame is None:
                    continue

                frame_attrs: dict[str, object] = {"frame": img_path.name}

                if "color" in attributes:
                    color_label, color_conf = _classify_color_hsv(frame)
                    frame_attrs["color"] = color_label
                    frame_attrs["color_confidence"] = round(color_conf, 3)
                    aggregated["color"].append((color_label, color_conf))

                if model is not None:
                    yolo_preds = _run_yolo_on_crop(model, frame, conf_threshold)

                    if "vehicle_type" in yolo_attrs:
                        best = max(yolo_preds, key=lambda x: x["confidence"]) if yolo_preds else None
                        frame_attrs["vehicle_type"] = best["class_name"] if best else None
                        frame_attrs["vehicle_type_confidence"] = round(best["confidence"], 3) if best else 0.0
                        if best:
                            aggregated["vehicle_type"].append((best["class_name"], best["confidence"]))

                    for attr in yolo_attrs:
                        if attr == "vehicle_type":
                            continue
                        match = next((p for p in yolo_preds if p["class_name"].lower() == attr.lower()), None)
                        frame_attrs[attr] = match["class_name"] if match else None
                        frame_attrs[f"{attr}_confidence"] = round(match["confidence"], 3) if match else 0.0
                        if match:
                            aggregated[attr].append((match["class_name"], match["confidence"]))

                per_frame_results.append(frame_attrs)

            summary = {
                "track_id": track_id,
                "class": class_name,
                "num_frames_analyzed": len(per_frame_results),
                "per_frame": per_frame_results
            }
            overall_conf = []
            for attr, votes in aggregated.items():
                if not votes:
                    summary[attr] = None
                    summary[f"{attr}_confidence"] = 0.0
                else:
                    label_scores = {}
                    for label, conf in votes:
                        label_scores[label] = label_scores.get(label, 0.0) + conf
                    best_label = max(label_scores, key=label_scores.__getitem__)
                    summary[attr] = best_label
                    summary[f"{attr}_confidence"] = round(label_scores[best_label] / len(votes), 3)
                    overall_conf.append(label_scores[best_label] / len(votes))

            summary["overall_confidence"] = round(sum(overall_conf) / len(overall_conf), 3) if overall_conf else 0.0
            results_out[key] = summary

        return JSONResponse({"status": "success", "results": results_out})

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Errore analyze_crops: {e}")
    finally:
        if tmp_zip_path.exists():
            os.remove(tmp_zip_path)
        if tmp_crops_dir.exists():
            shutil.rmtree(tmp_crops_dir, ignore_errors=True)
