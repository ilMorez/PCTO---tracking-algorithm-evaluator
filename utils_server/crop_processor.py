from pathlib import Path
from collections import defaultdict
import cv2
import numpy as np


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
                    cv2.imwrite(str(folder / f"frame_{frame_idx}.jpg"), crop)

        frame_idx += 1

    cap.release()


def extract_track_crops(video_path: str, results: list, output_base_dir: Path, tracker_name: str,
                         num_crops=5, margin=5, default_class="unknown"):
    """
    Estrae i crop delle bbox dal video originale per ogni track_id unico (singolo tracker).
    Struttura finale: output_base_dir / tracker_name / nome_classe / track_id / frame_X.jpg
    """
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


def _classify_color_hsv(bgr_image) -> tuple[str, float]:
    hsv = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]

    mean_s = np.mean(s)
    mean_v = np.mean(v)

    if mean_s < 30:
        if mean_v > 200:
            return "white", 0.9
        elif mean_v < 50:
            return "black", 0.9
        else:
            return "gray", 0.8

    mask = (s > 30) & (v > 30)
    if mask.sum() < 50:
        if mean_v > 200:
            return "white", 0.8
        elif mean_v < 50:
            return "black", 0.8
        else:
            return "gray", 0.7

    h_vals = h[mask].astype(float)

    color_ranges = [
        ("red",    [(0, 10), (160, 179)]),
        ("orange", [(10, 25)]),
        ("yellow", [(25, 35)]),
        ("green",  [(35, 85)]),
        ("cyan",   [(85, 100)]),
        ("blue",   [(100, 130)]),
        ("purple", [(130, 160)]),
    ]

    scores = {}
    total = len(h_vals)
    for color_name, ranges in color_ranges:
        count = 0
        for lo, hi in ranges:
            count += int(((h_vals >= lo) & (h_vals <= hi)).sum())
        scores[color_name] = count / total if total > 0 else 0.0

    best_color = max(scores, key=scores.__getitem__)
    confidence = scores[best_color]

    if confidence < 0.3:
        if mean_v > 200:
            return "white", 0.7
        elif mean_v < 50:
            return "black", 0.7
        else:
            return "gray", 0.7

    return best_color, min(confidence * 1.5, 1.0)


def _run_yolo_on_crop(model, bgr_image, conf_threshold: float = 0.25) -> list[dict]:
    results = model.predict(
        source=bgr_image,
        conf=conf_threshold,
        verbose=False,
        stream=False,
    )
    preds = []
    for r in results:
        boxes = r.boxes
        if boxes is None:
            continue
        for box in boxes:
            cls_id = int(box.cls[0])
            conf   = float(box.conf[0])
            xyxy   = box.xyxy[0].tolist()
            preds.append({
                "class_name": model.names[cls_id],
                "confidence": round(conf, 3),
                "bbox":       [round(v, 1) for v in xyxy],
            })
    preds.sort(key=lambda x: x["confidence"], reverse=True)
    print(f"[DEBUG] Predizioni su crop: {preds}")
    return preds