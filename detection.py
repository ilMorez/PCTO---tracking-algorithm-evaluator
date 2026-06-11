import cv2
import json
from pathlib import Path
from ultralytics import YOLO

class Detector:
    """ Classe DETECTOR
        Si occupa di:
            - caricare il modello YOLO specificato a runtime
            - elaborare il video frame-by-frame
            - identificare gli oggetti (detection) delle classi target
            - salvare le detections in JSON
        Output JSON: [{
            "frame_id": 0,
            "timestamp": 0.0,
            "detections": [[x1, y1, x2, y2, conf, class_name], ...]
        }, ...]
    """

    def __init__(self):
        self.model = None

    def run_detection(
        self,
        par_video_path: str,
        model_name: str,
        yolo_params: dict,
        target_classes: list,           # es. ["car", "person"] — lista di classi da rilevare
        par_output_json: str = "output/detections.json",
        progress_callback=None,
    ):
        """
        Args:
            par_video_path:   path al video
            model_name:       nome/path modello YOLO
            yolo_params:      parametri YOLO (conf, iou, imgsz, ...)
            target_classes:   lista di classi da tenere (filtra tutto il resto)
            par_output_json:  path output JSON
            progress_callback: callable(current_frame, total_frames)

        Returns:
            lista dataset detections, ogni detection include class_name come 6° campo
        """
        output_path = Path(par_output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        target_set   = set(target_classes)
        dataset      = []
        frame_number = 0

        self.model = YOLO(model_name)

        cap          = cv2.VideoCapture(par_video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        for result in self.model(source=par_video_path, **yolo_params):
            timestamp  = cap.get(cv2.CAP_PROP_POS_MSEC)
            detections = []

            for box in result.boxes:
                class_name = self.model.names[int(box.cls[0])]
                if class_name not in target_set:
                    continue
                conf            = float(box.conf[0])
                x1, y1, x2, y2 = (float(v) for v in box.xyxy[0])
                detections.append([
                    round(x1, 1), round(y1, 1),
                    round(x2, 1), round(y2, 1),
                    round(conf, 4),
                    class_name,         
                ])

            dataset.append({
                "frame_id":   frame_number,
                "timestamp":  round(timestamp, 2),
                "detections": detections,
            })

            if progress_callback is not None and total_frames > 0:
                progress_callback(frame_number, total_frames)

            if frame_number % 50 == 0:
                print(f"\tDetection: elaborati {frame_number} frame")

            frame_number += 1

        cap.release()

        with open(output_path, "w") as file:
            json.dump(dataset, file, indent=4)

        total_dets = sum(len(f["detections"]) for f in dataset)
        print(f"Detection completata: {frame_number} frame, {total_dets} detection totali")
        print(f"Modello usato: {model_name} | Classi: {target_classes}")
        print(f"Salvato: {output_path}\n")

        return dataset


def filter_detections_by_class(detections: list, class_name: str) -> list:
    """
    Filtra il dataset di detections tenendo solo una classe specifica.
    Rimuove il campo class_name dal formato (i tracker si aspettano [x1,y1,x2,y2,conf]).

    Args:
        detections: lista frame con detections nel formato multi-classe
        class_name: classe da tenere

    Returns:
        lista frame con sole detection della classe richiesta, formato [x1,y1,x2,y2,conf]
    """
    filtered = []
    for frame in detections:
        frame_dets = [
            d[:5]  # solo [x1, y1, x2, y2, conf] — i tracker non conoscono la classe
            for d in frame["detections"]
            if len(d) >= 6 and d[5] == class_name
        ]
        filtered.append({
            "frame_id":   frame["frame_id"],
            "timestamp":  frame["timestamp"],
            "detections": frame_dets,
        })
    return filtered
