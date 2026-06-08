import cv2
import json
from pathlib import Path
from ultralytics import YOLO
from config import TARGET_CLASS

class Detector:
    """ Classe DETECTOR
        Si occupa di:
            - caricare il modello YOLO specificato a runtime
            - elaborare il video frame-by-frame
            - identificare gli oggetti (detection) della classe target
            - salvare le detections in JSON
        Pipeline:
            1. prende un frame del video alla volta
            2. passa il frame a YOLO
            3. filtra solo gli oggetti della classe desiderata
            4. estrae coordinate bbox e confidence score
            5. salva i risultati in JSON
        Output JSON: [{
            "frame_id": 0,
            "timestamp": 0.0,
            "detections": [[x1, y1, x2, y2, conf], ...]
        }, ...]
    """

    def __init__(self):
        self.model       = None
        self.target_class = TARGET_CLASS

    def run_detection(
        self,
        par_video_path: str,
        model_name: str,
        yolo_params: dict,
        par_output_json: str = "output/detections.json",
        progress_callback=None,
    ):
        """ Esegue il rilevamento degli oggetti su un video intero.

            Args:
                par_video_path:  path al video (es. "video/traffic.mp4")
                model_name:      nome/path del modello YOLO (es. "yolov8n.pt")
                yolo_params:     dizionario di parametri passati al modello
                                 (conf, iou, imgsz, half, stream, verbose, …)
                par_output_json: path di output JSON
                progress_callback: callable(current_frame, total_frames) opzionale

            Returns:
                lista del dataset delle detections
        """
        output_path = Path(par_output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        dataset      = []
        frame_number = 0

        self.model = YOLO(model_name)

        cap          = cv2.VideoCapture(par_video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # YOLO itera i frame in streaming; cv2 serve solo per il timestamp
        for result in self.model(source=par_video_path, **yolo_params):
            timestamp  = cap.get(cv2.CAP_PROP_POS_MSEC)
            detections = []

            for box in result.boxes:
                if self.model.names[int(box.cls[0])] != self.target_class:
                    continue
                conf            = float(box.conf[0])
                x1, y1, x2, y2 = (float(v) for v in box.xyxy[0])
                detections.append([
                    round(x1, 1), round(y1, 1),
                    round(x2, 1), round(y2, 1),
                    round(conf, 4),
                ])

            dataset.append({
                "frame_id":  frame_number,
                "timestamp": round(timestamp, 2),
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
        print(f"Modello usato: {model_name}")
        print(f"Salvato: {output_path}\n")

        return dataset
