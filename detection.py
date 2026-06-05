import cv2
import json
from pathlib import Path
from ultralytics import YOLO
from config import TARGET_CLASS, MODEL_PATH

class Detector:
    """ Classe DETECTOR
        Si occupa di:
            - caricare il modello YOLO pre-addestrato
            - elaborare il video frame-by-frame
            - identificare gli oggetti (detection) della classe target
            - salvare le detections in JSON
        Pipeline:
            1. prende uin frame del video alla volta
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
    
    def __init__(self, par_model_path=MODEL_PATH):
        """ Inizializza il detector caricando il modello YOLO.
                Args:
                    par_model_path: path al file del modello YOLO
        """
        self.model = YOLO(par_model_path)
        self.target_class = TARGET_CLASS
        
    def run_detection(self, par_video_path: str, par_output_json: str = "output/detections.json", progress_callback = None):
        """ Esegue il rilevamento degli oggetti su un video intero:
                1. apre il video file
                2. per ogni frame:
                    - passa a YOLO
                    - estrae bounding box
                    - filtra classe target
                    - arrotonda valori per risparmiare spazio JSON
                3. salva tutto in JSON
                Args:
                    par_video_path: path al video (es. "video/traffic.mp4")
                    par_output_json: path di output JSON (default: "output/detections.json")
                Returns:
                    lista del dataset (utile se il codice caller vuole usarlo subito)
        """
        # Crea la cartella di output se non esiste
        output_path = Path(par_output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        dataset = []
        frame_number = 0

        cap = cv2.VideoCapture(par_video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        for result in self.model(source=par_video_path, stream=True, verbose=False): # result è un oggetto Results di YOLO che contiene: result.boxes: lista di bounding box rilevate nel frame, result.names: dizionario {class_id: class_name}
            timestamp  = cap.get(cv2.CAP_PROP_POS_MSEC)
            detections = []

            for box in result.boxes: # box è un oggetto che contiene: box.cls: classe, box.conf: confidenza (0.0 - 1.0), box.xyxy: coordinate [x1, y1, x2, y2]
                if self.model.names[int(box.cls[0])] != TARGET_CLASS:
                    continue
                conf = float(box.conf[0])
                x1, y1, x2, y2 = (float(v) for v in box.xyxy[0])
                
                detections.append([
                    round(x1, 1), round(y1, 1),
                    round(x2, 1), round(y2, 1),
                    round(conf, 4)
                ])

            frames_dict = {
                "frame_id": frame_number,
                "timestamp":  round(timestamp, 2),
                "detections": detections
            }
            
            dataset.append(frames_dict)
            
            if progress_callback is not None and total_frames > 0:
                progress_callback(frame_number, total_frames)
            
            if frame_number % 50 == 0:
                print(f"\tDetection: elaborati {frame_number} frame")
            
            frame_number += 1

        cap.release()

        with open(output_path, "w") as file: # Apre il file in modalità write
            json.dump(dataset, file, indent=4) # Salva il dataset in formato JSON leggibile

        total_dets = sum(len(f["detections"]) for f in dataset)
        print(f"Detection completata: {frame_number} frame, {total_dets} detection totali")
        print(f"Salvato: {output_path}\n")
        
        return dataset
