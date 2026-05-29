import cv2
import numpy as np
from BaseTracker import BaseTracker
from config import *
import time

   
class OpticalFlowTracker(BaseTracker):
    """ Implementazione di un tracker basasto su OPTICAL FLOW (Lucas-Kanade)
        I pixel degli oggetti in un video si muovono in maniere coerente, 
        quindi possiamo stimare il movimento degli oggetti calcolando l'optical flow:
            1. si estraggono i centroidi dalle bbox
            2. si calolca l'optical flow tra frame consecutivi
            3. si predice la nuova posizione dei ceentroidi
            4. si associano i centroidi predetti con le nuove detection
        Si tratta di un tracker molto low level (basato sui pixel) che è molto semplice ma molto sensibile a rotazioni, cambio di scala e occlusion"""
    PARAMETER_SPECS = [{'name': 'par_bbox_size', 'label': 'BBox Size', 'type': 'int', 'default': 30, 'min': 2, 'max': 200, 'step': 2}]

    def __init__(self, par_bbox_size=30):
        super().__init__("OpticalFlow")
        self.bbox_size = par_bbox_size

    def run(self, par_detection_data: list, par_video_path: str) -> list:
        results = []
        start_time = time.time()
        cap = cv2.VideoCapture(par_video_path)
        
        prev_gray = None # Frame precedente in scala di grigi
        
        for frame_data in par_detection_data:
            ret, frame = cap.read()
            if not ret: 
                break
            
            frame_number = frame_data["frame_id"]
            if frame_number % FRAME_SKIP != 0: 
                continue
                
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            dets = frame_data["detections"]
            elapsed_time = time.time() - start_time
            
            # Se è il primo frame o se le detection sono 0 non possiamo calcolare
            if prev_gray is not None and len(dets) > 0:
                # Estraiamo i centri delle bboxes correnti
                pts = []
                for det in dets:
                    pts.append([(det[0] + det[2]) / 2, (det[1] + det[3]) / 2])
                
                points = np.array(pts, dtype=np.float32).reshape(-1, 1, 2) # Converte in formato numpy richiesto da calcOpticalFlowPyrLK
                # Calcola il flusso ottico, stima dove si spostano i punti dal frame precedente a questo
                next_points, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, gray, points, None)
                
                for idx, (pt, st) in enumerate(zip(next_points, status)):
                    if st == 1: # Il flusso è stato calcolato con successo
                        cx, cy = pt[0]
                        # Ricostruiamo una Bounding Box fittizia attorno al punto (es. 30x30 pixel)
                        # in modo che l'evaluator non calcoli un'area pari a zero.
                        results.append({
                            "frame": frame_number, 
                            "track_id": idx,
                            "x1": float(cx - 15), "y1": float(cy - 15), 
                            "x2": float(cx + 15), "y2": float(cy + 15),
                            "time": elapsed_time
                        })
            
            prev_gray = gray
            
        cap.release()
        return results
    