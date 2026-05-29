import numpy as np
from BaseTracker import BaseTracker
from norfair import Detection, Tracker as LibNorfairTracker
from config import *
import time

class NorfairTracker(BaseTracker):
    """ Implementazione con Norfair: tracker leggero basato su distanza euclidea.
        algoritmo:
            1. estrae il centroide della bbox
            2. calcola la distanza euclidea tra i centroidi frame-a-frame
            3. associa in base a una soglia di distanza (non controlla le feature grafiche come DeepSORT)
        È un algoritmo molto veloce e semplice, ma è sensibiel alle occlusion e può fare ID switch facilmente"""
    PARAMETER_SPECS = [{'name': 'par_distance_function', 'label': 'Distance function', 'type': 'text', 'default': 'euclidean'}, {'name': 'par_distance_threshold', 'label': 'Distance threshold', 'type': 'int', 'default': 50, 'min': 0, 'max': 1000, 'step': 1}]

    def __init__(self, par_distance_function="euclidean", par_distance_threshold=50):
        super().__init__("Norfair")
        self.tracker = LibNorfairTracker(par_distance_function, par_distance_threshold) # distance_threshold: soglia di associazione (in pixel)
        
    def run(self, par_detection_data: list, par_video_path: str) -> list:
        results = []
        start_time = time.time()

        for frame_data in par_detection_data:
            frame_number = frame_data["frame_id"]
            if frame_number % FRAME_SKIP != 0:
                continue

            # Crea le detection per Norfair: solo il centroide
            # Norfair accetta array di coordinate [[x, y]]
            norfair_dets = [
                Detection(points=np.array([[(x1 + x2) / 2, (y1 + y2) / 2]]))
                for x1, y1, x2, y2, _ in frame_data["detections"]
            ]
            for obj in self.tracker.update(detections=norfair_dets):
                if obj.id is None: # Salta se l'ID non è ancora assegnato
                    continue
                cx, cy = obj.estimate[0] # Estrae il centroide stimato (può essere leggermente diverso da quello calcolato, grazie al Kalman filter interno)
                elapsed_time = time.time() - start_time
                results.append({
                    "frame": frame_number, "track_id": obj.id,
                    "x1": cx, "y1": cy, "x2": cx, "y2": cy,
                    "time": elapsed_time
                })
        return results
