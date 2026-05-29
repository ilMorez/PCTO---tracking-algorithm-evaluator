import numpy as np
from BaseTracker import BaseTracker
from motpy import Detection as MotpyDetection, MultiObjectTracker
from config import *
import time  
  
class MotpyTracker(BaseTracker):
    """ Implementazione di Motpy tracker, si tratta di una libreria di python che utilizza un apporccio probabilistico:
        1. predice la posizione degli oggetti
        2. associa nuove detection alle tracce
        3. aggiorna il modello cinematico
    Si tratta di un algrotimo di tracking che sfrutta una libreria dedicata a ben documentata, però non arriva a competere con gli algoritmi più moderni"""
    PARAMETER_SPECS = [{'name': 'par_dt', 'label': 'Time step (dt)', 'type': 'float', 'default': 0.03, 'min': 0.001, 'max': 1.0, 'step': 0.001}]

    def __init__(self, par_dt=0.03):
        super().__init__("Motpy")
        self.tracker = MultiObjectTracker(dt=par_dt)
        
    def run(self, par_detection_data: list, par_video_path: str) -> list:
        results = []
        start_time = time.time()    
        for frame_data in par_detection_data:
            frame_number = frame_data["frame_id"]
            
            if frame_number % FRAME_SKIP != 0: 
                continue
            
            dets = frame_data["detections"]
            motpy_dets = []
            
            for d in dets:
                motpy_dets.append(MotpyDetection(box=np.array(d[:4]), score=d[4]))
            
            self.tracker.step(detections=motpy_dets)
            tracks = self.tracker.active_tracks()
            
            for t in tracks:
                x1, y1, x2, y2 = t.box
                elapsed_time = time.time() - start_time
                results.append({
                    "frame": frame_number,
                    "track_id": int(hash(t.id) % 10000),
                    "x1": float(x1), "y1": float(y1),
                    "x2": float(x2), "y2": float(y2),
                        "time": elapsed_time
                })
        return results
   