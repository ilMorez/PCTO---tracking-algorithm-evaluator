import numpy as np
from BaseTracker import BaseTracker
from config import *
import time

class VelocityTrack:
    """ Classe helper per VelocityTracker.
        Si tratta di una track che include anche la velocità sulle x e la velocità sulla y"""
    def __init__(self, track_id, bbox):
        self.id = track_id
        self.bbox = bbox
        self.velocity = np.array([0, 0]) # velocità [vx, vy]
        self.age = 0

class VelocityTracker(BaseTracker):
    """ Implementazione di un tracker basato sulla velocità costante degli oggetti.
        Si basa sul concetto che gli oggetti si muovo tendenzialmente con velocità quasi costante, non si hanno accelerazioni improvvise (a meno di occlusion):
            1. per ogni track viene predetta la nuove posozione aggiungendo la velocità a quella precedente
            2. viene associata la prediction con detection basata su distanza
            3. viene aggiornata la velocità
            4. viene aggiornata la track
        Si tratta di un tracker smeplice, veloce e che funziona bene con oggeti con movimento uniforme,
        ma non gestisce l'eventuale accelerazione ed è sensibile all'occlusion"""
    PARAMETER_SPECS = [{'name': 'par_distance_threshold', 'label': 'Distance threshold', 'type': 'int', 'default': 100, 'min': 0, 'max': 1000, 'step': 1}, {'name': 'par_max_age', 'label': 'Max age', 'type': 'int', 'default': 30, 'min': 1, 'max': 300, 'step': 1}]

    def __init__(self, par_distance_threshold=100, par_max_age=30):
        super().__init__("Velocity")
        self.distance_threshold = par_distance_threshold
        self.max_age = par_max_age
        self.tracks = []
        self.next_id = 0

    def center(self, box):
        return np.array([(box[0] + box[2]) / 2, (box[1] + box[3]) / 2])

    def run(self, par_detection_data: list, par_video_path: str) -> list:
        results = []
        start_time = time.time()
        
        for frame_data in par_detection_data:
            frame_number = frame_data["frame_id"]
            if frame_number % FRAME_SKIP != 0: continue
            
            dets = frame_data["detections"]
            updated_tracks = []
            used = set()

            # Per ogni track trova la miglior detection corrispondente
            for track in self.tracks:
                predicted_center = self.center(track.bbox) + track.velocity # Predici la nuova posizione basata sulla velocità
                best_dist = 1e9
                best_idx = -1
                
                # Trova la detection più vicina alla predizione
                for i, det in enumerate(dets):
                    if i in used: 
                        continue # Detection già assegnata
                    det_center = self.center(det[:4])
                    dist = np.linalg.norm(predicted_center - det_center)
                    if dist < best_dist:
                        best_dist = dist
                        best_idx = i

                # Se la distanza è sotto soglia, aggiorna la track
                if best_dist < self.distance_threshold:
                    det = dets[best_idx]
                    old_center = self.center(track.bbox)
                    new_center = self.center(det[:4])
                    track.velocity = new_center - old_center
                    track.bbox = det[:4]
                    track.age = 0
                    updated_tracks.append(track)
                    used.add(best_idx)
                else:
                    # Nessuna detection, track invecchia
                    track.age += 1
                    if track.age < self.max_age:
                        updated_tracks.append(track)

            # Crea track nuove per le detection non assegnate
            for i, det in enumerate(dets):
                if i not in used:
                    updated_tracks.append(VelocityTrack(self.next_id, det[:4]))
                    self.next_id += 1

            self.tracks = updated_tracks
            elapsed_time = time.time() - start_time
            
            for track in self.tracks:
                if track.age == 0:
                    x1, y1, x2, y2 = track.bbox
                    results.append({
                        "frame": frame_number, "track_id": track.id,
                        "x1": float(x1), "y1": float(y1), "x2": float(x2), "y2": float(y2),
                        "time": elapsed_time
                    })
        return results