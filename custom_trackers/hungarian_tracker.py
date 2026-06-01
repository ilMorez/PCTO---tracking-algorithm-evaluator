import numpy as np
from BaseTracker import BaseTracker
from config import *
import time
from scipy.optimize import linear_sum_assignment
from filterpy.kalman import KalmanFilter

class KalmanTrack:
    """ Classe helper per HungarianTracker, rappresenta un track singola con Kalman filter per previsione.
        Essa ha i metodi:
            - PREDICT, che va a stimare la posizione futura basandosi sul modello
            - UPDATE, che va a correggere al stima con la nuova osservazione
        Il Kalman filter serve a rendere il tracker più robousto ad ecentuali occlusion""" 
    def __init__(self, par_track_id, par_bbox):
        self.id = par_track_id
        
        # dim_x=7: stato è [x1, y1, x2, y2, vx, vy, scale]
        #          (posizione + velocità + fattore di scala)
        # dim_z=4: osservazione è [x1, y1, x2, y2] (la bbox)
        self.kf = KalmanFilter(dim_x=7, dim_z=4)
        self.kf.F = np.eye(7) # F: matrice di transizione (previsione)
        
        # H: matrice di osservazione (come osservazione → stato)
        # Osserviamo solo le coordinate, non velocità
        self.kf.H = np.zeros((4,7))
        self.kf.H[:4, :4] = np.eye(4)
        self.kf.P *= 10 # Covarianza dello stato (iniziale) moltiplicato per 10 per partire con alta incertezza
        self.kf.R *= 1 # Covarianza della misurazione bassa, si traduce in fiducia alta nelle misurazioni YOLO
        self.kf.Q *= 0.01 # Covarianza del rumore del processo bassa, in quanto assumiamo che il modello sia accurato
        self.kf.x[:4] = np.array(par_bbox).reshape((4, 1)) # Stato iniziale: poniamo la bbox osservata
        self.age = 0 # Quanti frame questa traccia è stata senza essere aggiornata
        
    def predict(self):
        self.kf.predict()
        return self.kf.x[:4].flatten()
    
    def update(self, par_bbox):
        self.kf.update(np.array(par_bbox))
        self.age = 0
        
    def get_bbox(self):
        return self.kf.x[:4].flatten()  # Corretto .X in .x
    
class HungarianTracker(BaseTracker):
    """ Implementazione di un tracker basato sull'Hungarian Algorithm:
            1. per ogni track va a predire la posizione attraverso Kalman
            2. calcola la distanza dei centroidi di prediction e deetection
            3. usa l'Hungarian Algorithm per trovare l'associazione ottimale
            4. aggiorna la tracks associate, crea nuove tracks ed elimina quelle vecchie
        Si tratta di un tracker implementato da zero, senza l'uso di una libreria dedicata, che è concettualmente semplice,
        deterministico, ma non ha feature exctraction ed è comunque sensibile ad occlusion"""
    PARAMETER_SPECS = [{'name': 'par_distance_threshold', 'label': 'Distance threshold', 'type': 'int', 'default': 100, 'min': 0, 'max': 1000, 'step': 1}, 
                       {'name': 'par_max_age', 'label': 'Max age', 'type': 'int', 'default': 30, 'min': 1, 'max': 300, 'step': 1}]

    def __init__(self, par_distance_threshold=100, par_max_age=30):
        super().__init__("Hungarian")
        self.distance_threshold = par_distance_threshold
        self.max_age = par_max_age
        self.tracks = [] # Lista di KalmanTrack attive
        self.next_id = 0 
        
    def center_distance(self, par_boxA, par_boxB): 
        """ Calcola e restituisce la distanza euclidea tra i centroidi di due bbox passate come parametro.
                args: 
                    par_boxA: [x1, y1, x2, y2]
                    par_boxB: [x1, y1, x2, y2]"""
        cx1 = (par_boxA[0] + par_boxA[2]) / 2
        cy1 = (par_boxA[1] + par_boxA[3]) / 2
        cx2 = (par_boxB[0] + par_boxB[2]) / 2
        cy2 = (par_boxB[1] + par_boxB[3]) / 2
        return np.sqrt((cx1 - cx2)**2 + (cy1 - cy2)**2)
    
    def run(self, par_detection_data: list, par_video_path: str, progress_callback=None) -> list:
        results = []
        start_time = time.time()
        total_frames = len(par_detection_data)
        
        for frame_data in par_detection_data:
            frame_number = frame_data["frame_id"]
            if frame_number % FRAME_SKIP != 0: continue
            
            dets = frame_data["detections"]
            predicted_boxes = [track.predict() for track in self.tracks] # Predice la prossima posizione di tutte le tracks

            if len(predicted_boxes) == 0: # Se non ci sono tracce ne crea di nuove per ogni detection
                for det in dets:
                    self.tracks.append(KalmanTrack(self.next_id, det[:4]))
                    self.next_id += 1
            else:
                # Calcola la matrice di costo (distanza tra prediction e detection)
                cost_matrix = np.zeros((len(predicted_boxes), len(dets)))
                for i, pred in enumerate(predicted_boxes):
                    for j, det in enumerate(dets):
                        cost_matrix[i, j] = self.center_distance(pred, det[:4])
                
                # effettua l'assegnamento ottimale con Hungarian algorithm
                rows, cols = linear_sum_assignment(cost_matrix)
                assigned_tracks = set()
                assigned_dets = set()

                for r, c in zip(rows, cols): # Aggiorna le tracce assegnate
                    if cost_matrix[r, c] < self.distance_threshold: # Se la distanza è sopra soglia la ignora 
                        self.tracks[r].update(dets[c][:4])
                        assigned_tracks.add(r)
                        assigned_dets.add(c)

                updated_tracks = []
                for i, track in enumerate(self.tracks): # Aggiorna l'età delle tracce non assegnate eliminando le troppo vecchie
                    if i not in assigned_tracks:
                        track.age += 1
                    if track.age < self.max_age:
                        updated_tracks.append(track)
                self.tracks = updated_tracks

                # Crea tracce nuove per le detection non assegnate
                for i, det in enumerate(dets):
                    if i not in assigned_dets:
                        self.tracks.append(KalmanTrack(self.next_id, det[:4]))
                        self.next_id += 1

            elapsed_time = time.time() - start_time
            for track in self.tracks:
                if track.age == 0: # Salva solo se aggiornato in questo frame
                    x1, y1, x2, y2 = track.get_bbox()
                    results.append({
                        "frame": frame_number, "track_id": track.id,
                        "x1": float(x1), "y1": float(y1), "x2": float(x2), "y2": float(y2),
                        "time": elapsed_time
                    })
                    
            if progress_callback is not None:
                progress_callback(frame_number, total_frames)
        return results
