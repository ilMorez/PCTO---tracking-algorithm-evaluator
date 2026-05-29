import numpy as np
from BaseTracker import BaseTracker
from config import *
import time
from scipy.optimize import linear_sum_assignment

try:
    from sklearn.metrics.pairwise import cosine_similarity
except ImportError:
    # Fallback minimale se sklearn non è installato
    def cosine_similarity(X, Y):
        dot_product = np.dot(X, Y.T)
        norm_X = np.linalg.norm(X)
        norm_Y = np.linalg.norm(Y)
        if norm_X == 0 or norm_Y == 0:
            return np.array([[0.0]])
        return dot_product / (norm_X * norm_Y)
   
class AppearanceTrack:
    """ Classe helper per AppearanceTracker, si tatta di una track singola con feature di apparenza.
        Diversamente da KalmanTrack non viene utilizzato il Kalman filter ma solo la feature""" 
    def __init__(self, par_track_id, par_bbox, par_embedding):
        self.id = par_track_id
        self.bbox = par_bbox
        self.embedding = par_embedding # Feature vector
        self.age = 0

class AppearanceTracker(BaseTracker):
    """ Implementazione di un tracker basato su feature di apparenza:
            1. estrae le feature
            2. calcola la similiarità coseno tra feature di tracce e detection
            3. associa basandosi sulla similarità
            4. aggiorna le tracks
        Si tratta di un tracker robusto ad eventuali deferomazioni, ma per funzionare been servirebbero feature neurali reali e non feature di bassa qualità (solo bbox)"""
    PARAMETER_SPECS = [{'name': 'par_similarity_threshold', 'label': 'Similarity threshold', 'type': 'float', 'default': 0.7, 'min': 0.0, 'max': 1.0, 'step': 0.01}, {'name': 'par_max_age', 'label': 'Max age', 'type': 'int', 'default': 30, 'min': 1, 'max': 300, 'step': 1}]

    def __init__(self, par_similarity_threshold=0.7, par_max_age=30):
        super().__init__("Appearance")
        self.similarity_threshold = par_similarity_threshold # Similarità minimia, percentuale normalizzata tra 0 e 1 (1 = identico)
        self.max_age = par_max_age
        self.tracks = []
        self.next_id = 0

    def run(self, par_detection_data: list, par_video_path: str) -> list:
        results = []
        start_time = time.time()
        
        for frame_data in par_detection_data:
            frame_number = frame_data["frame_id"]
            if frame_number % FRAME_SKIP != 0: continue
            
            dets = frame_data["detections"]
            
            # NOTA:
            # Il formato JSON standard NON contiene feature embedding.
            # Qui generiamo un finto embedding basato sulle coordinate bbox.
            # Per un vero tracker di apparenza, serve una rete neurale
            # che estrae feature dal patch della bbox (es: ResNet, etc).
            embeddings = [np.array([d[0], d[1], d[2], d[3]], dtype=np.float32) for d in dets]
            
            if len(self.tracks) == 0:
                for det, emb in zip(dets, embeddings):
                    self.tracks.append(AppearanceTrack(self.next_id, det[:4], emb))
                    self.next_id += 1
            else:
                # Calcola matrice di costo (1 - similarità coseno)
                cost_matrix = np.zeros((len(self.tracks), len(dets)))
                for i, track in enumerate(self.tracks):
                    for j, emb in enumerate(embeddings):
                        sim = cosine_similarity(track.embedding.reshape(1, -1), emb.reshape(1, -1))[0][0] # Similarità coseno tra feature
                        cost_matrix[i, j] = 1 - sim # Costo = 1 - similarità (così la similarità alta = costo basso)

                # Associazione ottimale
                rows, cols = linear_sum_assignment(cost_matrix)
                assigned_tracks = set()
                assigned_dets = set()

                # Aggiorna tracce associate
                for r, c in zip(rows, cols):
                    similarity = 1 - cost_matrix[r, c]
                    if similarity > self.similarity_threshold:
                        self.tracks[r].bbox = dets[c][:4]
                        self.tracks[r].embedding = embeddings[c]
                        self.tracks[r].age = 0
                        assigned_tracks.add(r)
                        assigned_dets.add(c)

                # Invecchia e rimuovi tracce vecchie
                updated_tracks = []
                for i, track in enumerate(self.tracks):
                    if i not in assigned_tracks: track.age += 1
                    if track.age < self.max_age: updated_tracks.append(track)
                self.tracks = updated_tracks

                # Crea tracce nuove
                for i, det in enumerate(dets):
                    if i not in assigned_dets:
                        self.tracks.append(AppearanceTrack(self.next_id, det[:4], embeddings[i]))
                        self.next_id += 1

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
    