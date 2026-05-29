from deep_sort_realtime.deepsort_tracker import DeepSort
import time
import cv2
from config import FRAME_SKIP, TARGET_CLASS
from BaseTracker import BaseTracker


class DeepSortTracker(BaseTracker):
    """ Implementazione di DeepSORT.
        algoritmo: 
            1. DETECTION: riceve le bbox da YOLO
            2. FEATURE EXTRACTION: estrae feature di apparenza
            3. ASSOCIATION: associa nuove detection alle tracce usando
                - distanza geometrica
                - similarità di apparenzaa
            4. FILTERING: Kalman filter per andare a predire le posizioni future
        
        è un algoritmo che gestisce bene le occlusion e si comporta bene con ID changes, ma è molto lento a causa della feature extraction
    """
    PARAMETER_SPECS = [{'name': 'par_max_iou_distance', 'label': 'Max IoU distance', 'type': 'float', 'default': 0.7, 'min': 0.0, 'max': 1.0, 'step': 0.01}, {'name': 'par_max_age', 'label': 'Max age', 'type': 'int', 'default': 30, 'min': 1, 'max': 300, 'step': 1}, {'name': 'par_n_init', 'label': 'N init', 'type': 'int', 'default': 3, 'min': 1, 'max': 30, 'step': 1}, {'name': 'par_nms_max_overlap', 'label': 'NMS max overlap', 'type': 'float', 'default': 1.0, 'min': 0.0, 'max': 1.0, 'step': 0.01}, {'name': 'par_max_cosine_distance', 'label': 'Max cosine distance', 'type': 'float', 'default': 0.2, 'min': 0.0, 'max': 1.0, 'step': 0.01}, {'name': 'par_nn_budget', 'label': 'NN budget', 'type': 'text', 'default': None}, {'name': 'par_gating_only_position', 'label': 'Gating only position', 'type': 'bool', 'default': False}, {'name': 'par_override_track_class', 'label': 'Override track class', 'type': 'text', 'default': None}, {'name': 'par_embedder', 'label': 'Embedder', 'type': 'text', 'default': 'mobilenet'}, {'name': 'par_half', 'label': 'Half precision', 'type': 'bool', 'default': True}, {'name': 'par_bgr', 'label': 'Input is BGR', 'type': 'bool', 'default': True}, {'name': 'par_embedder_gpu', 'label': 'Use GPU for embedder', 'type': 'bool', 'default': True}, {'name': 'par_embedder_model_name', 'label': 'Embedder model name', 'type': 'text', 'default': None}, {'name': 'par_embedder_wts', 'label': 'Embedder weights', 'type': 'text', 'default': None}, {'name': 'par_polygon', 'label': 'Polygon mode', 'type': 'bool', 'default': False}, {'name': 'par_today', 'label': 'Today / timestamp', 'type': 'text', 'default': None}]

    def __init__(self,
                        par_max_iou_distance=0.7,
                        par_max_age=30,
                        par_n_init=3,
                        par_nms_max_overlap=1.0,
                        par_max_cosine_distance=0.2,
                        par_nn_budget=None,
                        par_gating_only_position=False,
                        par_override_track_class=None,
                        par_embedder="mobilenet",
                        par_half=True,
                        par_bgr=True,
                        par_embedder_gpu=True,
                        par_embedder_model_name=None,
                        par_embedder_wts=None,
                        par_polygon=False,
                        par_today=None,):
        super().__init__(par_name="DeepSORT")    
        self.tracker = DeepSort(max_iou_distance=par_max_iou_distance,
                                max_age=par_max_age,
                                n_init=par_n_init,
                                nms_max_overlap=par_nms_max_overlap,
                                max_cosine_distance=par_max_cosine_distance,
                                nn_budget=par_nn_budget,
                                gating_only_position=par_gating_only_position,
                                override_track_class=par_override_track_class,
                                embedder=par_embedder,
                                half=par_half,
                                bgr=par_bgr,
                                embedder_gpu=par_embedder_gpu,
                                embedder_model_name=par_embedder_model_name,
                                embedder_wts=par_embedder_wts,
                                polygon=par_polygon,
                                today=par_today
                                )
       
   
    def run(self, par_detections_data: list, par_video_path: str) -> list:
        """
            Esegue il tracking con DeepSORT.
            Input:
            - par_detections_data: lista di {"frame_id", "detections", ...}
            - par_video_path: path al video (usato per leggere i frame)
            Output:
            - lista di {"frame", "track_id", "x1", "y1", "x2", "y2", "time"}
            """ 
        results = []
        start_time = time.time()
        cap = cv2.VideoCapture(par_video_path)
        
        for frame_data in par_detections_data:
            ret, frame = cap.read()
            if not ret:
                break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_number = frame_data["frame_id"]
            if frame_number % FRAME_SKIP != 0:
                continue
            if frame_number % 50 == 0:
                print(f"\tDeepSORT frame {frame_number}")
                
            # Converte le detection nel formato atteso da DeepSORT:
            # [x1, y1, width, height] (NOT [x1, y1, x2, y2])
            deepsort_input = [
                ([x1, y1, x2 - x1, y2 - y1], conf, TARGET_CLASS)
                for x1, y1, x2, y2, conf in frame_data["detections"]
            ]
            
            # Aggiorna il tracker con le nuove detection
            # DeepSORT estrae le feature dal frame e associa alle tracce
            for t in self.tracker.update_tracks(deepsort_input, frame=frame_rgb):
                if not t.is_confirmed(): # Filtra tracce confermate (scarta tracce nuove non ancora sicure)
                    continue
                l, t_y, r, b = t.to_ltrb() # Estrae le coordinate (formato [left, top, right, bottom])  streamlit run app.py
                elapsed_time = time.time() - start_time
                results.append({
                    "frame": frame_number, 
                    "track_id": int(t.track_id),
                    "x1": float(l), "y1": float(t_y), "x2": float(r), "y2": float(b),
                    "time": elapsed_time
                })   

        cap.release()
        return results
