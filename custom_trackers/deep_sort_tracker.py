from deep_sort_realtime.deepsort_tracker import DeepSort
import time
import cv2
import traceback
import numpy as np
from config import FRAME_SKIP, TARGET_CLASS
from BaseTracker import BaseTracker

# Importazioni per CLIP (HuggingFace)
try:
    from transformers import CLIPProcessor, CLIPModel
    import torch
    from PIL import Image
except ImportError:
    pass

class DeepSortTracker(BaseTracker):
    """ 
    Implementazione di DeepSORT con supporto CLIP e filtro semantico tramite query testuale.
    """
    
    PARAMETER_SPECS = [
        {'name': 'par_max_iou_distance', 'label': 'Max IoU distance', 'type': 'float', 'default': 0.7, 'min': 0.0, 'max': 1.0, 'step': 0.01},
        {'name': 'par_max_age', 'label': 'Max age', 'type': 'int', 'default': 50, 'min': 1, 'max': 300, 'step': 1},
        {'name': 'par_n_init', 'label': 'N init', 'type': 'int', 'default': 3, 'min': 1, 'max': 30, 'step': 1},
        {'name': 'par_nms_max_overlap', 'label': 'NMS max overlap', 'type': 'float', 'default': 0.9, 'min': 0.0, 'max': 1.0, 'step': 0.01},
        {'name': 'par_max_cosine_distance', 'label': 'Max cosine distance', 'type': 'float', 'default': 0.21, 'min': 0.0, 'max': 1.0, 'step': 0.01},
        {'name': 'par_nn_budget', 'label': 'NN budget', 'type': 'text', 'default': None},
        {'name': 'par_gating_only_position', 'label': 'Gating only position', 'type': 'bool', 'default': False},
        {'name': 'par_override_track_class', 'label': 'Override track class', 'type': 'text', 'default': None},
        {'name': 'par_embedder', 'label': 'Embedder', 'type': 'select', 'default': 'clip', 'options': ['mobilenet', 'efficientnet', 'resnet', 'clip']},
        {'name': 'par_half', 'label': 'Half precision', 'type': 'bool', 'default': True},
        {'name': 'par_bgr', 'label': 'Input is BGR', 'type': 'bool', 'default': True},
        {'name': 'par_embedder_gpu', 'label': 'Use GPU for embedder', 'type': 'bool', 'default': True},
        {'name': 'par_embedder_model_name', 'label': 'Embedder model name', 'type': 'text', 'default': None},
        {'name': 'par_embedder_wts', 'label': 'Embedder weights', 'type': 'text', 'default': None},
        {'name': 'par_polygon', 'label': 'Polygon mode', 'type': 'bool', 'default': False},
        {'name': 'par_today', 'label': 'Today / timestamp', 'type': 'text', 'default': None},
        {'name': 'par_orig', 'label': 'Usa coordinate originali (orig)', 'type': 'bool', 'default': True},
        {'name': 'text_query', 'label': 'Query testuale (es. "red car")', 'type': 'text', 'default': ''},
        {'name': 'text_threshold', 'label': 'Soglia similarità testo-immagine', 'type': 'float', 'default': 0.25, 'min': 0.0, 'max': 1.0, 'step': 0.01},
    ]

    def __init__(self,
                        par_max_iou_distance=0.7,
                        par_max_age=75,
                        par_n_init=3,
                        par_nms_max_overlap=1.0,
                        par_max_cosine_distance=0.28,
                        par_nn_budget=None,
                        par_gating_only_position=False,
                        par_override_track_class=None,
                        par_embedder="clip",
                        par_half=True,
                        par_bgr=True,
                        par_embedder_gpu=True,
                        par_embedder_model_name=None,
                        par_embedder_wts=None,
                        par_polygon=False,
                        par_today=None,
                        par_orig=True,
                        text_query="",
                        text_threshold=0.25):
        super().__init__(par_name="DeepSORT")    
        
        self.is_clip = (par_embedder == "clip")
        embedder_value = None if self.is_clip else par_embedder

        self.tracker = DeepSort(max_iou_distance=par_max_iou_distance,
                                max_age=par_max_age,
                                n_init=par_n_init,
                                nms_max_overlap=par_nms_max_overlap,
                                max_cosine_distance=par_max_cosine_distance,
                                nn_budget=par_nn_budget,
                                gating_only_position=par_gating_only_position,
                                override_track_class=par_override_track_class,
                                embedder=embedder_value, 
                                half=par_half,
                                bgr=par_bgr,
                                embedder_gpu=par_embedder_gpu,
                                embedder_model_name=par_embedder_model_name,
                                embedder_wts=par_embedder_wts,
                                polygon=par_polygon,
                                today=par_today
                                )
        self.orig = par_orig
        self.text_query = text_query
        self.text_threshold = text_threshold
        
        if self.is_clip:
            print("[DeepSORT] Inizializzazione pipeline CLIP...")
            self.device = "cuda" if torch.cuda.is_available() and par_embedder_gpu else "cpu"
            model_id = "openai/clip-vit-base-patch32"
            self.model = CLIPModel.from_pretrained(model_id).to(self.device)
            self.processor = CLIPProcessor.from_pretrained(model_id)
            if self.text_query:
                self.text_embedding = self._get_text_embedding(self.text_query)
                print(f"[DeepSORT] Query testuale: '{self.text_query}' (soglia={self.text_threshold})")
            else:
                self.text_embedding = None
            print("[DeepSORT] Pipeline CLIP pronta e attiva!")

    def _get_text_embedding(self, text):
        """Calcola embedding normalizzato del testo."""
        inputs = self.processor(text=[text], return_tensors="pt", padding=True).to(self.device)
        with torch.no_grad():
            features = self.model.get_text_features(**inputs)
        # Assicurarsi che features sia un tensore
        if hasattr(features, 'pooler_output'):
            features = features.pooler_output
        elif hasattr(features, 'last_hidden_state'):
            features = features.last_hidden_state.mean(dim=1)
        features = features / features.norm(p=2, dim=-1, keepdim=True)
        return features.cpu().numpy()[0]

    def _get_clip_embeddings(self, crops):
        """ Estrae i descrittori ad alta stabilità dai ritagli immagine """
        if not crops:
            return []
        pil_crops = [Image.fromarray(crop) for crop in crops]
        inputs = self.processor(images=pil_crops, return_tensors="pt").to(self.device)
        with torch.no_grad():
            output = self.model.get_image_features(**inputs)
        # Estrai il tensore delle features
        if hasattr(output, 'pooler_output'):
            features = output.pooler_output
        elif hasattr(output, 'last_hidden_state'):
            features = output.last_hidden_state.mean(dim=1)
        else:
            features = output
        # Normalizzazione
        features = features / features.norm(p=2, dim=-1, keepdim=True)
        return list(features.cpu().numpy())
       
    def run(self, par_detections_data: list, par_video_path: str, progress_callback=None) -> list:
        results = []
        start_time = time.time()
        cap = cv2.VideoCapture(par_video_path)
        total_frames = len(par_detections_data)
        
        try:
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
                    
                dets = frame_data["detections"]
                filtered_dets = []
                crops = []
                embeds = []
                
                for det in dets:
                    x1, y1, x2, y2, conf = det[:5]
                    h, w, _ = frame_rgb.shape
                    ix1, iy1, ix2, iy2 = max(0, int(x1)), max(0, int(y1)), min(w, int(x2)), min(h, int(y2))
                    if ix2 > ix1 and iy2 > iy1:
                        crop = frame_rgb[iy1:iy2, ix1:ix2]
                    else:
                        crop = np.zeros((224, 224, 3), dtype=np.uint8)
                    
                    if self.is_clip:
                        embed_img = self._get_clip_embeddings([crop])[0]
                        if self.text_embedding is not None:
                            sim = np.dot(embed_img, self.text_embedding)
                            if sim < self.text_threshold:
                                continue
                        embeds.append(embed_img)
                    filtered_dets.append(det)
                    crops.append(crop)
                
                deepsort_input = [
                    ([x1, y1, x2 - x1, y2 - y1], conf, TARGET_CLASS)
                    for x1, y1, x2, y2, conf in filtered_dets
                ]
                
                if not self.is_clip:
                    embeds = None
                
                try:
                    for t in self.tracker.update_tracks(deepsort_input, embeds=embeds, frame=frame_rgb):
                        if not t.is_confirmed() or t.time_since_update > 0:
                            continue
                        l, t_y, r, b = t.to_ltrb(orig=self.orig)
                        elapsed_time = time.time() - start_time
                        results.append({
                            "frame": frame_number, 
                            "track_id": int(t.track_id),
                            "x1": float(l), "y1": float(t_y), "x2": float(r), "y2": float(b),
                            "time": elapsed_time
                        })
                except Exception as tracker_err:
                    print(f"\n[ERRORE] Errore critico nel loop delle tracce al frame {frame_number}: {tracker_err}")
                    traceback.print_exc()
                    raise tracker_err
                
                if progress_callback is not None:
                    progress_callback(frame_number, total_frames)
                    
        except Exception as main_err:
            print(f"\n[ERRORE] Fallimento nel metodo run: {main_err}")
            traceback.print_exc()
            raise main_err
        finally:
            print("Rilascio della risorsa VideoCapture in corso...")
            cap.release()
            
        return results
