import cv2
import numpy as np
import time
from BaseTracker import BaseTracker
from config import FRAME_SKIP

try:
    import trackers as tk
    import supervision as sv
    TRACKERS_LIB_AVAILABLE = True
except ImportError:
    TRACKERS_LIB_AVAILABLE = False

class ByteTrackTracker(BaseTracker):
    """ Implementazione di ByteTrack, un tracker veloce e accurato che ha vinto in MOT20/21 benchmark.
        Novità principali:
            - usa due associazioni: HIGH-CONFIDENCE e LOW-CONFIDENCE detection
            - recupera tracce "morte" grazie alla low-confidence detection
        Si tratta di un tracker veloce e robusto"""
    PARAMETER_SPECS = [
        {'name': 'par_lost_track_buffer', 'label': 'Lost Track Buffer (frames)', 'type': 'int', 'default': 30, 'min': 1, 'max': 300, 'step': 1},
        {'name': 'par_frame_rate', 'label': 'Frame Rate (FPS)', 'type': 'float', 'default': 30.0, 'min': 1.0, 'max': 120.0, 'step': 1.0},
        {'name': 'par_track_activation_threshold', 'label': 'Track Activation Threshold', 'type': 'float', 'default': 0.7, 'min': 0.0, 'max': 1.0, 'step': 0.05},
        {'name': 'par_minimum_consecutive_frames', 'label': 'Min Consecutive Frames', 'type': 'int', 'default': 2, 'min': 1, 'max': 30, 'step': 1},
        {'name': 'par_minimum_iou_threshold', 'label': 'Min IoU Threshold', 'type': 'float', 'default': 0.1, 'min': 0.0, 'max': 1.0, 'step': 0.05},
        {'name': 'par_high_conf_det_threshold', 'label': 'High Conf Det Threshold', 'type': 'float', 'default': 0.6, 'min': 0.0, 'max': 1.0, 'step': 0.05}
    ]

    def __init__(
        self, 
        par_lost_track_buffer=30, 
        par_frame_rate=30.0, 
        par_track_activation_threshold=0.7, 
        par_minimum_consecutive_frames=2, 
        par_minimum_iou_threshold=0.1, 
        par_high_conf_det_threshold=0.6
    ):
        super().__init__("ByteTrack")
        if TRACKERS_LIB_AVAILABLE:
            self.tracker = tk.ByteTrackTracker(
                lost_track_buffer=par_lost_track_buffer,
                frame_rate=par_frame_rate,
                track_activation_threshold=par_track_activation_threshold,
                minimum_consecutive_frames=par_minimum_consecutive_frames,
                minimum_iou_threshold=par_minimum_iou_threshold,
                high_conf_det_threshold=par_high_conf_det_threshold
            )
        else:
            self.tracker = None
    
    def run(self, par_detection_data: list, par_video_path: str, progress_callback=None) -> list:
        return _run_supervision_tracker(self.tracker, par_detection_data, progress_callback)

class OCSortTracker(BaseTracker):
    """ Implementazione di OC-SORT, un tracker che va a migliorare SORT classico usando l'osservazione diretta:
        va ad utilizzare solo la geometria, niente feature, utilizzando una strategia di associazione migliore rispetto al classico SORT.
        Si tratta di un tracekr veloce e accurato come bytetrack, non utilizza il deep learning"""
    PARAMETER_SPECS = [
        {'name': 'par_lost_track_buffer', 'label': 'Lost Track Buffer (frames)', 'type': 'int', 'default': 30, 'min': 1, 'max': 300, 'step': 1},
        {'name': 'par_frame_rate', 'label': 'Frame Rate (FPS)', 'type': 'float', 'default': 30.0, 'min': 1.0, 'max': 120.0, 'step': 1.0},
        {'name': 'par_minimum_consecutive_frames', 'label': 'Min Consecutive Frames', 'type': 'int', 'default': 3, 'min': 1, 'max': 30, 'step': 1},
        {'name': 'par_minimum_iou_threshold', 'label': 'Min IoU Threshold', 'type': 'float', 'default': 0.3, 'min': 0.0, 'max': 1.0, 'step': 0.05},
        {'name': 'par_direction_consistency_weight', 'label': 'Direction Consistency Weight', 'type': 'float', 'default': 0.2, 'min': 0.0, 'max': 1.0, 'step': 0.05},
        {'name': 'par_high_conf_det_threshold', 'label': 'High Conf Det Threshold', 'type': 'float', 'default': 0.6, 'min': 0.0, 'max': 1.0, 'step': 0.05},
        {'name': 'par_delta_t', 'label': 'Delta T', 'type': 'int', 'default': 3, 'min': 1, 'max': 20, 'step': 1}
    ]

    def __init__(
        self,
        par_lost_track_buffer=30,
        par_frame_rate=30.0,
        par_minimum_consecutive_frames=3,
        par_minimum_iou_threshold=0.3,
        par_direction_consistency_weight=0.2,
        par_high_conf_det_threshold=0.6,
        par_delta_t=3
    ):
        super().__init__("OC-SORT")
        if TRACKERS_LIB_AVAILABLE:
            self.tracker = tk.OCSORTTracker(
                lost_track_buffer=par_lost_track_buffer,
                frame_rate=par_frame_rate,
                minimum_consecutive_frames=par_minimum_consecutive_frames,
                minimum_iou_threshold=par_minimum_iou_threshold,
                direction_consistency_weight=par_direction_consistency_weight,
                high_conf_det_threshold=par_high_conf_det_threshold,
                delta_t=par_delta_t
            )
        else:
            self.tracker = None
    def run(self, par_detection_data: list, par_video_path: str, progress_callback=None) -> list:
        return _run_supervision_tracker(self.tracker, par_detection_data, progress_callback)

class SortTracker(BaseTracker):
    """ Implementazione di SORT (Simple Online and Realtime Tracking), l'algorimto classico:
            1. Ungherian algorithm per associazione centroidi
            2. Kalman filter per previsione
        Si tratta di un tracker smeplice e veloce, ma è sensibile all'occlusion e porta a ID switch frequenti"""
    PARAMETER_SPECS = [
        {'name': 'par_lost_track_buffer', 'label': 'Lost Track Buffer (frames)', 'type': 'int', 'default': 30, 'min': 1, 'max': 300, 'step': 1},
        {'name': 'par_frame_rate', 'label': 'Frame Rate (FPS)', 'type': 'float', 'default': 30.0, 'min': 1.0, 'max': 120.0, 'step': 1.0},
        {'name': 'par_track_activation_threshold', 'label': 'Track Activation Threshold', 'type': 'float', 'default': 0.25, 'min': 0.0, 'max': 1.0, 'step': 0.05},
        {'name': 'par_minimum_consecutive_frames', 'label': 'Min Consecutive Frames', 'type': 'int', 'default': 3, 'min': 1, 'max': 30, 'step': 1},
        {'name': 'par_minimum_iou_threshold', 'label': 'Min IoU Threshold', 'type': 'float', 'default': 0.3, 'min': 0.0, 'max': 1.0, 'step': 0.05}
    ]

    def __init__(
        self,
        par_lost_track_buffer=30,
        par_frame_rate=30.0,
        par_track_activation_threshold=0.25,
        par_minimum_consecutive_frames=3,
        par_minimum_iou_threshold=0.3
    ):
        super().__init__("SORT")
        if TRACKERS_LIB_AVAILABLE:
            self.tracker = tk.SORTTracker(
                lost_track_buffer=par_lost_track_buffer,
                frame_rate=par_frame_rate,
                track_activation_threshold=par_track_activation_threshold,
                minimum_consecutive_frames=par_minimum_consecutive_frames,
                minimum_iou_threshold=par_minimum_iou_threshold
            )
        else:
            self.tracker = None
    def run(self, par_detection_data: list, par_video_path: str, progress_callback=None) -> list:
        return _run_supervision_tracker(self.tracker, par_detection_data, progress_callback)

class BoTSortTracker(BaseTracker):
    """ Implementazione di BoT-SORT, un'estensione di SORT con feature di apaprenza tramite un feature exctractor.
        Rispetto a SORT:
            1 usa feature di apparenza estratte da una rete neurale
            2. combina la distanza geometrica e la feature similarity
            3. ha accesso al frame grezzo proprio come DeepSORT
        Si tratta di un buon equilibrio tra velocità ed accuratezza, andando però a richeidere una GPU per avere prestazioni elevate."""
    PARAMETER_SPECS = [
        {'name': 'par_lost_track_buffer', 'label': 'Lost Track Buffer (frames)', 'type': 'int', 'default': 30, 'min': 1, 'max': 300, 'step': 1},
        {'name': 'par_frame_rate', 'label': 'Frame Rate (FPS)', 'type': 'float', 'default': 30.0, 'min': 1.0, 'max': 120.0, 'step': 1.0},
        {'name': 'par_track_activation_threshold', 'label': 'Track Activation Threshold', 'type': 'float', 'default': 0.7, 'min': 0.0, 'max': 1.0, 'step': 0.05},
        {'name': 'par_minimum_consecutive_frames', 'label': 'Min Consecutive Frames', 'type': 'int', 'default': 2, 'min': 1, 'max': 30, 'step': 1},
        {'name': 'par_minimum_iou_threshold_first_assoc', 'label': 'Min IoU First Association', 'type': 'float', 'default': 0.2, 'min': 0.0, 'max': 1.0, 'step': 0.05},
        {'name': 'par_minimum_iou_threshold_second_assoc', 'label': 'Min IoU Second Association', 'type': 'float', 'default': 0.5, 'min': 0.0, 'max': 1.0, 'step': 0.05},
        {'name': 'par_minimum_iou_threshold_unconfirmed_assoc', 'label': 'Min IoU Unconfirmed Association', 'type': 'float', 'default': 0.3, 'min': 0.0, 'max': 1.0, 'step': 0.05},
        {'name': 'par_high_conf_det_threshold', 'label': 'High Conf Det Threshold', 'type': 'float', 'default': 0.6, 'min': 0.0, 'max': 1.0, 'step': 0.05},
        {'name': 'par_enable_cmc', 'label': 'Enable CMC (Camera Motion Comp.)', 'type': 'bool', 'default': True},
        {'name': 'par_cmc_method', 'label': 'CMC Method', 'type': 'select', 'default': 'sparseOptFlow', 'options': ['sparseOptFlow', 'orb', 'ecc']},
        {'name': 'par_cmc_downscale', 'label': 'CMC Downscale', 'type': 'int', 'default': 2, 'min': 1, 'max': 8, 'step': 1},
        {'name': 'par_instant_first_frame_activation', 'label': 'Instant First Frame Activation', 'type': 'bool', 'default': True}
    ]

    def __init__(
        self,
        par_lost_track_buffer=30,
        par_frame_rate=30.0,
        par_track_activation_threshold=0.7,
        par_minimum_consecutive_frames=2,
        par_minimum_iou_threshold_first_assoc=0.2,
        par_minimum_iou_threshold_second_assoc=0.5,
        par_minimum_iou_threshold_unconfirmed_assoc=0.3,
        par_high_conf_det_threshold=0.6,
        par_enable_cmc=True,
        par_cmc_method="sparseOptFlow",
        par_cmc_downscale=2,
        par_instant_first_frame_activation=True
    ):
        super().__init__("BoT-SORT")
        if TRACKERS_LIB_AVAILABLE:
            self.tracker = tk.BoTSORTTracker(
                lost_track_buffer=par_lost_track_buffer,
                frame_rate=par_frame_rate,
                track_activation_threshold=par_track_activation_threshold,
                minimum_consecutive_frames=par_minimum_consecutive_frames,
                minimum_iou_threshold_first_assoc=par_minimum_iou_threshold_first_assoc,
                minimum_iou_threshold_second_assoc=par_minimum_iou_threshold_second_assoc,
                minimum_iou_threshold_unconfirmed_assoc=par_minimum_iou_threshold_unconfirmed_assoc,
                high_conf_det_threshold=par_high_conf_det_threshold,
                enable_cmc=par_enable_cmc,
                cmc_method=par_cmc_method,
                cmc_downscale=par_cmc_downscale,
                instant_first_frame_activation=par_instant_first_frame_activation
            )
        else:
            self.tracker = None
    def run(self, par_detection_data: list, par_video_path: str, progress_callback=None) -> list:
        if self.tracker is None: return []
        results, start_time = [], time.time()
        cap = cv2.VideoCapture(par_video_path)
        total_frames = len(par_detection_data)
        for frame_data in par_detection_data:
            ret, frame = cap.read()
            if not ret: break
            frame_number = frame_data["frame_id"]
            if frame_number % FRAME_SKIP != 0: continue
            dets = frame_data["detections"]
            sv_dets = sv.Detections.empty() if not dets else sv.Detections(xyxy=np.array([d[:4] for d in dets], dtype=np.float32), confidence=np.array([d[4] for d in dets], dtype=np.float32), class_id=np.zeros(len(dets), dtype=int))
            tracked_dets = self.tracker.update(sv_dets, frame=frame)
            if tracked_dets is not None and tracked_dets.tracker_id is not None:
                for i in range(len(tracked_dets.xyxy)):
                    x1, y1, x2, y2 = tracked_dets.xyxy[i]
                    results.append({"frame": frame_number, "track_id": int(tracked_dets.tracker_id[i]), "x1": float(x1), "y1": float(y1), "x2": float(x2), "y2": float(y2), "time": time.time() - start_time})
            if progress_callback is not None:
                progress_callback(frame_number, total_frames)
        cap.release()
        return results

def _run_supervision_tracker(tracker_instance, par_detection_data, progress_callback=None):
    if tracker_instance is None: return []
    results, start_time = [], time.time()
    total_frames = len(par_detection_data)
    for frame_data in par_detection_data:
        frame_number = frame_data["frame_id"]
        if frame_number % FRAME_SKIP != 0: continue
        dets = frame_data["detections"]
        sv_dets = sv.Detections.empty() if not dets else sv.Detections(xyxy=np.array([d[:4] for d in dets], dtype=np.float32), confidence=np.array([d[4] for d in dets], dtype=np.float32), class_id=np.zeros(len(dets), dtype=int))
        tracked_dets = tracker_instance.update(sv_dets)
        if tracked_dets is not None and tracked_dets.tracker_id is not None:
            for i in range(len(tracked_dets.xyxy)):
                x1, y1, x2, y2 = tracked_dets.xyxy[i]
                results.append({"frame": frame_number, "track_id": int(tracked_dets.tracker_id[i]), "x1": float(x1), "y1": float(y1), "x2": float(x2), "y2": float(y2), "time": time.time() - start_time})
        if progress_callback is not None:
            progress_callback(frame_number, total_frames)
    return results
