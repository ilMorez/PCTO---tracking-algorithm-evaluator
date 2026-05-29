from .deep_sort_tracker import DeepSortTracker
from .norfair_tracker import NorfairTracker
from .motpy_tracker import MotpyTracker
from .optical_flow_tracker import OpticalFlowTracker
from .hungarian_tracker import HungarianTracker
from .appearance_tracker import AppearanceTracker
from .velocity_tracker import VelocityTracker
from .external_libs_trackers import ByteTrackTracker, OCSortTracker, SortTracker, BoTSortTracker

# Registro globale accessibile da app.py
TRACKER_REGISTRY = {
    "DeepSORT": DeepSortTracker,
    "ByteTrack": ByteTrackTracker,
    "OC-SORT": OCSortTracker,
    "SORT": SortTracker,
    "BoT-SORT": BoTSortTracker,
    "Norfair": NorfairTracker,
    "Motpy": MotpyTracker,
    "Hungarian": HungarianTracker,
    "OpticalFlow": OpticalFlowTracker,
    "Appearance": AppearanceTracker,
    "Velocity": VelocityTracker
}