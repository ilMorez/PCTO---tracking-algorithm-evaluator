from pathlib import Path


MODEL_PATH   = "yolo26n.pt"
VIDEO_LIST   = [
    "1.mp4",
    #"2.mp4",
    #"4.mp4",
    #"5.mp4",
    #"4608285-uhd_3840_2160_24fps.mp4",
    #"6734146-hd_1920_1080_24fps.mp4",
    #"14702860_3840_2160_50fps.mp4",
    #"2165-155327596 - Trim.mp4",
    #"28293-369325244.mp4",
    #"28294-369325253.mp4",
    #"42679-432102847.mp4",
    #"14702862_3840_2160_50fps.mp4"
    ]
TARGET_CLASS = "car"
FRAME_SKIP   = 1
OUTPUT_DIR      = Path("output")
DETECTIONS_FILE = OUTPUT_DIR / "detections.json"