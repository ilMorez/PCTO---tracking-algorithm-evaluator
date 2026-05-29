import streamlit as st
import pandas as pd
import json
from pathlib import Path
import cv2
from moviepy.video.io.VideoFileClip import VideoFileClip
from custom_trackers import TRACKER_REGISTRY
from visualization import Visualizer
from evaluator import TrackerEvaluator
from detection import Detector

st.set_page_config(page_title="Dashboard Comparazione Video", layout="wide")
st.title("Comparazione BBox Grezze vs Video Tracciato")
st.markdown("---")

OUTPUT_DIR = Path("output")
UPLOAD_DIR = Path("uploaded_videos")
OUTPUT_DIR.mkdir(exist_ok=True)
UPLOAD_DIR.mkdir(exist_ok=True)

def converti_in_h264(par_input_path: Path, par_output_path: Path):
    if not par_input_path.exists():
        return False

    try:
        clip = VideoFileClip(str(par_input_path))
        clip.write_videofile(str(par_output_path), codec="libx264", audio=False, logger=None)
        clip.close()
        return True

    except Exception as e:
        st.error(f"Errore durante la conversione con MoviePy: {e}")
        return False

def calcola_metriche_csv(csv_path):
    try:
        df_temp = pd.read_csv(csv_path)
        total_frames = int(df_temp["frame"].max() + 1) if not df_temp.empty else 1
        evaluator = TrackerEvaluator(par_total_frames=total_frames)
        return evaluator.evaluate(str(csv_path))
    except Exception:
        return None

def render_tracker_parameters(tracker_cls, tracker_key: str):
    specs = getattr(tracker_cls, "PARAMETER_SPECS", [])
    tracker_kwargs = {}

    if not specs:
        st.info("Questo tracker non espone parametri modificabili.")
        return tracker_kwargs

    st.subheader("Parametri del tracker")
    st.caption("Modifica i valori dei parametri prima di avviare l'elaborazione.")

    if len(specs) == 1:
        columns = [st.container()]
    else:
        columns = st.columns(2)

    for idx, spec in enumerate(specs):
        container = columns[idx % len(columns)]
        name = spec["name"]
        label = spec.get("label", name)
        default = spec.get("default")
        widget_key = f"{tracker_key}__{name}"
        field_type = spec.get("type", "text")

        min_value = spec.get("min")
        max_value = spec.get("max")
        step = spec.get("step")

        if field_type == "int":
            value = container.number_input(
                label,
                min_value=int(min_value) if min_value is not None else None,
                max_value=int(max_value) if max_value is not None else None,
                value=int(default) if default is not None else 0,
                step=int(step) if step is not None else 1,
                key=widget_key,
            )
        elif field_type == "float":
            value = container.number_input(
                label,
                min_value=float(min_value) if min_value is not None else None,
                max_value=float(max_value) if max_value is not None else None,
                value=float(default) if default is not None else 0.0,
                step=float(step) if step is not None else 0.01,
                format="%.4f",
                key=widget_key,
            )
        elif field_type == "bool":
            value = container.checkbox(
                label,
                value=bool(default),
                key=widget_key,
            )
        elif field_type == "select":
            options = spec.get("options", [])
            if not options:
                value = container.text_input(label, value="" if default is None else str(default), key=widget_key)
                value = value.strip() or None
            else:
                index = options.index(default) if default in options else 0
                value = container.selectbox(label, options, index=index, key=widget_key)
        else:
            value = container.text_input(
                label,
                value="" if default is None else str(default),
                key=widget_key,
                help="Lascia vuoto per passare None" if default is None else None,
            )
            value = value.strip() or None

        tracker_kwargs[name] = value

    return tracker_kwargs

uploaded_file = st.file_uploader("Carica un file video (.mp4):", type=["mp4"])
if uploaded_file is not None:

    video_salvato_path = UPLOAD_DIR / uploaded_file.name
    with open(video_salvato_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    video_output_folder = OUTPUT_DIR / video_salvato_path.stem
    video_output_folder.mkdir(parents=True, exist_ok=True)
    detections_json_path = video_output_folder / "detections.json"

    if not detections_json_path.exists():
        with st.spinner("YOLO sta estraendo le Bounding Box dal video..."):
            detector = Detector()
            detector.run_detection(str(video_salvato_path), str(detections_json_path))

    with open(detections_json_path, "r") as f:
        detections_data = json.load(f)

    col_video_or, col_video_dett = st.columns(2)

    with col_video_or:
        st.markdown(f'Video: "{video_salvato_path.stem}"')
        st.video(video_salvato_path)

    with col_video_dett:
        cap = cv2.VideoCapture(str(video_salvato_path))
        fps = cap.get(cv2.CAP_PROP_FPS)
        frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        total_detections = sum(len(f["detections"]) for f in detections_data)
        st.metric(label="Durata video", value=f"{(frames / fps):.2f} s")
        st.metric(label="Frame Rate", value=f"{fps:.2f} FPS")
        st.metric(label="Frame Totali", value=f"{frames} frames")
        st.metric(label="Dimensioni", value=f"{int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}X{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))} px")
        st.metric(label="Detections totali", value=f"{total_detections}")
        cap.release()

    tracker_type = st.selectbox(
        "Seleziona l'algoritmo da applicare:",
        list(TRACKER_REGISTRY.keys())
    )

    TrackerClass = TRACKER_REGISTRY[tracker_type]
    
    tracker_kwargs = render_tracker_parameters(TrackerClass, tracker_key=tracker_type.lower())

    video_raw_path = video_output_folder / "raw_detections.mp4"
    video_tracked_filename = f"{tracker_type.lower()}.mp4"
    video_tracked_path = video_output_folder / video_tracked_filename
    video_raw_converted = video_output_folder / "raw_detections_h264.mp4"
    video_tracked_converted = video_output_folder / f"{tracker_type.lower()}_h264.mp4"

    if st.button("Avvia Elaborazione"):
        with st.spinner(f"Applicazione di {tracker_type} e generazione video in corso..."):
            tracker_inst = TrackerClass(**tracker_kwargs)
            tracker_inst.setup_video_paths(video_salvato_path.name)
            risultati = tracker_inst.run(detections_data, str(video_salvato_path))
            tracker_inst.save_to_csv(risultati)

            visualizer = Visualizer(par_output_dir=str(video_output_folder))
            print("Annotazione video tracker")
            visualizer.draw_tracks(str(tracker_inst.csv_path), video_tracked_filename, str(video_salvato_path))
            converti_in_h264(video_tracked_path, video_tracked_converted)
            print("Annotazione video YOLO")
            visualizer.draw_raw_detections(detections_data, "raw_detections.mp4", video_salvato_path)
            converti_in_h264(video_raw_path, video_raw_converted)

    v_col1, v_col2 = st.columns(2)
    with v_col1:
        st.markdown("**Rilevamenti Grezzi (YOLO - Box Blu):**")
        if video_raw_converted.exists():
            st.video(str(video_raw_converted))
        else:
            st.caption("Avvia l'elaborazione per generare l'output visivo.")

    with v_col2:
        st.markdown(f"**Video Tracciato ({tracker_type} - Box Verdi + ID):**")
        if video_tracked_converted.exists():
            st.video(str(video_tracked_converted))
        else:
            st.caption("Avvia l'elaborazione per generare l'output visivo.")

    st.markdown("---")
    csv_corrente = video_output_folder / f"{tracker_type.lower()}.csv"
    if csv_corrente.exists():
        st.subheader("Metriche di Performance Calcolate")
        metriche = calcola_metriche_csv(csv_corrente)
        if metriche:
            st.dataframe(pd.DataFrame([metriche], index=[tracker_type.upper()]))