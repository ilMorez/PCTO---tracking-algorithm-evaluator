import io
import base64
import zipfile
import json
from pathlib import Path
from datetime import datetime

import cv2
import pandas as pd
import requests
import streamlit as st

from custom_trackers import TRACKER_REGISTRY
from evaluator import TrackerEvaluator

st.set_page_config(page_title="Dashboard Comparazione Video", layout="wide")
st.title("Comparazione BBox Grezze vs Video Tracciato")
st.markdown("---")

OUTPUT_DIR = Path("output")
UPLOAD_DIR = Path("uploaded_videos")
OUTPUT_DIR.mkdir(exist_ok=True)
UPLOAD_DIR.mkdir(exist_ok=True)


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("Configurazione Server")
    SERVER_URL = st.text_input(
        "URL del server",
        value="http://localhost:8000",
        help="Indirizzo base del server FastAPI",
    ).rstrip("/")

    st.markdown("---")
    st.header("Modello YOLO")
    YOLO_MODELS = [
        "yolo26n.pt",
        "yolov8n.pt", "yolov8s.pt", "yolov8m.pt", "yolov8l.pt", "yolov8x.pt",
        "yolov9c.pt", "yolov9e.pt",
        "yolo11n.pt", "yolo11s.pt", "yolo11m.pt", "yolo11l.pt", "yolo11x.pt",
    ]
    selected_model = st.selectbox("Modello YOLO", YOLO_MODELS, index=0)
    custom_model   = st.text_input("Oppure path/nome modello custom", value="").strip()
    model_name     = custom_model if custom_model else selected_model

    st.markdown("##### Parametri inferenza")
    yolo_conf    = st.slider("Confidence threshold", 0.01, 1.0, 0.25, 0.01)
    yolo_iou     = st.slider("IoU threshold (NMS)",  0.01, 1.0, 0.45, 0.01)
    yolo_imgsz   = st.selectbox("imgsz", [320, 416, 480, 640, 768, 1024, 1280], index=3)
    yolo_half    = st.checkbox("Half precision (FP16)", value=False)
    yolo_verbose = st.checkbox("Verbose YOLO", value=False)

    yolo_params = {
        "conf": yolo_conf, "iou": yolo_iou, "imgsz": yolo_imgsz,
        "half": yolo_half, "stream": True, "verbose": yolo_verbose,
    }


# ---------------------------------------------------------------------------
# Chiamate server
# ---------------------------------------------------------------------------

def server_detect(video_path: Path, model_name: str, yolo_params: dict):
    """Ritorna (detections, video_bytes_or_None)."""
    endpoint = f"{SERVER_URL}/api/detect"
    try:
        with open(video_path, "rb") as vf:
            resp = requests.post(
                endpoint,
                data={"model_name": model_name, "yolo_params_json": json.dumps(yolo_params)},
                files={"video": (video_path.name, vf, "video/mp4")},
                timeout=600,
                proxies={"http": None, "https": None},
            )
        if resp.status_code != 200:
            try:
                detail = resp.json().get("detail", resp.text)
            except Exception:
                detail = resp.text
            st.error(f"Errore `/api/detect` [{resp.status_code}]:\n```\n{detail[:1000]}\n```")
            return None, None

        data        = resp.json()
        detections  = data.get("detections")
        yolo_b64    = data.get("yolo_video")
        video_bytes = base64.b64decode(yolo_b64) if yolo_b64 else None
        return detections, video_bytes

    except requests.exceptions.ConnectionError:
        st.error(f"Impossibile connettersi al server: `{endpoint}`")
        return None, None
    except Exception as e:
        st.error(f"Errore `/api/detect`:\n```\n{e}\n```")
        return None, None


def server_track(video_path: Path, tracker_name: str, detections_data: list, tracker_params: dict):
    """Ritorna (tracking_results_list, video_bytes, csv_bytes) oppure (None, None, None)."""
    endpoint = f"{SERVER_URL}/api/track"
    try:
        with open(video_path, "rb") as vf:
            resp = requests.post(
                endpoint,
                data={
                    "tracker_name":        tracker_name,
                    "detections_json":     json.dumps(detections_data),
                    "tracker_params_json": json.dumps(tracker_params),
                },
                files={"video": (video_path.name, vf, "video/mp4")},
                timeout=600,
                proxies={"http": None, "https": None},
            )

        if resp.status_code != 200:
            try:
                detail = resp.json().get("detail", resp.text)
            except Exception:
                detail = resp.text
            st.error(f"Errore `/api/track` [{resp.status_code}]:\n```\n{detail[:1000]}\n```")
            return None, None, None

        # Risposta è uno zip
        zf        = zipfile.ZipFile(io.BytesIO(resp.content))
        names     = zf.namelist()
        mp4_name  = next((n for n in names if n.endswith(".mp4")), None)
        csv_name  = next((n for n in names if n.endswith(".csv")), None)
        video_bytes = zf.read(mp4_name) if mp4_name else None
        csv_bytes   = zf.read(csv_name) if csv_name else None

        # Parsing CSV → lista risultati (per compatibilità con tab2)
        results = []
        if csv_bytes:
            import csv as _csv
            reader = _csv.DictReader(
                line for line in csv_bytes.decode().splitlines()
                if not line.startswith("#")
            )
            for row in reader:
                results.append({k: (int(v) if k in ("frame", "track_id") else float(v)) for k, v in row.items()})

        return results, video_bytes, csv_bytes

    except requests.exceptions.ConnectionError:
        st.error(f"Impossibile connettersi al server: `{endpoint}`")
        return None, None, None
    except Exception as e:
        st.error(f"Errore `/api/track`:\n```\n{e}\n```")
        return None, None, None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def calcola_metriche_csv(csv_path: Path):
    try:
        df_temp      = pd.read_csv(csv_path, comment="#")
        total_frames = int(df_temp["frame"].max() + 1) if not df_temp.empty else 1
        evaluator    = TrackerEvaluator(par_total_frames=total_frames)
        return evaluator.evaluate(str(csv_path))
    except Exception:
        return None


def render_tracker_parameters(tracker_cls, tracker_key: str):
    specs          = getattr(tracker_cls, "PARAMETER_SPECS", [])
    tracker_kwargs = {}

    if not specs:
        st.info("Questo tracker non espone parametri modificabili.")
        return tracker_kwargs

    st.subheader("Parametri del tracker")
    st.caption("Modifica i valori dei parametri prima di avviare l'elaborazione.")

    columns = [st.container()] if len(specs) == 1 else st.columns(2)

    for idx, spec in enumerate(specs):
        container  = columns[idx % len(columns)]
        name       = spec["name"]
        label      = spec.get("label", name)
        default    = spec.get("default")
        widget_key = f"{tracker_key}__{name}"
        field_type = spec.get("type", "text")
        min_value  = spec.get("min")
        max_value  = spec.get("max")
        step       = spec.get("step")

        if field_type == "int":
            value = container.number_input(label, min_value=int(min_value) if min_value is not None else None,
                                           max_value=int(max_value) if max_value is not None else None,
                                           value=int(default) if default is not None else 0,
                                           step=int(step) if step is not None else 1, key=widget_key)
        elif field_type == "float":
            value = container.number_input(label, min_value=float(min_value) if min_value is not None else None,
                                           max_value=float(max_value) if max_value is not None else None,
                                           value=float(default) if default is not None else 0.0,
                                           step=float(step) if step is not None else 0.01,
                                           format="%.4f", key=widget_key)
        elif field_type == "bool":
            value = container.checkbox(label, value=bool(default), key=widget_key)
        elif field_type == "select":
            options = spec.get("options", [])
            if not options:
                value = container.text_input(label, value="" if default is None else str(default), key=widget_key)
                value = value.strip() or None
            else:
                index = options.index(default) if default in options else 0
                value = container.selectbox(label, options, index=index, key=widget_key)
        else:
            value = container.text_input(label, value="" if default is None else str(default), key=widget_key,
                                         help="Lascia vuoto per passare None" if default is None else None)
            value = value.strip() or None

        tracker_kwargs[name] = value

    return tracker_kwargs


# ---------------------------------------------------------------------------
# UI principale
# ---------------------------------------------------------------------------
uploaded_file = st.file_uploader("Carica un file video (.mp4):", type=["mp4"])

if uploaded_file is not None:
    peso           = uploaded_file.size
    video_stem     = Path(uploaded_file.name).stem
    nuovo_nome     = f"{video_stem}_{peso}.mp4"
    video_salvato_path = UPLOAD_DIR / nuovo_nome

    with open(video_salvato_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    model_slug          = model_name.replace(".pt", "").replace("/", "_").replace("\\", "_")
    video_output_folder = OUTPUT_DIR / f"{video_salvato_path.stem}__{model_slug}"
    video_output_folder.mkdir(parents=True, exist_ok=True)

    detections_json_path = video_output_folder / "detections.json"
    yolo_video_path      = video_output_folder / "raw_detections_h264.mp4"

    # --- Detection (con cache) ---
    if not detections_json_path.exists():
        with st.spinner(f"Detection in corso sul server con **{model_name}**…"):
            detections_data, yolo_video_bytes = server_detect(video_salvato_path, model_name, yolo_params)
        if detections_data is None:
            st.stop()

        with open(detections_json_path, "w") as f:
            json.dump(detections_data, f, indent=4)

        if yolo_video_bytes and not yolo_video_path.exists():
            yolo_video_path.write_bytes(yolo_video_bytes)

    with open(detections_json_path, "r") as f:
        detections_data = json.load(f)

    # --- Info video ---
    col_video_or, col_video_dett = st.columns(2)
    with col_video_or:
        st.video(str(video_salvato_path))
    with col_video_dett:
        cap = cv2.VideoCapture(str(video_salvato_path))
        fps    = cap.get(cv2.CAP_PROP_FPS)
        frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        total_detections = sum(len(f["detections"]) for f in detections_data)
        st.metric("Durata video",    f"{(frames / fps):.2f} s")
        st.metric("Frame Rate",      f"{fps:.2f} FPS")
        st.metric("Frame Totali",    f"{frames} frames")
        st.metric("Dimensioni",      f"{int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}X{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))} px")
        st.metric("Detections totali", str(total_detections))
        cap.release()

    st.markdown("---")

    tab1, tab2 = st.tabs(["Tracking Test", "Comparazione Analytics"])

    # --- TAB 1 ---
    with tab1:
        tracker_type = st.selectbox("Seleziona l'algoritmo da applicare:", list(TRACKER_REGISTRY.keys()))
        TrackerClass   = TRACKER_REGISTRY[tracker_type]
        tracker_kwargs = render_tracker_parameters(TrackerClass, tracker_key=tracker_type.lower())

        video_tracked_converted = video_output_folder / f"{tracker_type.lower()}_h264.mp4"

        if st.button("Avvia Elaborazione"):
            with st.spinner(f"Tracking **{tracker_type}** in corso sul server (video + CSV)…"):
                risultati, tracked_bytes, csv_bytes = server_track(
                    video_salvato_path, tracker_type, detections_data, tracker_kwargs,
                )

            if risultati is None:
                st.stop()

            # Salva video tracciato
            if tracked_bytes:
                video_tracked_converted.write_bytes(tracked_bytes)

            # Salva CSV con timestamp (compatibile con tab2)
            if csv_bytes:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                csv_path  = video_output_folder / f"{tracker_type.lower()}_{timestamp}.csv"
                csv_path.write_bytes(csv_bytes)

        v_col1, v_col2 = st.columns(2)
        with v_col1:
            st.markdown("**Rilevamenti Grezzi (YOLO - Box Blu):**")
            if yolo_video_path.exists():
                st.video(str(yolo_video_path))
            else:
                st.caption("Avvia l'elaborazione per generare l'output visivo.")
        with v_col2:
            st.markdown(f"**Video Tracciato ({tracker_type} - Box Verdi + ID):**")
            if video_tracked_converted.exists():
                st.video(str(video_tracked_converted))
            else:
                st.caption("Avvia l'elaborazione per generare l'output visivo.")

    # --- TAB 2 ---
    with tab2:
        st.subheader("Analisi Performance dei Tracker Elaborati")

        metriche_globali = {}
        for nome_tracker in TRACKER_REGISTRY.keys():
            csv_files = sorted(video_output_folder.glob(f"{nome_tracker.lower()}_*.csv"))
            if csv_files:
                metriche = calcola_metriche_csv(csv_files[-1])
                if metriche:
                    metriche_globali[nome_tracker.upper()] = metriche

        if metriche_globali:
            df_comparativo = pd.DataFrame.from_dict(metriche_globali, orient="index")

            st.markdown("##### Tabella Riassuntiva")
            st.dataframe(df_comparativo)

            st.markdown("---")
            st.markdown("##### Grafici Comparativi delle Metriche")

            mappa_nomi_metriche = {
                "id_switches": "ID Switches", "fragmentation": "Fragmentation",
                "kinematic_jumps": "Kinematic Jumps", "track_coverage": "Track Coverage %",
                "time": "Tempo di Elaborazione in secondi",
                "avg_track_length": "Lunghezza Media Tracce in frame",
                "max_track_length": "Lunghezza Massima Tracce in frame",
                "num_tracks": "Numero di Tracce Uniche Rilevate",
                "avg_id_lifetime": "Durata Vita Media ID in frame",
                "max_id_lifetime": "Durata Vita Massima ID in frame",
                "total_detections": "Rilevamenti Totali Tracciati",
                "spurious_tracks_ratio": "Rapporto Tracce Spurie",
            }
            lista_colori = [
                "#06b6d4","#ea580c","#7c3aed","#10b981","#e11d48",
                "#2563eb","#d946ef","#84cc16","#f59e0b","#4f46e5",
                "#f43f5e","#34A853",
            ]
            colonne_disponibili = list(df_comparativo.columns)
            for i in range(0, len(colonne_disponibili), 2):
                c1, c2  = st.columns(2)
                col_1   = colonne_disponibili[i]
                label_1 = mappa_nomi_metriche.get(col_1, col_1.replace("_", " ").title())
                with c1:
                    st.markdown(f"##### {label_1}")
                    st.bar_chart(df_comparativo[col_1], color=lista_colori[i % len(lista_colori)], height=450)
                if i + 1 < len(colonne_disponibili):
                    col_2   = colonne_disponibili[i + 1]
                    label_2 = mappa_nomi_metriche.get(col_2, col_2.replace("_", " ").title())
                    with c2:
                        st.markdown(f"##### {label_2}")
                        st.bar_chart(df_comparativo[col_2], color=lista_colori[(i+1) % len(lista_colori)], height=450)
        else:
            st.info("Nessuna metrica disponibile. Torna nel tab 'Uso dei Tracker' ed avvia l'elaborazione di almeno un algoritmo per popolare i grafici.")
