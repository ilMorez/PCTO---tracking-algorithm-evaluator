import streamlit as st
import pandas as pd
import json
from pathlib import Path
import cv2
from moviepy.video.io.VideoFileClip import VideoFileClip
from custom_trackers import TRACKER_REGISTRY
from visualization import Visualizer
from evaluator import TrackerEvaluator
import requests

st.set_page_config(page_title="Dashboard Comparazione Video", layout="wide")
st.title("Comparazione BBox Grezze vs Video Tracciato")
st.markdown("---")

OUTPUT_DIR = Path("output")
UPLOAD_DIR = Path("uploaded_videos")
OUTPUT_DIR.mkdir(exist_ok=True)
UPLOAD_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# Configurazione server (sidebar) — unica aggiunta alla grafica originale
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("Configurazione Server")
    SERVER_URL = st.text_input(
        "URL del server",
        value="http://localhost:8000",
        help="Indirizzo base del server FastAPI"
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
    custom_model = st.text_input(
        "Oppure path/nome modello custom",
        value="",
        help="Lascia vuoto per usare il modello selezionato sopra"
    ).strip()
    model_name = custom_model if custom_model else selected_model

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

def server_detect(video_path: Path, model_name: str, yolo_params: dict) -> list | None:
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
        try:
            data = resp.json()
        except Exception:
            st.error(f"Risposta non JSON [{resp.status_code}]:\n```\n{resp.text[:1000]}\n```")
            return None
        if resp.status_code != 200:
            st.error(f"Errore `/api/detect` [{resp.status_code}]:\n```\n{data.get('detail', resp.text)}\n```")
            return None
        return data.get("detections")
    except requests.exceptions.ConnectionError:
        st.error(f"Impossibile connettersi al server: `{endpoint}`")
        return None
    except Exception as e:
        st.error(f"Errore `/api/detect`:\n```\n{e}\n```")
        return None


def server_track(video_path: Path, tracker_name: str, detections_data: list, tracker_params: dict):
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
        try:
            data = resp.json()
        except Exception:
            st.error(
                f"Il server ha risposto con stato **{resp.status_code}** ma la risposta non è JSON.\n\n"
                f"**Risposta grezza:**\n```\n{resp.text[:1000]}\n```"
            )
            return None
        if resp.status_code != 200:
            st.error(f"Errore server `/api/track` [{resp.status_code}]:\n```\n{data.get('detail', resp.text)}\n```")
            return None
        return data.get("tracking_results")
    except requests.exceptions.ConnectionError:
        st.error(f"Impossibile connettersi al server: `{endpoint}`\nVerifica che sia avviato e che l'URL sia corretto.")
        return None
    except Exception as e:
        st.error(f"Errore imprevisto nella chiamata `/api/track`:\n```\n{e}\n```")
        return None

def converti_in_h264(par_input_path: Path, par_output_path: Path, desc="Conversione Video"):
    if not par_input_path.exists():
        return False

    try:
        from proglog import ProgressBarLogger
        
        status_text = st.empty()
        progress_bar = st.progress(0.0)
        
        class StreamlitMoviePyLogger(ProgressBarLogger):
            def bars_callback(self, bar, attr, value, old_value=None):
                if bar in self.bars:
                    total = self.bars[bar].get('total', 0)
                    if total > 0:
                        percentage = value / total
                        progress_bar.progress(min(float(percentage), 1.0))
                        status_text.text(f"{desc}: {value}/{total} frame ({int(percentage * 100)}%)")
        custom_logger = StreamlitMoviePyLogger()
        
        clip = VideoFileClip(str(par_input_path))
        clip.write_videofile(
            str(par_output_path), 
            codec="libx264", 
            audio=False, 
            logger=custom_logger
        )
        clip.close()
        
        status_text.empty()
        progress_bar.empty()
        return True

    except Exception as e:
        st.error(f"Errore durante la conversione con MoviePy: {e}")
        return False

def calcola_metriche_csv(csv_path):
    try:
        df_temp = pd.read_csv(csv_path, comment='#')
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

# ---------------------------------------------------------------------------
# UI principale — identica all'originale
# ---------------------------------------------------------------------------
uploaded_file = st.file_uploader("Carica un file video (.mp4):", type=["mp4"])

if uploaded_file is not None:
    peso = uploaded_file.size
    video_stem = Path(uploaded_file.name).stem
    nuovo_nome = f"{video_stem}_{peso}.mp4"
    video_salvato_path = UPLOAD_DIR / nuovo_nome

    with open(video_salvato_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # cartella output include il modello così cache detection diverse non collidono
    model_slug = model_name.replace(".pt", "").replace("/", "_").replace("\\", "_")
    video_output_folder = OUTPUT_DIR / f"{video_salvato_path.stem}__{model_slug}"
    video_output_folder.mkdir(parents=True, exist_ok=True)
    detections_json_path = video_output_folder / "detections.json"

    if not detections_json_path.exists():
        with st.spinner(f"Detection in corso sul server con **{model_name}**…"):
            detections_data = server_detect(video_salvato_path, model_name, yolo_params)
        if detections_data is None:
            st.stop()
        with open(detections_json_path, "w") as f:
            json.dump(detections_data, f, indent=4)


    with open(detections_json_path, "r") as f:
        detections_data = json.load(f)

    col_video_or, col_video_dett = st.columns(2)
    with col_video_or:
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

    st.markdown("---")

    tab1, tab2 = st.tabs(["Tracking Test", "Comparazione Analytics"])

    # --- TAB 1: USO DEI TRACKER ---
    with tab1:
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
            visualizer = Visualizer(par_output_dir=str(video_output_folder))
            placeholder_tracker = st.empty()
            
            with placeholder_tracker.container():
                st.info(f"Invio richiesta di tracking **{tracker_type}** al server…")
                with st.spinner("Tracking in corso sul server…"):
                    risultati = server_track(
                        video_salvato_path,
                        tracker_type,
                        detections_data,
                        tracker_kwargs,
                    )

            placeholder_tracker.empty()

            if risultati is None:
                st.stop()

            # Salva CSV localmente per metriche e visualizzazione
            tracker_inst = TrackerClass(**tracker_kwargs)
            tracker_inst.output_dir = video_output_folder.parent
            tracker_inst.setup_video_paths(video_output_folder.name)
            tracker_inst.save_to_csv(risultati, params=tracker_kwargs)

            placeholder_annot_track = st.empty()
            
            with placeholder_annot_track.container():
                st.info("Generazione del video finale tracciato...")
                bar_annot = st.progress(0)
                text_annot = st.empty()

                def update_annot_track_ui(current_frame, total_frames):
                    pct = min(current_frame / total_frames, 1.0)
                    bar_annot.progress(pct)
                    text_annot.text(f"Disegno tracce ({tracker_type}): frame {current_frame} / {total_frames}")

                print("Annotazione video tracker")
                visualizer.draw_tracks(str(tracker_inst.csv_path), video_tracked_filename, str(video_salvato_path), progress_callback=update_annot_track_ui)

                converti_in_h264(video_tracked_path, video_tracked_converted)
            placeholder_annot_track.empty()

            if not video_raw_converted.exists():
                placeholder_annot_yolo = st.empty()
                
                with placeholder_annot_yolo.container():
                    st.info("Generazione del video con i rilevamenti YOLO grezzi...")
                    bar_yolo = st.progress(0)
                    text_yolo = st.empty()

                    def update_annot_yolo_ui(current_frame, total_frames):
                        pct = min(current_frame / total_frames, 1.0)
                        bar_yolo.progress(pct)
                        text_yolo.text(f"Disegno Bounding Box YOLO: frame {current_frame} / {total_frames}")

                    print("Annotazione video YOLO")
                    visualizer.draw_raw_detections(detections_data, "raw_detections.mp4", video_salvato_path, progress_callback=update_annot_yolo_ui)
                    
                    converti_in_h264(video_raw_path, video_raw_converted)
                placeholder_annot_yolo.empty()
            else:
                print("Video YOLO grezzo già presente.")

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

    with tab2:
        st.subheader("Analisi Performance dei Tracker Elaborati")
        
        metriche_globali = {}
        for nome_tracker in TRACKER_REGISTRY.keys():
            csv_files = sorted(video_output_folder.glob(f"{nome_tracker.lower()}_*.csv"))
            if csv_files:
                metriche = calcola_metriche_csv(csv_files[-1])  # prende l'ultimo
                if metriche:
                    metriche_globali[nome_tracker.upper()] = metriche

        if metriche_globali:
            df_comparativo = pd.DataFrame.from_dict(metriche_globali, orient="index")
            
            st.markdown("##### Tabella Riassuntiva")
            st.dataframe(df_comparativo)
            
            st.markdown("---")
            st.markdown("##### Grafici Comparativi delle Metriche")
            
            mappa_nomi_metriche = {
                "id_switches": "ID Switches",
                "fragmentation": "Fragmentation",
                "kinematic_jumps": "Kinematic Jumps",
                "track_coverage": "Track Coverage %",
                "time": "Tempo di Elaborazione in secondi",
                "avg_track_length": "Lunghezza Media Tracce in frame",
                "max_track_length": "Lunghezza Massima Tracce in frame",
                "num_tracks": "Numero di Tracce Uniche Rilevate",
                "avg_id_lifetime": "Durata Vita Media ID in frame",
                "max_id_lifetime": "Durata Vita Massima ID in frame",
                "total_detections": "Rilevamenti Totali Tracciati",
                "spurious_tracks_ratio": "Rapporto Tracce Spurie"
            }
            
            lista_colori = [
                "#06b6d4",  
                "#ea580c",  
                "#7c3aed",  
                "#10b981",  
                "#e11d48",  
                "#2563eb",  
                "#d946ef",  
                "#84cc16",  
                "#f59e0b",  
                "#4f46e5",  
                "#f43f5e",  
                "#34A853"   
            ]
            
            colonne_disponibili = list(df_comparativo.columns)
            
            for i in range(0, len(colonne_disponibili), 2):
                c1, c2 = st.columns(2)
                
                col_1 = colonne_disponibili[i]
                label_1 = mappa_nomi_metriche.get(col_1, col_1.replace("_", " ").title())
                colore_1 = lista_colori[i % len(lista_colori)]
                with c1:
                    st.markdown(f"##### {label_1}")
                    st.bar_chart(df_comparativo[col_1], color=colore_1, height=450)
                
                if i + 1 < len(colonne_disponibili):
                    col_2 = colonne_disponibili[i+1]
                    label_2 = mappa_nomi_metriche.get(col_2, col_2.replace("_", " ").title())
                    colore_2 = lista_colori[(i + 1) % len(lista_colori)]
                    with c2:
                        st.markdown(f"##### {label_2}")
                        st.bar_chart(df_comparativo[col_2], color=colore_2, height=450)
                
        else:
            st.info("Nessuna metrica disponibile. Torna nel tab 'Uso dei Tracker' ed avvia l'elaborazione di almeno un algoritmo per popolare i grafici.")
