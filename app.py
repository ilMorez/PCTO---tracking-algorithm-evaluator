import io
import base64
import zipfile
import json
import csv as _csv
from pathlib import Path
from datetime import datetime

import cv2
import pandas as pd
import requests
import streamlit as st

from custom_trackers import TRACKER_REGISTRY
from evaluator import TrackerEvaluator
from visualization import CLASS_COLOR_PALETTE

st.set_page_config(page_title="Dashboard Comparazione Video", layout="wide")
st.title("Comparazione BBox Grezze vs Video Tracciato")
st.markdown("---")

OUTPUT_DIR = Path("output")
UPLOAD_DIR = Path("uploaded_videos")
OUTPUT_DIR.mkdir(exist_ok=True)
UPLOAD_DIR.mkdir(exist_ok=True)

if "detections_data" not in st.session_state:
    st.session_state["detections_data"] = None
if "detection_id" not in st.session_state:
    st.session_state["detection_id"] = None
if "detections_video_key" not in st.session_state:
    st.session_state["detections_video_key"] = None

# Classi YOLO comuni — l'utente può aggiungerne altre via text_input
DEFAULT_YOLO_CLASSES = [
    "person", "car", "truck", "bus", "motorcycle", "bicycle",
    "dog", "cat", "bird", "boat", "aeroplane", "train",
]


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("Configurazione Server")
    SERVER_URL = st.text_input(
        "URL del server", value="http://localhost:8000",
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
    yolo_device  = st.selectbox(
        "Device", ["auto", "cpu", "cuda:0"], index=0,
        help="'auto' lascia decidere a YOLO (GPU se disponibile sul server). "
             "Forza 'cuda:0' se il server ha una GPU, 'cpu' altrimenti.",
    )

    yolo_params = {
        "conf": yolo_conf, "iou": yolo_iou, "imgsz": yolo_imgsz,
        "half": yolo_half, "stream": True, "verbose": yolo_verbose,
    }
    if yolo_device != "auto":
        yolo_params["device"] = yolo_device

    st.markdown("---")
    st.header("Classi da rilevare")
    selected_classes = st.multiselect(
        "Seleziona le classi target:",
        options=DEFAULT_YOLO_CLASSES,
        default=["car"],
        help="YOLO rileverà solo queste classi",
    )
    extra_classes = st.text_input(
        "Aggiungi classi custom (virgola-separate):", value=""
    ).strip()
    if extra_classes:
        for c in extra_classes.split(","):
            c = c.strip()
            if c and c not in selected_classes:
                selected_classes.append(c)

    if not selected_classes:
        st.warning("Seleziona almeno una classe.")
        selected_classes = ["car"]

    st.markdown("---")
    st.header("Ground Truth (opzionale)")
    gt_file = st.file_uploader(
        "Carica file ground truth (.txt, formato MOT):",
        type=["txt"],
        help="Formato atteso: frame, track_id, x, y, w, h, conf, class, visibility",
    )

# ---------------------------------------------------------------------------
# Chiamate server
# ---------------------------------------------------------------------------

def server_detect(video_path: Path, model_name: str, yolo_params: dict, target_classes: list):
    """Esegue la detection sul server. Ritorna (detections, detection_id, yolo_video_bytes).

    Il video grezzo NON viene incluso nella risposta JSON principale (niente base64):
    se il server fornisce un detection_id, il video va scaricato separatamente con
    server_fetch_detect_video().
    """
    endpoint = f"{SERVER_URL}/api/detect"
    try:
        with open(video_path, "rb") as vf:
            resp = requests.post(
                endpoint,
                data={
                    "model_name":          model_name,
                    "yolo_params_json":    json.dumps(yolo_params),
                    "target_classes_json": json.dumps(target_classes),
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
            st.error(f"Errore `/api/detect` [{resp.status_code}]:\n```\n{detail[:1000]}\n```")
            return None, None, None
        data         = resp.json()
        detections   = data.get("detections")
        detection_id = data.get("detection_id")

        video_bytes = None
        if data.get("has_raw_video") and detection_id:
            video_bytes = server_fetch_detect_video(detection_id)
        elif data.get("yolo_video"):
            # Fallback legacy: server vecchio che manda ancora il video in base64
            video_bytes = base64.b64decode(data["yolo_video"])

        return detections, detection_id, video_bytes
    except requests.exceptions.ConnectionError:
        st.error(f"Impossibile connettersi al server: `{endpoint}`")
        return None, None, None
    except Exception as e:
        st.error(f"Errore `/api/detect`:\n```\n{e}\n```")
        return None, None, None


def server_fetch_detect_video(detection_id: str):
    """Scarica in streaming il video con i rilevamenti grezzi, dato un detection_id."""
    endpoint = f"{SERVER_URL}/api/detect/{detection_id}/video"
    try:
        resp = requests.get(
            endpoint, timeout=600,
            proxies={"http": None, "https": None},
        )
        if resp.status_code != 200:
            # Non bloccante: il video grezzo è opzionale
            return None
        return resp.content
    except Exception:
        return None


def server_track(video_path: Path, tracker_name: str, detections_data: list,
                 tracker_params: dict, target_class: str = "", extract_crops: bool = False,
                 output_dir: Path = None, detection_id: str = None):
    endpoint = f"{SERVER_URL}/api/track"
    try:
        data = {
            "tracker_name":        tracker_name,
            "tracker_params_json": json.dumps(tracker_params),
            "target_class":        target_class,
            "extract_crops":       extract_crops,
        }
        # Preferisce detection_id (dati già in cache sul server): evita di
        # ritrasmettere l'intera struttura detections a ogni richiesta.
        if detection_id:
            data["detection_id"] = detection_id
        else:
            data["detections_json"] = json.dumps(detections_data)

        with open(video_path, "rb") as vf:
            resp = requests.post(
                endpoint,
                data=data,
                files={"video": (video_path.name, vf, "video/mp4")},
                timeout=600,
                proxies={"http": None, "https": None},
            )
        if resp.status_code != 200:
            try:
                detail = resp.json().get("detail", resp.text)
            except Exception:
                detail = resp.text
            # Se il detection_id non è più valido sul server (cache scaduta),
            # ritenta una sola volta inviando i dati completi.
            if resp.status_code == 404 and detection_id and detections_data is not None:
                return server_track(
                    video_path, tracker_name, detections_data, tracker_params,
                    target_class=target_class, extract_crops=extract_crops,
                    output_dir=output_dir, detection_id=None,
                )
            st.error(f"Errore `/api/track` [{resp.status_code}]:\n```\n{detail[:1000]}\n```")
            return None, None, None
        zf          = zipfile.ZipFile(io.BytesIO(resp.content))
        names       = zf.namelist()
        mp4_name    = next((n for n in names if n.endswith(".mp4")), None)
        csv_name    = next((n for n in names if n.endswith(".csv")), None)
        video_bytes = zf.read(mp4_name) if mp4_name else None
        csv_bytes   = zf.read(csv_name) if csv_name else None
        
        if extract_crops and output_dir:
            for member in zf.infolist():
                if member.filename.startswith("crops/") and len(member.filename) > 6:
                    if member.filename.endswith('/'):
                        continue
                    # Rimuove il prefisso 'crops/' così estrae direttamente la cartella classe accanto al CSV
                    rel_path = Path(member.filename).relative_to("crops")
                    target_path = output_dir / rel_path
                    target_path.parent.mkdir(parents=True, exist_ok=True)
                    target_path.write_bytes(zf.read(member))
        
        results = []
        if csv_bytes:
            reader = _csv.DictReader(
                line for line in csv_bytes.decode().splitlines() if not line.startswith("#")
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


def server_track_multi(video_path: Path, assignments: list, detections_data: list,
                        extract_crops: bool = False, output_dir: Path = None,
                        detection_id: str = None):
    endpoint = f"{SERVER_URL}/api/track_multi"
    try:
        data = {
            "assignments_json": json.dumps(assignments),
            "extract_crops":    extract_crops,
        }
        if detection_id:
            data["detection_id"] = detection_id
        else:
            data["detections_json"] = json.dumps(detections_data)

        with open(video_path, "rb") as vf:
            resp = requests.post(
                endpoint,
                data=data,
                files={"video": (video_path.name, vf, "video/mp4")},
                timeout=600,
                proxies={"http": None, "https": None},
            )
        if resp.status_code != 200:
            try:
                detail = resp.json().get("detail", resp.text)
            except Exception:
                detail = resp.text
            if resp.status_code == 404 and detection_id and detections_data is not None:
                return server_track_multi(
                    video_path, assignments, detections_data,
                    extract_crops=extract_crops, output_dir=output_dir, detection_id=None,
                )
            st.error(f"Errore `/api/track_multi` [{resp.status_code}]:\n```\n{detail[:1000]}\n```")
            return None, None
        zf        = zipfile.ZipFile(io.BytesIO(resp.content))
        names     = zf.namelist()
        mp4_name  = next((n for n in names if n.endswith(".mp4")), None)
        csv_names = [n for n in names if n.endswith(".csv")]
        video_bytes = zf.read(mp4_name) if mp4_name else None
        
        if extract_crops and output_dir:
            for member in zf.infolist():
                if member.filename.startswith("crops/") and len(member.filename) > 6:
                    if member.filename.endswith('/'):
                        continue
                    rel_path = Path(member.filename).relative_to("crops")
                    target_path = output_dir / rel_path
                    target_path.parent.mkdir(parents=True, exist_ok=True)
                    target_path.write_bytes(zf.read(member))
        
        csv_map = {n.replace("_tracking.csv", ""): zf.read(n) for n in csv_names}
        return video_bytes, csv_map
    except requests.exceptions.ConnectionError:
        st.error(f"Impossibile connettersi al server: `{endpoint}`")
        return None, None
    except Exception as e:
        st.error(f"Errore `/api/track_multi`:\n```\n{e}\n```")
        return None, None


# ---------------------------------------------------------------------------
# Helpers (invariati)
# ---------------------------------------------------------------------------

def calcola_metriche_csv(csv_path: Path):
    try:
        df_temp      = pd.read_csv(csv_path, comment="#")
        total_frames = int(df_temp["frame"].max() + 1) if not df_temp.empty else 1
        evaluator    = TrackerEvaluator(par_total_frames=total_frames)
        return evaluator.evaluate(str(csv_path))
    except Exception:
        return None


def parse_gt_file(gt_file_obj) -> pd.DataFrame | None:
    try:
        content = gt_file_obj.read().decode("utf-8")
        rows = []
        for line in content.splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split(",")
            if len(parts) < 6:
                continue
            frame_id = int(float(parts[0]))
            track_id = int(float(parts[1]))
            if track_id == -1:
                continue
            x, y, w, h = float(parts[2]), float(parts[3]), float(parts[4]), float(parts[5])
            rows.append({"frame": frame_id, "track_id": track_id, "x": x, "y": y, "w": w, "h": h})
        return pd.DataFrame(rows) if rows else None
    except Exception:
        return None


def calcola_metriche_gt(gt_df: pd.DataFrame) -> dict:
    total_frames  = int(gt_df["frame"].max()) + 1
    unique_tracks = gt_df["track_id"].nunique()
    track_lengths = gt_df.groupby("track_id")["frame"].count()
    avg_len       = float(track_lengths.mean())
    max_len       = int(track_lengths.max())
    total_det     = len(gt_df)
    avg_lifetime  = float(gt_df.groupby("track_id").apply(lambda g: g["frame"].max() - g["frame"].min() + 1).mean())
    max_lifetime  = int(gt_df.groupby("track_id").apply(lambda g: g["frame"].max() - g["frame"].min() + 1).max())
    track_coverage = (total_det / (unique_tracks * total_frames) * 100) if unique_tracks > 0 else 0.0
    return {
        "num_tracks": unique_tracks, "total_detections": total_det,
        "avg_track_length": round(avg_len, 2), "max_track_length": max_len,
        "avg_id_lifetime": round(avg_lifetime, 2), "max_id_lifetime": max_lifetime,
        "track_coverage": round(track_coverage, 4),
        "id_switches": float("nan"), "fragmentation": float("nan"),
        "kinematic_jumps": float("nan"), "spurious_tracks_ratio": float("nan"), "time": float("nan"),
    }


def render_tracker_parameters(tracker_cls, tracker_key: str):
    specs          = getattr(tracker_cls, "PARAMETER_SPECS", [])
    tracker_kwargs = {}
    if not specs:
        st.info("Questo tracker non espone parametri modificabili.")
        return tracker_kwargs
    with st.expander("Parametri del tracker"):
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
    peso               = uploaded_file.size
    video_stem         = Path(uploaded_file.name).stem
    nuovo_nome         = f"{video_stem}_{peso}.mp4"
    video_salvato_path = UPLOAD_DIR / nuovo_nome

    with open(video_salvato_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    model_slug          = model_name.replace(".pt", "").replace("/", "_").replace("\\", "_")
    classes_slug        = "_".join(sorted(selected_classes))
    video_output_folder = OUTPUT_DIR / f"{video_salvato_path.stem}__{model_slug}__{classes_slug}"
    video_output_folder.mkdir(parents=True, exist_ok=True)

    detections_json_path = video_output_folder / "detections.json"
    yolo_video_path      = video_output_folder / "raw_detections_h264.mp4"

    # --- Detection ---
    detection_key = f"{video_salvato_path.stem}__{model_slug}__{classes_slug}"

    # Avvisa se i parametri sono cambiati rispetto all'ultima detection eseguita
    warning_placeholder = st.empty()
    if (st.session_state["detections_data"] is not None
            and st.session_state["detections_video_key"] != detection_key):
        warning_placeholder.warning(
            "Video, modello o classi sono cambiati rispetto all'ultima detection. "
            "Premi **Avvia Detection** per aggiornare."
        )
 
    if st.button("Avvia Detection", key="btn_det"):
        detection_id = None
        if not detections_json_path.exists():
            with st.spinner(f"Detection in corso con **{model_name}** per classi: {selected_classes}…"):
                det_result, detection_id, yolo_video_bytes = server_detect(
                    video_salvato_path, model_name, yolo_params, selected_classes
                )
            if det_result is None:
                st.stop()
            with open(detections_json_path, "w") as f:
                json.dump(det_result, f, indent=4)
            if yolo_video_bytes and not yolo_video_path.exists():
                yolo_video_path.write_bytes(yolo_video_bytes)
        else:
            with open(detections_json_path, "r") as f:
                det_result = json.load(f)
 
        st.session_state["detections_data"]      = det_result
        st.session_state["detection_id"]         = detection_id
        st.session_state["detections_video_key"] = detection_key
        warning_placeholder.empty()

    # Legge da session_state: persiste tra re-run senza rieseguire la detection
    detections_data = st.session_state["detections_data"]
    detection_id    = st.session_state["detection_id"]

    # Info video + conteggio classi (visibile solo se detection disponibile)
    if detections_data is not None:
        col_video_or, col_video_dett = st.columns(2)
        with col_video_or:
            st.video(str(video_salvato_path))
        with col_video_dett:
            cap    = cv2.VideoCapture(str(video_salvato_path))
            fps    = cap.get(cv2.CAP_PROP_FPS)
            frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            st.metric("Durata video",  f"{(frames / fps):.2f} s")
            st.metric("Frame Rate",    f"{fps:.2f} FPS")
            st.metric("Frame Totali",  f"{frames} frames")
            st.metric("Dimensioni",    f"{int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))} px")
            cap.release()
            class_counts = {}
            for frame in detections_data:
                for det in frame["detections"]:
                    cls = det[5] if len(det) >= 6 else "?"
                    class_counts[cls] = class_counts.get(cls, 0) + 1
            if class_counts:
                st.markdown("**Detections per classe:**")
                for cls, cnt in class_counts.items():
                    st.markdown(f"- **{cls}**: {cnt}")

    st.markdown("---")

    if detections_data is None:
        st.info("Premi **Avvia Detection** per caricare le detection prima di usare i tracker.")
        st.stop()

    tab1, tab2, tab3, tab4 = st.tabs([
        "Tracking Singolo",
        "Multi-Classe (un tracker per classe)",
        "Tracker VS Tracker",
        "Comparazione Analytics",
    ])

    # -----------------------------------------------------------------------
    # TAB 1 — Tracking singolo (una sola classe)
    # -----------------------------------------------------------------------
    with tab1:
        col_cls, col_trk = st.columns(2)
        with col_cls:
            single_class = st.selectbox(
                "Classe da tracciare:", selected_classes, key="tab1_class"
            )
        with col_trk:
            tracker_type = st.selectbox(
                "Algoritmo:", list(TRACKER_REGISTRY.keys()), key="tab1_tracker"
            )

        TrackerClass   = TRACKER_REGISTRY[tracker_type]
        tracker_kwargs = render_tracker_parameters(TrackerClass, tracker_key=f"tab1_{tracker_type}")

        tracker_dir = video_output_folder / tracker_type
        tracker_dir.mkdir(parents=True, exist_ok=True)
        video_tracked_converted = tracker_dir / f"{tracker_type.lower()}_{single_class}_h264.mp4"
        
        extract_crops = st.checkbox(
            "Estrai crop delle tracce (BBox)", 
            value=False, 
            key="tab1_extract_crops",
            help="Salva le immagini croppate dell'oggetto lungo il suo percorso escludendo i bordi."
        )

        if st.button("Avvia Elaborazione", key="btn_tab1"):
            with st.spinner(f"Tracking **{tracker_type}** su **{single_class}**…"):
                risultati, tracked_bytes, csv_bytes = server_track(
                    video_salvato_path, tracker_type, detections_data,
                    tracker_kwargs, target_class=single_class,
                    extract_crops=extract_crops, output_dir=video_output_folder,
                    detection_id=detection_id,
                )
            if risultati is None:
                st.stop()
            if tracked_bytes:
                video_tracked_converted.write_bytes(tracked_bytes)
            if csv_bytes:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                csv_path  = tracker_dir / f"{tracker_type.lower()}_{single_class}_{timestamp}.csv"
                csv_path.write_bytes(csv_bytes)

        v_col1, v_col2 = st.columns(2)
        with v_col1:
            st.markdown("**Rilevamenti Grezzi (YOLO):**")
            if yolo_video_path.exists():
                st.video(str(yolo_video_path))
            else:
                st.caption("Avvia l'elaborazione per generare l'output visivo.")
        with v_col2:
            st.markdown(f"**Video Tracciato ({tracker_type} — {single_class}):**")
            if video_tracked_converted.exists():
                st.video(str(video_tracked_converted))
            else:
                st.caption("Avvia l'elaborazione per generare l'output visivo.")

    # -----------------------------------------------------------------------
    # TAB 2 — Multi-classe: un tracker per ogni classe → video unico
    # -----------------------------------------------------------------------
    with tab2:
        st.subheader("Assegna un tracker a ogni classe rilevata")
        st.markdown(
            "Ogni classe viene tracciata con il proprio algoritmo. "
            "Il risultato è un unico video con colori diversi per classe."
        )

        assignments_ui = []
        for idx, cls in enumerate(selected_classes):
            color_bgr = CLASS_COLOR_PALETTE[idx % len(CLASS_COLOR_PALETTE)]
            col_trk, col_params = st.columns(2)
            with col_trk:
                trk = st.selectbox(
                    f"Tracker per **{cls}**:",
                    list(TRACKER_REGISTRY.keys()),
                    key=f"multi_tracker_{cls}",
                )
            with col_params:
                TrackerCls = TRACKER_REGISTRY[trk]
                kwargs     = render_tracker_parameters(TrackerCls, tracker_key=f"multi_{cls}_{trk}")
            assignments_ui.append({
                "tracker_name":   trk,
                "target_class":   cls,
                "tracker_params": kwargs,
            })

        multi_video_path = video_output_folder / "multi_class_tracked_h264.mp4"
        
        extract_crops_multi = st.checkbox(
            "Estrai crop delle tracce (BBox) per tutte le classi", 
            value=False, 
            key="tab2_extract_crops",
            help="Salva le immagini croppate degli oggetti di tutte le classi lungo il loro percorso escludendo i bordi."
        )

        if st.button("Avvia Multi-Tracking", key="btn_tab2"):
            with st.spinner("Multi-tracking in corso sul server…"):
                video_bytes, csv_map = server_track_multi(
                    video_salvato_path, assignments_ui, detections_data,
                    extract_crops=extract_crops_multi, output_dir=video_output_folder,
                    detection_id=detection_id,
                )
            if video_bytes is None:
                st.stop()
            multi_video_path.write_bytes(video_bytes)
            if csv_map:
                for name, data in csv_map.items():
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    parts     = name.split("_", 1)
                    cls_part  = parts[0] if len(parts) > 1 else name
                    trk_part  = parts[1] if len(parts) > 1 else "unknown"
                    original_trk_name = next((t for t in TRACKER_REGISTRY.keys() if t.lower() == trk_part), trk_part)
                    trk_dir = video_output_folder / original_trk_name
                    trk_dir.mkdir(parents=True, exist_ok=True)
                    
                    p = trk_dir / f"{trk_part}_{cls_part}_{timestamp}.csv"
                    p.write_bytes(data)

        if multi_video_path.exists():
            st.markdown("**Video Multi-Classe:**")
            st.video(str(multi_video_path))
        else:
            st.caption("Avvia il Multi-Tracking per generare il video.")

    # -----------------------------------------------------------------------
    # TAB 3 — Tracker VS Tracker (su una stessa classe)
    # -----------------------------------------------------------------------
    with tab3:
        vs_class = st.selectbox("Classe per il confronto:", selected_classes, key="tab3_class")

        col1, col2 = st.columns(2)
        for col, suffix in [(col1, "col1"), (col2, "col2")]:
            with col:
                trk_type = st.selectbox(
                    "Algoritmo:", list(TRACKER_REGISTRY.keys()), key=f"tracker_type_{suffix}"
                )
                TrackerCls_  = TRACKER_REGISTRY[trk_type]
                trk_kwargs_  = render_tracker_parameters(TrackerCls_, tracker_key=f"{trk_type.lower()}_{suffix}")
                trk_dir = video_output_folder / trk_type
                trk_dir.mkdir(parents=True, exist_ok=True)
                video_out_ = trk_dir / f"{trk_type.lower()}_{vs_class}_{suffix}_h264.mp4"

                if st.button("Avvia Elaborazione", key=f"btn_{suffix}"):
                    with st.spinner(f"Tracking **{trk_type}** su **{vs_class}**…"):
                        risultati, tracked_bytes, csv_bytes = server_track(
                            video_salvato_path, trk_type, detections_data,
                            trk_kwargs_, target_class=vs_class,
                            detection_id=detection_id,
                        )
                    if risultati is None:
                        st.stop()
                    if tracked_bytes:
                        video_out_.write_bytes(tracked_bytes)
                    if csv_bytes:
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        p = trk_dir / f"{trk_type.lower()}_{vs_class}_{timestamp}.csv"
                        p.write_bytes(csv_bytes)

                st.markdown(f"**Video Tracciato ({trk_type} — {vs_class}):**")
                if video_out_.exists():
                    st.video(str(video_out_))
                else:
                    st.caption("Avvia l'elaborazione per generare l'output visivo.")

    # -----------------------------------------------------------------------
    # TAB 4 — Analytics
    # -----------------------------------------------------------------------
    with tab4:
        st.subheader("Analisi Performance dei Tracker Elaborati")
 
        metriche_globali = {}
 
        all_csvs = sorted(video_output_folder.rglob("*.csv"))
        tracker_names_lower = {t.lower(): t for t in TRACKER_REGISTRY.keys()}
        best_csv = {}
 
        for csv_path in all_csvs:
            stem      = csv_path.stem
            matched_trk = None
            matched_cls = None
            for trk_lower, trk_display in tracker_names_lower.items():
                trk_flat  = trk_lower.replace("-", "").replace(" ", "")
                stem_flat = stem.replace("-", "").replace(" ", "")
                if stem_flat.startswith(trk_flat):
                    remainder = stem[len(trk_lower):].lstrip("_")
                    for cls in selected_classes:
                        if remainder.startswith(cls + "_") or remainder == cls:
                            matched_trk = trk_display
                            matched_cls = cls
                            break
                if matched_trk:
                    break
            if matched_trk and matched_cls:
                key = (matched_trk, matched_cls)
                if key not in best_csv or csv_path.name > best_csv[key].name:
                    best_csv[key] = csv_path
 
        for (trk_display, cls), csv_path in best_csv.items():
            metriche = calcola_metriche_csv(csv_path)
            if metriche:
                metriche_globali[f"{trk_display.upper()} [{cls}]"] = metriche
 
        if gt_file is not None:
            gt_file.seek(0)
            gt_df = parse_gt_file(gt_file)
            if gt_df is not None:
                metriche_globali["GROUND TRUTH"] = calcola_metriche_gt(gt_df)
            else:
                st.warning("Impossibile parsificare il file ground truth.")
 
        if metriche_globali:
            df_comparativo = pd.DataFrame.from_dict(metriche_globali, orient="index")
            st.markdown("##### Tabella Riassuntiva")
            st.dataframe(df_comparativo)
 
            st.markdown("---")
            st.markdown("##### Grafici Comparativi")
 
            mappa_nomi_metriche = {
                "id_switches": "ID Switches", "fragmentation": "Fragmentation",
                "kinematic_jumps": "Kinematic Jumps", "track_coverage": "Track Coverage %",
                "time": "Tempo di Elaborazione (s)",
                "avg_track_length": "Lunghezza Media Tracce (frame)",
                "max_track_length": "Lunghezza Massima Tracce (frame)",
                "num_tracks": "Numero di Tracce Uniche",
                "avg_id_lifetime": "Durata Vita Media ID (frame)",
                "max_id_lifetime": "Durata Vita Massima ID (frame)",
                "total_detections": "Rilevamenti Totali Tracciati",
                "spurious_tracks_ratio": "Rapporto Tracce Spurie",
            }
 
            # --- Filtri classe e tracker ---
            classi_disponibili  = sorted({k.split("[")[1].rstrip("]") for k in metriche_globali if "[" in k})
            tracker_disponibili = sorted({k.split(" [")[0] for k in metriche_globali if "[" in k})
            if "GROUND TRUTH" in metriche_globali:
                tracker_disponibili.append("GROUND TRUTH")
 
            f_col1, f_col2 = st.columns(2)
            with f_col1:
                classi_scelte = st.multiselect(
                    "Filtra per classe:", options=classi_disponibili,
                    default=classi_disponibili, key="tab4_filter_class",
                )
            with f_col2:
                tracker_scelti = st.multiselect(
                    "Filtra per tracker:", options=tracker_disponibili,
                    default=tracker_disponibili, key="tab4_filter_tracker",
                )
 
            righe_filtrate = []
            for k in metriche_globali:
                if k == "GROUND TRUTH":
                    if "GROUND TRUTH" in tracker_scelti:
                        righe_filtrate.append(k)
                else:
                    trk = k.split(" [")[0]
                    cls = k.split("[")[1].rstrip("]")
                    if trk in tracker_scelti and cls in classi_scelte:
                        righe_filtrate.append(k)
 
            if not righe_filtrate:
                st.info("Nessun risultato con i filtri selezionati.")
            else:
                df_filtrato = df_comparativo.loc[righe_filtrate]
                lista_colori = [
                    "#06b6d4","#ea580c","#7c3aed","#10b981","#e11d48",
                    "#2563eb","#d946ef","#84cc16","#f59e0b","#4f46e5",
                ]
                colonne_disponibili = list(df_filtrato.columns)
                for i in range(0, len(colonne_disponibili), 2):
                    c1, c2  = st.columns(2)
                    col_1   = colonne_disponibili[i]
                    label_1 = mappa_nomi_metriche.get(col_1, col_1.replace("_", " ").title())
                    with c1:
                        st.markdown(f"##### {label_1}")
                        st.bar_chart(df_filtrato[col_1], color=lista_colori[i % len(lista_colori)], height=450)
                    if i + 1 < len(colonne_disponibili):
                        col_2   = colonne_disponibili[i + 1]
                        label_2 = mappa_nomi_metriche.get(col_2, col_2.replace("_", " ").title())
                        with c2:
                            st.markdown(f"##### {label_2}")
                            st.bar_chart(df_filtrato[col_2], color=lista_colori[(i+1) % len(lista_colori)], height=450)
        else:
            st.info("Nessuna metrica disponibile. Avvia l'elaborazione di almeno un tracker.")
