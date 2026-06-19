"""
app.py — Punto di ingresso e orchestratore UI.

Contiene esclusivamente la definizione del layout Streamlit e il flusso
principale. Tutta la logica è delegata ai moduli sotto utils/:

    utils/network.py     → chiamate HTTP al server FastAPI
    utils/processing.py  → calcolo metriche, parsing ground truth
    utils/ui_helpers.py  → componenti Streamlit riutilizzabili
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st
import torch

from custom_trackers import TRACKER_REGISTRY
from visualization import CLASS_COLOR_PALETTE

from utils_client.network import (
    server_detect,
    server_track,
    server_track_multi,
    server_analyze_crops,
)
from utils_client.processing import (
    calcola_metriche_csv,
    parse_gt_file,
    calcola_metriche_gt,
    collect_best_csvs,
)
from utils_client.ui_helpers import render_tracker_parameters, render_video_metrics

# ---------------------------------------------------------------------------
# Configurazione pagina
# ---------------------------------------------------------------------------
st.set_page_config(page_title="Dashboard Comparazione Video", layout="wide")
st.title("Comparazione BBox Grezze vs Video Tracciato")
st.markdown("---")

OUTPUT_DIR = Path("output")
UPLOAD_DIR = Path("uploaded_videos")
OUTPUT_DIR.mkdir(exist_ok=True)
UPLOAD_DIR.mkdir(exist_ok=True)

# Session state
for key in ("detections_data", "detection_id", "detections_video_key"):
    if key not in st.session_state:
        st.session_state[key] = None

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
        "yoloe-26n-seg.pt", "yoloe-26s-seg.pt", "yoloe-26m-seg.pt",
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
        help="'auto' lascia decidere a YOLO. Forza 'cuda:0' se il server ha una GPU.",
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
# Upload video
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

    warning_placeholder = st.empty()
    if (
        st.session_state["detections_data"] is not None
        and st.session_state["detections_video_key"] != detection_key
    ):
        warning_placeholder.warning(
            "Video, modello o classi sono cambiati rispetto all'ultima detection. "
            "Premi **Avvia Detection** per aggiornare."
        )

    if st.button("Avvia Detection", key="btn_det"):
        detection_id = None
        if not detections_json_path.exists():
            with st.spinner(f"Detection in corso con **{model_name}** per classi: {selected_classes}…"):
                det_result, detection_id, yolo_video_bytes = server_detect(
                    SERVER_URL, video_salvato_path, model_name, yolo_params, selected_classes
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

    detections_data = st.session_state["detections_data"]
    detection_id    = st.session_state["detection_id"]

    # Anteprima video + metriche (solo se detection disponibile)
    if detections_data is not None:
        col_video_or, col_video_dett = st.columns(2)
        with col_video_or:
            st.video(str(video_salvato_path))
        with col_video_dett:
            cap = cv2.VideoCapture(str(video_salvato_path))
            render_video_metrics(cap, detections_data)

    st.markdown("---")

    if detections_data is None:
        st.info("Premi **Avvia Detection** per caricare le detection prima di usare i tracker.")
        st.stop()

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "Tracking Singolo",
        "Multi-Classe (un tracker per classe)",
        "Tracker VS Tracker",
        "Comparazione Analytics",
        "Ricerca Semantica",
        "Analisi Attributi (YOLO-E)",
    ])

    # -----------------------------------------------------------------------
    # TAB 1 — Tracking singolo
    # -----------------------------------------------------------------------
    with tab1:
        col_cls, col_trk = st.columns(2)
        with col_cls:
            single_class = st.selectbox("Classe da tracciare:", selected_classes, key="tab1_class")
        with col_trk:
            tracker_type = st.selectbox("Algoritmo:", list(TRACKER_REGISTRY.keys()), key="tab1_tracker")

        TrackerClass   = TRACKER_REGISTRY[tracker_type]
        tracker_kwargs = render_tracker_parameters(TrackerClass, tracker_key=f"tab1_{tracker_type}")

        tracker_dir = video_output_folder / tracker_type
        tracker_dir.mkdir(parents=True, exist_ok=True)
        video_tracked_converted = tracker_dir / f"{tracker_type.lower()}_{single_class}_h264.mp4"

        extract_crops = st.checkbox(
            "Estrai crop delle tracce (BBox)", value=False, key="tab1_extract_crops",
            help="Salva le immagini croppate dell'oggetto lungo il suo percorso.",
        )

        if st.button("Avvia Elaborazione", key="btn_tab1"):
            with st.spinner(f"Tracking **{tracker_type}** su **{single_class}**…"):
                risultati, tracked_bytes, csv_bytes = server_track(
                    SERVER_URL, video_salvato_path, tracker_type, detections_data,
                    tracker_kwargs, target_class=single_class,
                    extract_crops=extract_crops, output_dir=video_output_folder,
                    embeddings_dir=tracker_dir, detection_id=detection_id,
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
    # TAB 2 — Multi-classe
    # -----------------------------------------------------------------------
    with tab2:
        st.subheader("Assegna un tracker a ogni classe rilevata")
        st.markdown(
            "Ogni classe viene tracciata con il proprio algoritmo. "
            "Il risultato è un unico video con colori diversi per classe."
        )

        assignments_ui = []
        for idx, cls in enumerate(selected_classes):
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
            "Estrai crop delle tracce (BBox) per tutte le classi", value=False,
            key="tab2_extract_crops",
        )

        if st.button("Avvia Multi-Tracking", key="btn_tab2"):
            with st.spinner("Multi-tracking in corso sul server…"):
                video_bytes, csv_map = server_track_multi(
                    SERVER_URL, video_salvato_path, assignments_ui, detections_data,
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
                    original_trk_name = next(
                        (t for t in TRACKER_REGISTRY.keys() if t.lower() == trk_part), trk_part
                    )
                    trk_dir = video_output_folder / original_trk_name
                    trk_dir.mkdir(parents=True, exist_ok=True)
                    (trk_dir / f"{trk_part}_{cls_part}_{timestamp}.csv").write_bytes(data)

        if multi_video_path.exists():
            st.markdown("**Video Multi-Classe:**")
            st.video(str(multi_video_path))
        else:
            st.caption("Avvia il Multi-Tracking per generare il video.")

    # -----------------------------------------------------------------------
    # TAB 3 — Tracker VS Tracker
    # -----------------------------------------------------------------------
    with tab3:
        vs_class = st.selectbox("Classe per il confronto:", selected_classes, key="tab3_class")

        col1, col2 = st.columns(2)
        for col, suffix in [(col1, "col1"), (col2, "col2")]:
            with col:
                trk_type   = st.selectbox(
                    "Algoritmo:", list(TRACKER_REGISTRY.keys()), key=f"tracker_type_{suffix}"
                )
                TrackerCls_ = TRACKER_REGISTRY[trk_type]
                trk_kwargs_ = render_tracker_parameters(
                    TrackerCls_, tracker_key=f"{trk_type.lower()}_{suffix}"
                )
                trk_dir  = video_output_folder / trk_type
                trk_dir.mkdir(parents=True, exist_ok=True)
                video_out_ = trk_dir / f"{trk_type.lower()}_{vs_class}_{suffix}_h264.mp4"

                if st.button("Avvia Elaborazione", key=f"btn_{suffix}"):
                    with st.spinner(f"Tracking **{trk_type}** su **{vs_class}**…"):
                        risultati, tracked_bytes, csv_bytes = server_track(
                            SERVER_URL, video_salvato_path, trk_type, detections_data,
                            trk_kwargs_, target_class=vs_class, detection_id=detection_id,
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

        metriche_globali: dict = {}

        best_csv = collect_best_csvs(
            video_output_folder, list(TRACKER_REGISTRY.keys()), selected_classes
        )
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
                "id_switches":           "ID Switches",
                "fragmentation":         "Fragmentation",
                "kinematic_jumps":       "Kinematic Jumps",
                "track_coverage":        "Track Coverage %",
                "time":                  "Tempo di Elaborazione (s)",
                "avg_track_length":      "Lunghezza Media Tracce (frame)",
                "max_track_length":      "Lunghezza Massima Tracce (frame)",
                "num_tracks":            "Numero di Tracce Uniche",
                "avg_id_lifetime":       "Durata Vita Media ID (frame)",
                "max_id_lifetime":       "Durata Vita Massima ID (frame)",
                "total_detections":      "Rilevamenti Totali Tracciati",
                "spurious_tracks_ratio": "Rapporto Tracce Spurie",
            }

            classi_disponibili  = sorted({
                k.split("[")[1].rstrip("]") for k in metriche_globali if "[" in k
            })
            tracker_disponibili = sorted({
                k.split(" [")[0] for k in metriche_globali if "[" in k
            })
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
                df_filtrato   = df_comparativo.loc[righe_filtrate]
                lista_colori  = [
                    "#06b6d4", "#ea580c", "#7c3aed", "#10b981", "#e11d48",
                    "#2563eb", "#d946ef", "#84cc16", "#f59e0b", "#4f46e5",
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
                            st.bar_chart(df_filtrato[col_2], color=lista_colori[(i + 1) % len(lista_colori)], height=450)
        else:
            st.info("Nessuna metrica disponibile. Avvia l'elaborazione di almeno un tracker.")

    # -----------------------------------------------------------------------
    # TAB 5 — Ricerca Semantica
    # -----------------------------------------------------------------------
    with tab5:
        st.subheader("Ricerca Semantica nei Crop")

        available_trackers = [
            d for d in video_output_folder.iterdir()
            if d.is_dir() and any(d.glob("*_embeddings.npz"))
        ]

        if not available_trackers:
            st.info("Nessun embedding disponibile. Avvia prima un tracking con DeepSORT.")
        else:
            tracker_names = [d.name for d in available_trackers]
            selected_tracker_name = st.selectbox("Seleziona tracker:", tracker_names, key="tab5_tracker")
            selected_tracker_dir  = video_output_folder / selected_tracker_name

            npz_files = sorted(selected_tracker_dir.glob("*_embeddings.npz"))
            npz_names = [f.name for f in npz_files]
            selected_npz_name = st.selectbox(
                "Seleziona file embedding (classe):", npz_names, key="tab5_npz"
            )
            selected_npz_path = selected_tracker_dir / selected_npz_name

            npz_stem_parts = selected_npz_name.replace("_embeddings.npz", "").split("_")
            embedding_class = npz_stem_parts[-1] if len(npz_stem_parts) > 1 else "unknown"

            text_query = st.text_input(
                "Query testuale (es. 'red car', 'person with backpack'):", key="tab5_query"
            )
            top_k = st.slider("Numero risultati da mostrare:", 1, 20, 5, key="tab5_topk")

            if st.button("Cerca", key="tab5_search"):
                if not text_query.strip():
                    st.warning("Inserisci una query testuale.")
                else:
                    try:
                        from transformers import CLIPProcessor, CLIPModel

                        with st.spinner("Calcolo embedding testuale..."):
                            device = "cuda" if torch.cuda.is_available() else "cpu"
                            model  = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
                            proc   = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
                            inputs = proc(text=[text_query], return_tensors="pt", padding=True).to(device)
                            with torch.no_grad():
                                feat = model.get_text_features(**inputs)
                            if hasattr(feat, "pooler_output"):
                                feat = feat.pooler_output
                            elif hasattr(feat, "last_hidden_state"):
                                feat = feat.last_hidden_state.mean(dim=1)
                            feat = feat / feat.norm(p=2, dim=-1, keepdim=True)
                            text_embed = feat.cpu().numpy()[0]

                        db      = np.load(selected_npz_path)
                        keys    = list(db.files)
                        matrix  = np.stack([db[k] for k in keys])
                        sims    = matrix @ text_embed
                        top_idx = np.argsort(sims)[::-1][:top_k]

                        st.markdown(f"**Top {top_k} tracce per query: '{text_query}'**")
                        for rank, i in enumerate(top_idx):
                            key      = keys[i]
                            sim      = float(sims[i])
                            track_id = key.split("__")[-1]
                            crop_dir = selected_tracker_dir / embedding_class / track_id

                            st.markdown(f"**#{rank + 1} — Track {track_id} — similarità: {sim:.4f}**")
                            if crop_dir.exists():
                                frames = sorted(crop_dir.glob("*.jpg"))
                                if frames:
                                    cols = st.columns(min(len(frames), 6))
                                    for col, frame_path in zip(cols, frames[:6]):
                                        col.image(str(frame_path), width="stretch")
                                else:
                                    st.caption("Nessun crop disponibile per questa traccia.")
                            else:
                                st.caption(f"Cartella crop non trovata: `{crop_dir}`")

                    except ImportError:
                        st.error("transformers e torch non sono installati sul client.")
                    except Exception as exc:
                        st.error(f"Errore: {exc}")

    # -----------------------------------------------------------------------
    # TAB 6 — Analisi Attributi (YOLO-E)
    # -----------------------------------------------------------------------
    with tab6:
        st.subheader("Analisi Attributi Veicoli con YOLO-E")
        st.markdown(
            "Esegui YOLO-E sui crop già estratti dal tracker per ottenere "
            "attributi aggiuntivi (colore, tipo, e qualsiasi altra classe YOLO) "
            "aggregati per ogni `track_id`."
        )

        tracker_dirs_with_crops = (
            [d for d in video_output_folder.iterdir()
             if d.is_dir() and any(d.rglob("*.jpg"))]
            if video_output_folder.exists() else []
        )

        if not tracker_dirs_with_crops:
            st.info(
                "Nessun crop disponibile. "
                "Avvia prima un tracking con **Estrai crop** abilitato (Tab 1 o Tab 2)."
            )
        else:
            col_trk6, col_cls6 = st.columns(2)
            with col_trk6:
                tracker_dir_names = [d.name for d in tracker_dirs_with_crops]
                selected_trk_dir  = st.selectbox("Tracker:", tracker_dir_names, key="tab6_tracker")
                base_crops_dir    = video_output_folder / selected_trk_dir

            class_subdirs = [
                d for d in base_crops_dir.iterdir()
                if d.is_dir() and any(d.rglob("*.jpg"))
            ]
            class_names = [d.name for d in class_subdirs]

            with col_cls6:
                selected_cls = st.selectbox("Classe:", class_names, key="tab6_class") if class_names else None

            if selected_cls:
                crops_to_analyze = base_crops_dir / selected_cls
                track_ids_found  = [
                    d.name for d in crops_to_analyze.iterdir()
                    if d.is_dir() and any(d.glob("*.jpg"))
                ]
                st.caption(
                    f"**{len(track_ids_found)}** tracce trovate con crop disponibili "
                    f"in `{selected_trk_dir}/{selected_cls}`."
                )

            st.markdown("---")

            col_model6, col_conf6 = st.columns(2)
            with col_model6:
                YOLOE_MODELS = [
                    "yoloe-26n-seg.pt", "yoloe-26s-seg.pt", "yoloe-26m-seg.pt",
                    "yoloe-26l-seg.pt", "yoloe-26x-seg.pt",
                ]
                yoloe_model = st.selectbox("Modello YOLO-E:", YOLOE_MODELS, index=1, key="tab6_model")
                custom_yoloe = st.text_input(
                    "Oppure path/nome modello custom:", value="", key="tab6_custom_model"
                ).strip()
                if custom_yoloe:
                    yoloe_model = custom_yoloe

            with col_conf6:
                conf_thr6 = st.slider("Confidence threshold", 0.01, 1.0, 0.25, 0.01, key="tab6_conf")
                top_n6    = st.slider("Max frame per traccia da analizzare", 1, 20, 5, key="tab6_topn")

            st.markdown("##### Attributi da estrarre")
            st.caption(
                "**color** e **vehicle_type** usano logica dedicata. "
                "Qualsiasi altra stringa viene cercata come classe YOLO-E."
            )

            col_a1, col_a2 = st.columns([2, 1])
            with col_a1:
                preset_attrs = st.multiselect(
                    "Attributi predefiniti:", options=["color"],
                    default=["color"], key="tab6_attrs_preset",
                )
            with col_a2:
                custom_attrs_raw = st.text_input(
                    "Attributi YOLO custom (virgola-separati):", value="",
                    key="tab6_attrs_custom",
                    help="Es: person, bicycle, license_plate",
                ).strip()

            custom_attrs = [a.strip() for a in custom_attrs_raw.split(",") if a.strip()]
            final_attrs  = list(dict.fromkeys(preset_attrs + custom_attrs))

            if final_attrs:
                st.markdown(f"**Attributi selezionati:** `{', '.join(final_attrs)}`")
            else:
                st.warning("Seleziona almeno un attributo.")

            st.markdown("---")

            if st.button("Avvia Analisi Attributi", key="btn_tab6") and selected_cls:
                if not final_attrs:
                    st.error("Seleziona almeno un attributo prima di procedere.")
                else:
                    with st.spinner(
                        f"Analisi YOLO-E su {len(track_ids_found)} tracce "
                        f"({selected_trk_dir} / {selected_cls})…"
                    ):
                        analysis_results = server_analyze_crops(
                            SERVER_URL, crops_to_analyze, yoloe_model,
                            final_attrs, conf_thr6, top_n6,
                        )

                    if analysis_results:
                        st.success(f"Analisi completata su {len(analysis_results)} tracce.")

                        ts          = datetime.now().strftime("%Y%m%d_%H%M%S")
                        result_path = (
                            video_output_folder / selected_trk_dir
                            / f"attributes_{selected_cls}_{ts}.json"
                        )
                        result_path.write_text(
                            json.dumps(analysis_results, indent=2, ensure_ascii=False)
                        )
                        st.caption(f"Risultati salvati in: `{result_path}`")

                        st.markdown("#### Tabella Riassuntiva")
                        table_rows = []
                        for key, info in analysis_results.items():
                            row = {
                                "track_id":   info.get("track_id", key),
                                "classe":     info.get("class", ""),
                                "n_frame":    info.get("num_frames_analyzed", 0),
                                "confidenza": info.get("overall_confidence", 0.0),
                            }
                            for attr in final_attrs:
                                row[attr]           = info.get(attr, "—")
                                row[f"{attr}_conf"] = info.get(f"{attr}_confidence", 0.0)
                            table_rows.append(row)

                        if table_rows:
                            # Attributi YOLO = tutti tranne "color"
                            yolo_attrs = [a for a in final_attrs if a != "color"]
                            
                            for row in table_rows:
                                # Conta quanti attributi YOLO sono stati rilevati (valore non nullo/non vuoto)
                                found = 0
                                confs = []
                                for attr in yolo_attrs:
                                    val = row.get(attr)
                                    # Considera rilevato se il valore non è None, "—", "unknown", o stringa vuota
                                    if val and val != "—" and val != "None" and val != "unknown":
                                        found += 1
                                        confs.append(row.get(f"{attr}_conf", 0.0))
                                row["num_yolo_found"] = found
                                row["yolo_avg_conf"] = sum(confs) / found if found > 0 else 0.0

                            # Se non ci sono attributi YOLO (solo "color"), usa la confidenza complessiva
                            if not yolo_attrs:
                                for row in table_rows:
                                    row["num_yolo_found"] = 1  # dummy per l'ordinamento
                                    row["yolo_avg_conf"] = row.get("confidenza", 0.0)

                            # Ordina: prima per numero di attributi trovati (decrescente), poi per media confidenza (decrescente)
                            df_attr = pd.DataFrame(table_rows).sort_values(
                                ["num_yolo_found", "yolo_avg_conf"], ascending=[False, False]
                            )
                            st.dataframe(df_attr, use_container_width=True)

                        st.markdown("#### Dettaglio per Traccia")
                        def detail_sort_key(item):
                            _, info = item

                            yolo_attrs = [a for a in final_attrs if a != "color"]

                            # Caso speciale: solo colore
                            if not yolo_attrs:
                                return (
                                    1,
                                    info.get("overall_confidence", 0.0)
                                )

                            found = 0
                            confs = []

                            for attr in yolo_attrs:
                                val = info.get(attr)

                                if val and val not in ["—", "None", "unknown"]:
                                    found += 1
                                    confs.append(info.get(f"{attr}_confidence", 0.0))

                            avg_conf = sum(confs) / found if found > 0 else 0.0

                            return (
                                found,
                                avg_conf
                            )

                        for key, info in sorted(
                            analysis_results.items(),
                            key=detail_sort_key,
                            reverse=True,
                        ):
                            track_id     = info.get("track_id", key)
                            attr_summary = "  |  ".join(
                                f"**{a}**: {info.get(a, '—')} "
                                f"({info.get(f'{a}_confidence', 0):.2f})"
                                for a in final_attrs if info.get(a)
                            )
                            with st.expander(
                                f"Track {track_id}  —  {attr_summary}  "
                                f"(conf. media: {info.get('overall_confidence', 0):.2f})"
                            ):
                                track_crop_dir = crops_to_analyze / track_id
                                if track_crop_dir.exists():
                                    crop_imgs = sorted(track_crop_dir.glob("*.jpg"))
                                    if crop_imgs:
                                        cols = st.columns(min(len(crop_imgs), 6))
                                        for col, img_p in zip(cols, crop_imgs[:6]):
                                            col.image(str(img_p), use_container_width=True)

                                per_frame = info.get("per_frame", [])
                                if per_frame:
                                    st.markdown("**Per frame:**")
                                    st.dataframe(pd.DataFrame(per_frame), use_container_width=True)