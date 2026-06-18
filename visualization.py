import cv2
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Palette colori BGR per classi diverse
CLASS_COLOR_PALETTE = [
    (0, 0, 255),       # rosso
    (0, 255, 0),       # verde
    (255, 0, 0),       # blu
    (0, 255, 255),     # giallo
    (255, 0, 255),     # magenta
    (255, 255, 0),     # ciano
    (128, 0, 255),     # arancione
    (255, 128, 0),     # azzurro
    (128, 255, 0),     # lime
    (0, 128, 255),     # rosa
]

def get_class_color(index: int):
    return CLASS_COLOR_PALETTE[index % len(CLASS_COLOR_PALETTE)]


class Visualizer:
    """ Classe VISUALIZER """

    def __init__(self, par_output_dir: str = "output"):
        self.output_dir = Path(par_output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # draw_tracks  — versione originale
    # ------------------------------------------------------------------
    def draw_tracks(
        self,
        par_csv_path: str,
        par_output_video_name: str,
        par_video_path: str,
        color: tuple = (0, 255, 0),
        label: str = "",
        progress_callback=None,
    ):
        """Disegna le tracce di un singolo CSV sul video."""
        cap          = cv2.VideoCapture(par_video_path)
        width        = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height       = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps          = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        out_path = self.output_dir / par_output_video_name
        out = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

        try:
            df = pd.read_csv(par_csv_path, comment="#")
        except Exception:
            df = pd.DataFrame(columns=["frame", "track_id", "x1", "y1", "x2", "y2"])

        frame_id = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_data = df[df["frame"] == frame_id]
            for _, row in frame_data.iterrows():
                x1, y1, x2, y2 = int(row["x1"]), int(row["y1"]), int(row["x2"]), int(row["y2"])
                t_id = int(row["track_id"])
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                tag = f"{label} ID:{t_id}" if label else f"ID:{t_id}"
                cv2.putText(frame, tag, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
            out.write(frame)
            frame_id += 1
            if progress_callback and total_frames > 0:
                progress_callback(frame_id, total_frames)

        cap.release()
        out.release()
        print(f"Video salvato: {out_path}")

    # ------------------------------------------------------------------
    # draw_multi_class_tracks 
    # ------------------------------------------------------------------
    def draw_multi_class_tracks(
        self,
        track_layers: list,             # lista di dict: {"csv_path", "color", "label"}
        par_output_video_name: str,
        par_video_path: str,
        progress_callback=None,
    ):
        """
        Disegna più layer di tracking (una classe per layer) su un unico video.

        Args:
            track_layers: lista di dizionari con chiavi:
                - csv_path (str): path al CSV del tracker per quella classe
                - color (tuple): colore BGR, es. (0,255,0)
                - label (str): etichetta da mostrare, es. "car" o "person"
            par_output_video_name: nome file video di output
            par_video_path: path al video originale
        """
        cap          = cv2.VideoCapture(par_video_path)
        width        = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height       = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps          = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        out_path = self.output_dir / par_output_video_name
        out = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

        # Carica tutti i CSV in memoria
        dataframes = []
        for layer in track_layers:
            try:
                df = pd.read_csv(layer["csv_path"], comment="#")
            except Exception:
                df = pd.DataFrame(columns=["frame", "track_id", "x1", "y1", "x2", "y2"])
            dataframes.append({
                "df":    df,
                "color": layer["color"],
                "label": layer.get("label", ""),
            })

        # Disegna legenda in alto a sinistra
        def draw_legend(frame):
            y_offset = 20
            for entry in dataframes:
                color = entry["color"]
                lbl   = entry["label"]
                cv2.rectangle(frame, (8, y_offset - 12), (22, y_offset + 2), color, -1)
                cv2.putText(frame, lbl, (28, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
                y_offset += 22

        frame_id = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            for entry in dataframes:
                df_frame = entry["df"][entry["df"]["frame"] == frame_id]
                color    = entry["color"]
                label    = entry["label"]
                for _, row in df_frame.iterrows():
                    x1, y1, x2, y2 = int(row["x1"]), int(row["y1"]), int(row["x2"]), int(row["y2"])
                    t_id = int(row["track_id"])
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    tag = f"{label} ID:{t_id}" if label else f"ID:{t_id}"
                    cv2.putText(frame, tag, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

            draw_legend(frame)
            out.write(frame)
            frame_id += 1
            if progress_callback and total_frames > 0:
                progress_callback(frame_id, total_frames)

        cap.release()
        out.release()
        print(f"Video multi-classe salvato: {out_path}")

    # ------------------------------------------------------------------
    # draw_raw_detections
    # ------------------------------------------------------------------
    def draw_raw_detections(
        self,
        par_detections_data: list,
        par_output_video_name: str,
        par_video_path: str,
        progress_callback=None,
    ):
        """Disegna bbox grezze YOLO multi-classe con colori per classe."""
        cap          = cv2.VideoCapture(par_video_path)
        width        = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height       = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps          = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        out_path = self.output_dir / par_output_video_name
        out = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

        det_dict   = {d["frame_id"]: d["detections"] for d in par_detections_data}
        class_colors = {}   # class_name -> colore BGR

        frame_id = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_id in det_dict:
                for det in det_dict[frame_id]:
                    x1, y1, x2, y2, conf = det[0], det[1], det[2], det[3], det[4]
                    class_name = det[5] if len(det) >= 6 else "obj"
                    x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])

                    if class_name not in class_colors:
                        class_colors[class_name] = get_class_color(len(class_colors))
                    color = class_colors[class_name]

                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame, f"{class_name} {conf:.2f}", (x1, y1 - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            out.write(frame)
            frame_id += 1
            if progress_callback and total_frames > 0:
                progress_callback(frame_id, total_frames)

        cap.release()
        out.release()

    # ------------------------------------------------------------------
    # plot_comparison / plot_processing_time
    # ------------------------------------------------------------------
    def plot_comparison(self, par_all_metrics: dict):
        names = list(par_all_metrics.keys())
        if not names:
            return
        metrics_to_plot = [
            ("id_switches", "ID Switches"), ("fragmentation", "Fragmentation"),
            ("num_tracks", "Numero di tracce"), ("avg_track_length", "Lunghezza media"),
            ("track_coverage", "Copertura tracce %"), ("total_detections", "Detection totali"),
            ("spurious_tracks_ratio", "% Tracce Spurie"), ("kinematic_jumps", "Salti di Velocità"),
            ("aspect_ratio_variance", "Varianza Forma BBox"),
        ]
        ncols = 3
        nrows = (len(metrics_to_plot) + ncols - 1) // ncols
        fig, axs = plt.subplots(nrows, ncols, figsize=(15, 4 * nrows))
        fig.suptitle("Confronto algoritmi di tracking", fontsize=16, fontweight="bold")
        axs = axs.flatten()
        for i, (key, title) in enumerate(metrics_to_plot):
            values = [par_all_metrics[name].get(key, 0) for name in names]
            bars = axs[i].bar(names, values, color=plt.cm.tab10.colors[:len(names)])
            axs[i].set_title(f"{title}\n", fontsize=10)
            axs[i].tick_params(axis="x", rotation=25)
            for bar, val in zip(bars, values):
                axs[i].text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.01,
                            str(round(val, 1)), ha="center", va="bottom", fontsize=8)
        for j in range(len(metrics_to_plot), len(axs)):
            fig.delaxes(axs[j])
        plt.tight_layout()
        out_path = self.output_dir / "tracker_comparison.png"
        plt.savefig(out_path, dpi=150)
        plt.close()

    def plot_processing_time(self, par_all_metrics: dict):
        names = list(par_all_metrics.keys())
        times = [par_all_metrics[name].get("time", 0) for name in names]
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.bar(names, times, color="steelblue", edgecolor="black")
        ax.set_ylabel("Tempo (secondi)", fontweight="bold")
        ax.set_title("Tempo di Elaborazione", fontweight="bold")
        ax.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        plt.savefig(self.output_dir / "processing_time.png", dpi=150)
        plt.close()
