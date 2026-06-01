import cv2
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from config import *

class Visualizer:
    """ Classe VISUALIZER
        si occupa di:
            - disegnare i track sulle bbox nei frame video (overlay visuale)
            - creare grafici di comparazione tra gli algoritmi
            - salvare video e immagini PNG con i risultati
        Output:
            1. video con bbox + track_id scritti
            2. grafici PNG di comparazione metrica per metrica
            3. grafico PNG dei tempi di elaborazione
    """
    def __init__(self, par_output_dir: str = "output"):
        """ Inizializza il visualizer.
                Args:
                    par_output_dir: cartella di output per video e grafici
        """
        self.output_dir = Path(par_output_dir)
        self. output_dir.mkdir(parents=True, exist_ok=True) # Crea la cartella di output se non esiste
        
    def draw_tracks(self, par_csv_path: str, par_output_video_name: str, par_video_path: str, progress_callback = None):
        """ Crea un video annotato con i track disegnati:
                1. legge il CSV con tutti i track
                2. apre il video originale
                3. per ogni frame:
                    - estrae le track di quel frame dal csv
                    - disegna le bbox e gli ID
                    - scrive nel video di output
                
                Input:
                    par_csv_path: path al CSV del tracker
                    par_output_video_name: nome file di output
                    par_video_path: path al video originale
                Output:
                    video MP4 con rettangoli verdi e ID numerico per ogni traccia
        """
        cap = cv2.VideoCapture(par_video_path)
        width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps    = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        out_path = self.output_dir / par_output_video_name
        out = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
        
        try:
            df = pd.read_csv(par_csv_path)
        except Exception:
            df = pd.DataFrame(columns=["frame", "track_id", "x1", "y1", "x2", "y2"])

        frame_id = 0
        while True:
            ret, frame = cap.read()
            if not ret: break
            
            frame_data = df[df["frame"] == frame_id]
            for _, row in frame_data.iterrows():
                x1, y1, x2, y2 = int(row["x1"]), int(row["y1"]), int(row["x2"]), int(row["y2"])
                t_id = int(row["track_id"])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"ID: {t_id}", (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            out.write(frame)
            frame_id += 1
            
            if progress_callback is not None and total_frames > 0:
                progress_callback(frame_id, total_frames)
            
        cap.release()
        out.release()
        print(f"Video salvato: {self.output_dir / par_output_video_name}")
        
    def plot_comparison(self, par_all_metrics: dict):
        """ Crea un grafico di comparazione di TUTTI gli algoritmi.
            Input:
                par_all_metrics: dizionario con risultati di tutti i tracker
                               formato: {
                                  "DEEPSORT": {metriche},
                                  "SORT": {metriche},
                                  ...
                               }
            Output:
                PNG salvato in "output/tracker_comparison.png"
                Contiene 9 subplot (uno per metrica)
        """
        names = list(par_all_metrics.keys())
        
        if not names:
            print("ERRORE, non sono presenti dati per generare il grafico di comparazione")
            return
        
        # Tutte le metriche da visualizzare, ogni tupla: (chiave_del_dizionario, titolo_da_mostrare)
        metrics_to_plot = [
            ("id_switches", "ID Switches"),
            ("fragmentation", "Fragmentation"),
            ("num_tracks", "Numero di tracce"),
            ("avg_track_length", "Lunghezza media"),
            ("track_coverage", "Copertura tracce %"), 
            ("total_detections", "Detection totali"),
            ("spurious_tracks_ratio", "% Tracce Spurie"),
            ("kinematic_jumps", "Salti di Velocità"),
            ("aspect_ratio_variance", "Varianza Forma BBox")
        ]
        
        num_metrics = len(metrics_to_plot)
        ncols = 3  # Teniamo 3 colonne fisse
        # Calcola le righe necessarie arrotondando per eccesso
        nrows = (num_metrics + ncols - 1) // ncols  
        
        # Adatta l'altezza della figura in base al numero di righe
        fig, axs = plt.subplots(nrows, ncols, figsize=(15, 4 * nrows))
        fig.suptitle("Confronto algoritmi di tracking", fontsize=16, fontweight="bold")
        
        axs = axs.flatten() # Trasforma la matrice di grafici in una lista piatta
        
        for i, (key, title) in enumerate(metrics_to_plot):
            values = [par_all_metrics[name].get(key, 0) for name in names] # Estrae i valori della metrica per tutti i tracker (il valore di default se non presente è messo a 0)
            
            bars = axs[i].bar(names, values, color=plt.cm.tab10.colors[:len(names)]) # Crea il grafico a barre
            axs[i].set_title(f"{title}\n", fontsize=10)
            axs[i].tick_params(axis='x', rotation=25) # Ruota i nomi dei tracker sull'asse X (così non si sovrappongono)
            
            for bar, val in zip(bars, values): # Aggiunge il valore in cima alla barra
                axs[i].text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() * 1.01,
                    str(round(val, 1)),
                    ha='center', va='bottom', fontsize=8
                )
        
        # --- RIMOZIONE RIQUADRI VUOTI ---
        # Se il numero di metriche non è un multiplo perfetto di 3, elimina i grafici vuoti rimasti
        for j in range(num_metrics, len(axs)):
            fig.delaxes(axs[j])
            
        plt.tight_layout()
        
        out_path = self.output_dir / "tracker_comparison.png"
        plt.savefig(out_path, dpi=150)
        plt.close()
        print(f"Grafico di confronto salvato in: {out_path}")
        
    def plot_processing_time(self, par_all_metrics: dict):
        """ Crea un grafico dei tempi di elaborazione.
                Input:
                    par_all_metrics: dizionario con risultati di tutti i tracker
                Output:
                    PNG salvato in "output/processing_time.png"
        """
        names = list(par_all_metrics.keys())
        times = [par_all_metrics[name].get("time", 0) for name in names] # Se un tracker non ha il tempo, usa 0
        
        fig, ax = plt.subplots(figsize=(10, 6)) # Crea una figura con un singolo subplot
        ax.bar(names, times, color='steelblue', edgecolor='black')
        ax.set_ylabel("Tempo (secondi)", fontweight='bold')
        ax.set_title("Tempo di Elaborazione", fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "processing_time.png", dpi=150)
        plt.close()

    def draw_raw_detections(self, par_detections_data: list, par_output_video_name: str, par_video_path: str, progress_callback = None):
        """ Crea un video annotato con le sole detection grezze di YOLO (BBox Blu + Confidenza) """
        cap = cv2.VideoCapture(par_video_path)
        width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps    = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        out_path = self.output_dir / par_output_video_name
        out = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

        det_dict = {d["frame_id"]: d["detections"] for d in par_detections_data}
        frame_id = 0

        while True:
            ret, frame = cap.read()
            if not ret: break

            if frame_id in det_dict:
                for det in det_dict[frame_id]:
                    x1, y1, x2, y2, conf = det
                    x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                    cv2.putText(frame, f"YOLO {conf:.2f}", (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

            out.write(frame)
            frame_id += 1
            
            if progress_callback is not None and total_frames > 0:
                progress_callback(frame_id, total_frames)

        cap.release()
        out.release()
