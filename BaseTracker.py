from abc import ABC, abstractmethod
from pathlib import Path
import csv

class BaseTracker(ABC):
    """ Classe astratta BASE TRACKER, si coccupa di:
            - definire l'interfaccia STANDARD per tutti gli algoritmi di tracking
            - gestire le CARTELLE DI OUTPUT (creazione automatica)
            - salvare i risultati in CSV (formato standardizzato per evaluator)
        
        Principio OOP: ogni tracker (DeepSORT, SORT, etc) deve ereditare da questa classe e implementare il metodo run().
        Tutti i tracker hanno in questo modo la stessa interfaccia, anche se l'implementazione interna è completamente diversa (polimorfismo)
        l'impelmentazione nel main rimane uguale, permetetndo di aggiungere facilemnte nuovi tracker in ogni monmento lasicando invariato il resto del codice.
    """

    def __init__(self, par_name: str, par_output_dir: str = "output"):
        """ Inizializza il tracker base
                Args:
                    par_name: nome univoco del tracker
                    par_output_dir: cartella dove salvare i risultati
            """
        self.name = par_name
        self.output_dir = Path(par_output_dir)
        
    @abstractmethod
    def run(self, detections: list, progress_callback=None) -> list:
        """ Metodo astratto che esegue l'algoritmo di tracking sulle detection di un video.
                Input:
                    detections: lista di frame con detections YOLO nel formato: [{"frame_id": 0, "detections": [[x1, y1, x2, y2, conf], ...]}, ...]
                Output:
                    lista di risultati tracciati nel formato: [{"frame": 0, "track_id": 1, "x1": 100, "y1": 50, "x2": 200, "y2": 150, "time": 0.001}, ...]
        """
        pass
    
    def setup_video_paths(self, par_video_name: str):
        """ Prepara le cartelle di output per un video specifico.
                Crea una struttura così:
                    output/
                    ├── video1/
                    │   ├── deepsort.csv
                    │   └── ...
                    └── video2/
                        └── ...
                Args:
                    par_video_name: path o nome del video (es. "video/traffic.mp4")
        """
        video_folder = self.output_dir / Path(par_video_name).stem # Path.stem estrae il nome del file senza estensione
        video_folder.mkdir(parents=True, exist_ok=True) # Crea la cartella (parents=True crea anche le parent directories se non esistono, exist_ok=True evita errore se la cartella esiste già)
        self.csv_path = video_folder / f"{self.name.lower()}.csv" # Salva il path completo del CSV di questo tracker per questo video
    
    def save_to_csv(self, results: list):
        """ Salva i risultati del tracking in un file CSV standardizzato:
            il file CSV ha sempre lo stesso formato per permettere all'evaluator di calcolare le metriche in modo uniforme.
            Formato CSV:
                frame | track_id | x1   | y1  | x2   | y2   | time
                ------|----------|------|-----|------|------|--------
                0     | 1        | 100  | 50  | 200  | 150  | 0.001
            Args:
                results: lista di dizionari con le tracce tracciate
                        ogni elemento: {"frame": int, "track_id": int, "x1": float, "y1": float, "x2": float, "y2": float, "time": float}
        """
        header_names = ["frame", "track_id", "x1", "y1", "x2", "y2", "time"]
        with open(self.csv_path, "w", newline="") as file: # Apre il file in modalità WRITE (sovrascrive)
            writer = csv.DictWriter(file, fieldnames=header_names) # Crea un writer che scrive dizionari come righe CSV
            writer.writeheader()
            writer.writerows(results)
            print(f"Risultati di {self.name} salvati in {self.csv_path}")
