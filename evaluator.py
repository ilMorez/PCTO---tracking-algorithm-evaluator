import pandas as pd
import numpy as np

class TrackerEvaluator:
    """CLASSE: TrackerEvaluator
        Si occupa di effettuale la valutaizone di un algoritmo di trakcking.
        calcola 9 metriche per andare a misurare la qualità di un tracker:
            - lunghezza delle singole tracce
            - cambi di ID
            - frammentazione delle tracce
            - copertura temporale
            - salti cinematici anomali
            - rapporto di tracce spurie (dei probabili falsi positivi)
            - varianza nel rapposto di aspetto delle bounding box
    """
    def __init__(self, par_total_frames: int):
        self.total_frames = par_total_frames # Il numero totale di frame di un video viene utilizzato per normalizzare le metriche su video diversi
        
    
    def _compute_track_lengths(self, df: pd.DataFrame): 
        """ Calcola la lunghezza delle singole track (numero di frame):
                - raggruppa il dataframe per track_id (singole tracce)
                - conta i frame unici per ogni traccia
            output: Series con index=track_id, vaues=lunghezza
        """
        return df.groupby("track_id")["frame"].count()
    
    def _compute_id_switches(self, df: pd.DataFrame) -> int: 
        """ Conta i cambi di indentità, ovvero quando un tracker assegna nuovi ID  a oggetti che dovrebbero mantenere lo stesso ID:
                1. ordina per frame
                2. par ogni farme estrae gli ID presenti
                3. confronta gli ID di questo frame con il frame precedente
                4. se compaiono nuovi ID inaspettati -> ID switch
            Si tratta di un'approssimazione non posedendo gorund truth per essere sicuri che ci sia cambio ID su stesso oggetto
        """
        df = df.sort_values("frame")
        switches = 0
        prev_ids = set()

        for frame in sorted(df["frame"].unique()):
            curr_ids = set(df.loc[df["frame"] == frame, "track_id"])
            if prev_ids:
                switches += len(curr_ids - prev_ids) # Aggiunge il numero di ID diversi (differenza fra i due set)
            prev_ids = curr_ids

        return switches
    
    def _compute_fragmentation(self, df: pd.DataFrame) -> int:
        """ Misura la frammetnazione delle tracce. 
            Una track perfetta occupa frame consecutivi, una frammentata ha dei "buchi" ovvero dei frame mancanti:
                1. estra i frame in cui appare ogni track_id
                2. calcola le differenze fra due frame (ordinati)
                3. se la differenza è maggiore di 1 si ha della fragmentation
                4. conta qunati buchi sono presenti
            interpretazione:
                - fragmentation = 0: track continua
                - fragmentation > 0: il tracker ha perso l'oggetto e l'ha ripreso, magari per occlusione
        """
        frag = 0
        for _, group in df.groupby("track_id"):
            frames = sorted(group["frame"].values)
            frag  += int(np.sum(np.diff(frames) > 1))
        return frag
    
    def _compute_track_coverage(self, df: pd.DataFrame) -> float:
        """ Misura la percentuale di viddeo coperta dalle track (fa una media normalizzata):
                1. calcola la lunghezza di ogni track
                2. normalizza dividendo per i frame totale
                3. fa la media di tutte le track
                4. converte in percentuale
        """
        lengths = self._compute_track_lengths(df)
        return float((lengths / self.total_frames).mean() * 100)
    
    def _compute_id_persistence(self, df: pd.DataFrame):
        """ Misura la persistenza degli ID, quindi qunato a lungo un ID rimane presente nel video (senza interruzioni).
            Il lifetime di una traccia viene calcolato aggiungendo uno alla differenza fra il primo frame e l'ultimo frame in cui è presente (conteggio frame parte da 0).
            In output abbiamo un dizionario con il lifetime medio, il lifetime massimo e il lifetime minimo (rispettivamente media delle track, track massima e track minima)
        """
        groups = df.groupby("track_id")["frame"].apply(list)
        lifetimes = []

        for frames in groups:
            frames = sorted(frames)
            lifetime = frames[-1] - frames[0] + 1
            lifetimes.append(lifetime)

        return {
            "avg_id_lifetime": float(np.mean(lifetimes)),
            "max_id_lifetime": int(np.max(lifetimes)),
            "min_id_lifetime": int(np.min(lifetimes))
        }
    
    def _compute_spurious_tracks_ratio(self, df: pd.DataFrame, par_min_frames: int = 5) -> float:
        """ Stima la percentuale di tracce spurie (dei possibili falsi positivi in quanto molto brevi: meno di par_min_frames):
            una track che dura poco è un probabile errore come un rilevamento sbagliato di YOLO per pochi frame oppure come rumore/oggetto fantasma.
            Resistuisce una percentuale di quate track sono spurie. 
            Si tratta di approssimazione non avendo ground truth
        """
        if df.empty:
            return 0.0
        lengths = self._compute_track_lengths(df)
        total_tracks = len(lengths)
        if total_tracks == 0:
            return 0.0
        
        # Conta quante tracce durano meno di 'min_frames'
        spurious_tracks = (lengths < par_min_frames).sum()
        return round(float((spurious_tracks / total_tracks) * 100), 2)

    def _compute_kinematic_jumps(self, df: pd.DataFrame, pixel_threshold_per_frame: float = 80.0) -> int:
        """ Conta i salti anomali nel movimento degli oggetti:
            un oggetto non può teletrasportarsi, quindi se la sua velocità suepra una soglia probabilmente si tratta di
            o un ID switch, o di un errore di rilevamento o di occlusion e ridetection sbagliata.
            logica:
                1. calcola il centroide della bbox
                2. per ognii track calcola le velocità consecutive
                3. se velocity è maggiore di una soglia si tratta di un salto anomalo.
            output: numero totale di salti rilevati
        """
        if df.empty:
            return 0
        
        # Calcola i centroidi
        df = df.copy()
        df["xc"] = (df["x1"] + df["x2"]) / 2
        df["yc"] = (df["y1"] + df["y2"]) / 2
        
        # Ordina per traccia e per frame
        df = df.sort_values(["track_id", "frame"])
        jumps = 0
        
        for _, group in df.groupby("track_id"):
            if len(group) < 2:
                continue
            
            # Differenze consecutive di spazio e tempo
            diff_x = group["xc"].diff().dropna()
            diff_y = group["yc"].diff().dropna()
            diff_frame = group["frame"].diff().dropna()
            
            # Distanza euclidea tra frame consecutivi
            distances = np.sqrt(diff_x**2 + diff_y**2)
            
            # Velocità (pixel al frame) per gestire eventuali frame saltati (FRAME_SKIP)
            speeds = distances / diff_frame
            
            # Se la velocità supera la soglia, consideriamo un salto anomalo
            jumps += int((speeds > pixel_threshold_per_frame).sum())
            
        return jumps

    def _compute_aspect_ratio_variance(self, df: pd.DataFrame) -> float:
        """ Misura l'instabilità geometrica delle bbox (ovvero al variazione del rapporto lunghezza altezza).
            un oggeto reale ha diemsnioni stabili, quindi se l'aspect ratio varia molto il tracker sta distorcendo l'oggetto:
                1. calcola l'aspect_ratio per ogni detection
                2. per ogni track calcola la varianza dell'aspect ratio
                3. calcola la media della variazione di tutte le track
        """
        if df.empty:
            return 0.0
        
        df = df.copy()
        w = df["x2"] - df["x1"]
        h = df["y2"] - df["y1"]
        
        # Evita divisioni per zero nel caso di bbox corrotte
        h = h.replace(0, 1e-5)
        df["aspect_ratio"] = w / h
        
        variances = []
        for _, group in df.groupby("track_id"):
            # Calcolia la varianza solo se la traccia ha almeno 3 punti
            if len(group) >= 3:
                variances.append(group["aspect_ratio"].var())
                
        valid_variances = [v for v in variances if not np.isnan(v)] # Filtra eventuali valori NaN
        
        return round(float(np.mean(valid_variances)), 4) if valid_variances else 0.0
        
    def evaluate(self, par_csv_path: str) -> dict:
        """ Legge il CSV di un tracker e ne calcola tutte le metriche di valutazione all'interno di un dict
        """
        df = pd.read_csv(par_csv_path, comment='#')
        df = df[df["track_id"] != -1]
        lengths = self._compute_track_lengths(df)
        persistence = self._compute_id_persistence(df)

        return {
            "total_detections": len(df),
            "num_tracks": int(df["track_id"].nunique()),
            "avg_track_length": round(float(lengths.mean()), 2),
            "max_track_length": int(lengths.max()),
            "id_switches": self._compute_id_switches(df),
            "fragmentation": self._compute_fragmentation(df),
            "track_coverage": round(self._compute_track_coverage(df), 2),
            "avg_id_lifetime": round(persistence["avg_id_lifetime"], 2),
            "max_id_lifetime": persistence["max_id_lifetime"],
            "time": round(df["time"].max(), 3),
            "spurious_tracks_ratio": self._compute_spurious_tracks_ratio(df, par_min_frames=5),
            "kinematic_jumps": self._compute_kinematic_jumps(df, pixel_threshold_per_frame=80.0),
            "aspect_ratio_variance": self._compute_aspect_ratio_variance(df)
        }