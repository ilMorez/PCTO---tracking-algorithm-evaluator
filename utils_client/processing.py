"""
utils/processing.py
Logiche di calcolo metriche e parsing dei file di ground truth.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from evaluator import TrackerEvaluator


# ---------------------------------------------------------------------------
# Metriche da CSV
# ---------------------------------------------------------------------------

def calcola_metriche_csv(csv_path: Path) -> dict | None:
    """Calcola le metriche di tracking leggendo un file CSV prodotto dal tracker."""
    try:
        df_temp      = pd.read_csv(csv_path, comment="#")
        total_frames = int(df_temp["frame"].max() + 1) if not df_temp.empty else 1
        evaluator    = TrackerEvaluator(par_total_frames=total_frames)
        return evaluator.evaluate(str(csv_path))
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Ground Truth
# ---------------------------------------------------------------------------

def parse_gt_file(gt_file_obj) -> pd.DataFrame | None:
    """Parsifica un file ground truth in formato MOT.

    Formato atteso: frame, track_id, x, y, w, h, conf, class, visibility
    """
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
            rows.append({"frame": frame_id, "track_id": track_id,
                         "x": x, "y": y, "w": w, "h": h})
        return pd.DataFrame(rows) if rows else None
    except Exception:
        return None


def calcola_metriche_gt(gt_df: pd.DataFrame) -> dict:
    """Deriva metriche statistiche direttamente dal DataFrame ground truth."""
    total_frames  = int(gt_df["frame"].max()) + 1
    unique_tracks = gt_df["track_id"].nunique()
    track_lengths = gt_df.groupby("track_id")["frame"].count()
    avg_len       = float(track_lengths.mean())
    max_len       = int(track_lengths.max())
    total_det     = len(gt_df)

    lifetimes     = gt_df.groupby("track_id").apply(
        lambda g: g["frame"].max() - g["frame"].min() + 1
    )
    avg_lifetime  = float(lifetimes.mean())
    max_lifetime  = int(lifetimes.max())

    track_coverage = (
        (total_det / (unique_tracks * total_frames) * 100) if unique_tracks > 0 else 0.0
    )

    return {
        "num_tracks":            unique_tracks,
        "total_detections":      total_det,
        "avg_track_length":      round(avg_len, 2),
        "max_track_length":      max_len,
        "avg_id_lifetime":       round(avg_lifetime, 2),
        "max_id_lifetime":       max_lifetime,
        "track_coverage":        round(track_coverage, 4),
        "id_switches":           float("nan"),
        "fragmentation":         float("nan"),
        "kinematic_jumps":       float("nan"),
        "spurious_tracks_ratio": float("nan"),
        "time":                  float("nan"),
    }


# ---------------------------------------------------------------------------
# Aggregazione CSV per la tab Analytics
# ---------------------------------------------------------------------------

def collect_best_csvs(
    video_output_folder: Path,
    tracker_registry_keys: list[str],
    selected_classes: list[str],
) -> dict[tuple[str, str], Path]:
    """Per ogni coppia (tracker, classe) trova il CSV più recente nella cartella di output."""
    all_csvs = sorted(video_output_folder.rglob("*.csv"))
    tracker_names_lower = {t.lower(): t for t in tracker_registry_keys}
    best_csv: dict[tuple[str, str], Path] = {}

    for csv_path in all_csvs:
        stem = csv_path.stem
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

    return best_csv