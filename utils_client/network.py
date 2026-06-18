"""
utils/network.py
Tutte le chiamate HTTP verso il server FastAPI.
"""

from __future__ import annotations

import base64
import csv as _csv
import io
import json
import zipfile
from pathlib import Path

import requests
import streamlit as st


# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------

def server_detect(
    server_url: str,
    video_path: Path,
    model_name: str,
    yolo_params: dict,
    target_classes: list,
) -> tuple[list | None, str | None, bytes | None]:
    """Esegue la detection sul server.

    Restituisce (detections, detection_id, yolo_video_bytes).
    Il video grezzo NON è incluso nella risposta JSON principale (niente base64):
    se il server fornisce un detection_id, il video va scaricato separatamente
    con server_fetch_detect_video().
    """
    endpoint = f"{server_url}/api/detect"
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
            _show_error(resp, "/api/detect")
            return None, None, None

        data         = resp.json()
        detections   = data.get("detections")
        detection_id = data.get("detection_id")

        video_bytes: bytes | None = None
        if data.get("has_raw_video") and detection_id:
            video_bytes = server_fetch_detect_video(server_url, detection_id)
        elif data.get("yolo_video"):
            # Fallback legacy: server vecchio che manda ancora il video in base64
            video_bytes = base64.b64decode(data["yolo_video"])

        return detections, detection_id, video_bytes

    except requests.exceptions.ConnectionError:
        st.error(f"Impossibile connettersi al server: `{endpoint}`")
        return None, None, None
    except Exception as exc:
        st.error(f"Errore `/api/detect`:\n```\n{exc}\n```")
        return None, None, None


def server_fetch_detect_video(server_url: str, detection_id: str) -> bytes | None:
    """Scarica in streaming il video con i rilevamenti grezzi, dato un detection_id."""
    endpoint = f"{server_url}/api/detect/{detection_id}/video"
    try:
        resp = requests.get(endpoint, timeout=600, proxies={"http": None, "https": None})
        return resp.content if resp.status_code == 200 else None
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Tracking singolo
# ---------------------------------------------------------------------------

def server_track(
    server_url: str,
    video_path: Path,
    tracker_name: str,
    detections_data: list | None,
    tracker_params: dict,
    *,
    target_class: str = "",
    extract_crops: bool = False,
    output_dir: Path | None = None,
    detection_id: str | None = None,
    embeddings_dir: Path | None = None,
) -> tuple[list | None, bytes | None, bytes | None]:
    """Invia video + detection al server e riceve video tracciato + CSV."""
    endpoint = f"{server_url}/api/track"
    try:
        data: dict = {
            "tracker_name":        tracker_name,
            "tracker_params_json": json.dumps(tracker_params),
            "target_class":        target_class,
            "extract_crops":       extract_crops,
        }
        # Preferisce detection_id (dati già in cache sul server)
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
            # Se la cache è scaduta, riprova inviando i dati completi
            if resp.status_code == 404 and detection_id and detections_data is not None:
                return server_track(
                    server_url, video_path, tracker_name, detections_data,
                    tracker_params, target_class=target_class,
                    extract_crops=extract_crops, output_dir=output_dir,
                    detection_id=None,
                )
            _show_error(resp, "/api/track")
            return None, None, None

        zf       = zipfile.ZipFile(io.BytesIO(resp.content))
        names    = zf.namelist()
        mp4_name = next((n for n in names if n.endswith(".mp4")), None)
        csv_name = next((n for n in names if n.endswith(".csv")), None)

        video_bytes = zf.read(mp4_name) if mp4_name else None
        csv_bytes   = zf.read(csv_name) if csv_name else None

        if extract_crops and output_dir:
            _extract_crops_from_zip(zf, output_dir)

        npz_name = next((n for n in names if n.endswith("_embeddings.npz")), None)
        if npz_name and embeddings_dir:
            (embeddings_dir / Path(npz_name).name).write_bytes(zf.read(npz_name))

        results = _parse_csv_bytes(csv_bytes)
        return results, video_bytes, csv_bytes

    except requests.exceptions.ConnectionError:
        st.error(f"Impossibile connettersi al server: `{endpoint}`")
        return None, None, None
    except Exception as exc:
        st.error(f"Errore `/api/track`:\n```\n{exc}\n```")
        return None, None, None


# ---------------------------------------------------------------------------
# Multi-tracking
# ---------------------------------------------------------------------------

def server_track_multi(
    server_url: str,
    video_path: Path,
    assignments: list,
    detections_data: list | None,
    *,
    extract_crops: bool = False,
    output_dir: Path | None = None,
    detection_id: str | None = None,
) -> tuple[bytes | None, dict | None]:
    """Tracking multi-classe: un tracker per ogni classe, video unico in uscita."""
    endpoint = f"{server_url}/api/track_multi"
    try:
        data: dict = {
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
            if resp.status_code == 404 and detection_id and detections_data is not None:
                return server_track_multi(
                    server_url, video_path, assignments, detections_data,
                    extract_crops=extract_crops, output_dir=output_dir, detection_id=None,
                )
            _show_error(resp, "/api/track_multi")
            return None, None

        zf        = zipfile.ZipFile(io.BytesIO(resp.content))
        names     = zf.namelist()
        mp4_name  = next((n for n in names if n.endswith(".mp4")), None)
        csv_names = [n for n in names if n.endswith(".csv")]

        video_bytes = zf.read(mp4_name) if mp4_name else None

        if extract_crops and output_dir:
            _extract_crops_from_zip(zf, output_dir)

        for npz_name in (n for n in names if n.endswith("_embeddings.npz")):
            if output_dir:
                tracker_name = Path(npz_name).name.split("_")[0]
                tracker_subdir = output_dir / tracker_name
                tracker_subdir.mkdir(parents=True, exist_ok=True)
                (tracker_subdir / Path(npz_name).name).write_bytes(zf.read(npz_name))

        csv_map = {n.replace("_tracking.csv", ""): zf.read(n) for n in csv_names}
        return video_bytes, csv_map

    except requests.exceptions.ConnectionError:
        st.error(f"Impossibile connettersi al server: `{endpoint}`")
        return None, None
    except Exception as exc:
        st.error(f"Errore `/api/track_multi`:\n```\n{exc}\n```")
        return None, None


# ---------------------------------------------------------------------------
# Analisi attributi (YOLO-E)
# ---------------------------------------------------------------------------

def server_analyze_crops(
    server_url: str,
    crops_dir: Path,
    model_name: str,
    attributes: list[str],
    conf_threshold: float = 0.25,
    top_n_crops: int = 5,
) -> dict | None:
    """Comprime i crop, li manda a /api/analyze_crops e restituisce i risultati per track_id."""
    endpoint = f"{server_url}/api/analyze_crops"

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for img_path in crops_dir.rglob("*.jpg"):
            arcname = img_path.relative_to(crops_dir)
            zf.write(img_path, arcname=str(arcname))
    buf.seek(0)

    if buf.getbuffer().nbytes == 0:
        st.error("Nessun crop trovato nella cartella selezionata.")
        return None

    try:
        resp = requests.post(
            endpoint,
            data={
                "model_name":      model_name,
                "attributes_json": json.dumps(attributes),
                "conf_threshold":  conf_threshold,
                "top_n_crops":     top_n_crops,
            },
            files={"crops_zip": ("crops.zip", buf, "application/zip")},
            timeout=600,
            proxies={"http": None, "https": None},
        )
        if resp.status_code != 200:
            _show_error(resp, "/api/analyze_crops")
            return None
        return resp.json().get("results", {})

    except requests.exceptions.ConnectionError:
        st.error(f"Impossibile connettersi al server: `{endpoint}`")
        return None
    except Exception as exc:
        st.error(f"Errore `/api/analyze_crops`:\n```\n{exc}\n```")
        return None


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _show_error(resp: requests.Response, endpoint: str) -> None:
    try:
        detail = resp.json().get("detail", resp.text)
    except Exception:
        detail = resp.text
    st.error(f"Errore `{endpoint}` [{resp.status_code}]:\n```\n{detail[:1000]}\n```")


def _extract_crops_from_zip(zf: zipfile.ZipFile, output_dir: Path) -> None:
    for member in zf.infolist():
        if member.filename.startswith("crops/") and len(member.filename) > 6:
            if member.filename.endswith("/"):
                continue
            rel_path    = Path(member.filename).relative_to("crops")
            target_path = output_dir / rel_path
            target_path.parent.mkdir(parents=True, exist_ok=True)
            target_path.write_bytes(zf.read(member))


def _parse_csv_bytes(csv_bytes: bytes | None) -> list[dict]:
    if not csv_bytes:
        return []
    reader = _csv.DictReader(
        line for line in csv_bytes.decode().splitlines() if not line.startswith("#")
    )
    return [
        {k: (int(v) if k in ("frame", "track_id") else float(v)) for k, v in row.items()}
        for row in reader
    ]