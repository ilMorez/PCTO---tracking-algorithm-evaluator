"""
utils/ui_helpers.py
Componenti UI riutilizzabili per Streamlit.
"""

from __future__ import annotations

import streamlit as st


# ---------------------------------------------------------------------------
# Parametri tracker
# ---------------------------------------------------------------------------

def render_tracker_parameters(tracker_cls, tracker_key: str) -> dict:
    """Renderizza i widget Streamlit per i parametri esposti da un tracker.

    Legge ``tracker_cls.PARAMETER_SPECS`` e genera automaticamente
    number_input / checkbox / selectbox secondo il campo ``type`` di ogni spec.

    Args:
        tracker_cls:  Classe del tracker (deve avere l'attributo PARAMETER_SPECS).
        tracker_key:  Prefisso univoco per le chiavi dei widget Streamlit.

    Returns:
        Dizionario ``{nome_parametro: valore}`` con i valori scelti dall'utente.
    """
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

            value = _render_single_param(
                container, field_type, label, default, widget_key,
                min_value, max_value, step, spec,
            )
            tracker_kwargs[name] = value

    return tracker_kwargs


def _render_single_param(
    container,
    field_type: str,
    label: str,
    default,
    widget_key: str,
    min_value,
    max_value,
    step,
    spec: dict,
):
    """Crea il singolo widget appropriato per il tipo di parametro."""
    if field_type == "int":
        return container.number_input(
            label,
            min_value=int(min_value) if min_value is not None else None,
            max_value=int(max_value) if max_value is not None else None,
            value=int(default) if default is not None else 0,
            step=int(step) if step is not None else 1,
            key=widget_key,
        )

    if field_type == "float":
        return container.number_input(
            label,
            min_value=float(min_value) if min_value is not None else None,
            max_value=float(max_value) if max_value is not None else None,
            value=float(default) if default is not None else 0.0,
            step=float(step) if step is not None else 0.01,
            format="%.4f",
            key=widget_key,
        )

    if field_type == "bool":
        return container.checkbox(label, value=bool(default), key=widget_key)

    if field_type == "select":
        options = spec.get("options", [])
        if not options:
            raw = container.text_input(
                label,
                value="" if default is None else str(default),
                key=widget_key,
            )
            return raw.strip() or None
        index = options.index(default) if default in options else 0
        return container.selectbox(label, options, index=index, key=widget_key)

    # Fallback: text
    raw = container.text_input(
        label,
        value="" if default is None else str(default),
        key=widget_key,
        help="Lascia vuoto per passare None" if default is None else None,
    )
    return raw.strip() or None


# ---------------------------------------------------------------------------
# Metriche video
# ---------------------------------------------------------------------------

def render_video_metrics(cap, detections_data: list) -> None:
    """Mostra FPS, frame totali, risoluzione e conteggi per classe."""
    import cv2

    fps    = cap.get(cv2.CAP_PROP_FPS)
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    st.metric("Durata video",  f"{(frames / fps):.2f} s")
    st.metric("Frame Rate",    f"{fps:.2f} FPS")
    st.metric("Frame Totali",  f"{frames} frames")
    st.metric(
        "Dimensioni",
        f"{int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x"
        f"{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))} px",
    )
    cap.release()

    class_counts: dict[str, int] = {}
    for frame in detections_data:
        for det in frame["detections"]:
            cls = det[5] if len(det) >= 6 else "?"
            class_counts[cls] = class_counts.get(cls, 0) + 1

    if class_counts:
        st.markdown("**Detections per classe:**")
        for cls, cnt in class_counts.items():
            st.markdown(f"- **{cls}**: {cnt}")