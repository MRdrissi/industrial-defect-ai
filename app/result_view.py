"""Affichage des résultats d'inférence dans Streamlit.

US-06-02 / US-06-03 : décision, indicateurs, carte d'anomalie et détails.
US-06-04 (IDL-77)   : heatmap superposée à l'image, alignée sur sa taille
                      d'origine, et identification du modèle (version).
US-06-04 (IDL-78)   : téléchargement du rapport d'inspection JSON.
"""

import numpy as np
import streamlit as st
from PIL import Image

from app.overlay import make_overlay
from app.report import report_filename, report_to_json
from src.inference.inference_contract import InferenceResult

METHOD_LABELS = {"cae": "Convolutional AutoEncoder", "patchcore": "PatchCore"}


def model_label(result: InferenceResult) -> str:
    """Identification lisible du modèle utilisé."""
    method = METHOD_LABELS.get(result.method, result.method)
    return (
        f"Modèle : {result.category.capitalize()} · {method}"
    )


def decision_label(result: InferenceResult) -> str:
    """Décision affichée : NORMAL, SUSPECT ou NON CALIBRÉE."""
    return {
        "normal": "NORMAL",
        "suspect": "SUSPECT",
        "non_calibree": "NON CALIBRÉE",
    }.get(result.decision, result.decision.upper())


def _normalized_raw_map(result: InferenceResult) -> np.ndarray:
    """Carte brute seule (comportement US-06-02 / US-06-03), pour le détail technique."""
    raw_map = result.anomaly_map.detach().cpu().numpy()

    if result.simulated:
        return np.asarray(np.clip(raw_map * 2, 0.0, 1.0), dtype=np.float32)

    minimum, maximum = float(raw_map.min()), float(raw_map.max())
    if maximum > minimum:
        return np.asarray((raw_map - minimum) / (maximum - minimum), dtype=np.float32)
    return np.zeros_like(raw_map, dtype=np.float32)



def display_result(
        result: InferenceResult,
        image: Image.Image | None = None,
        report: dict | None = None,
        details_page=None,
) -> None:
    """Affiche uniquement la synthèse de l'inspection."""

    st.subheader("Résultat de l'inspection")

    score_col, threshold_col, latency_col, verdict_col = st.columns(4)

    with score_col:
        st.metric("Score brut", f"{result.score:.4f}")

    with threshold_col:
        st.metric(
            "Seuil",
            "Non calibré"
            if result.threshold is None
            else f"{result.threshold:.4f}",
        )

    with latency_col:
        st.metric("Latence", f"{result.latency_ms:.1f} ms")

    with verdict_col:
        st.caption("Verdict")

        if result.threshold is None or result.decision == "non_calibree":
            st.badge("NON CALIBRÉ", color="gray")
        elif result.decision == "normal":
            st.badge("NORMAL", color="green")
        elif result.decision == "suspect":
            st.badge("SUSPECT", color="red")

    if report is not None:
        _, download_col, details_col = st.columns(
            [5, 2.4, 1.6],
            vertical_alignment="center",
        )

        with download_col:
            st.download_button(
                "Télécharger le rapport",
                data=report_to_json(report),
                file_name=report_filename(report),
                mime="application/json",
                on_click="ignore",
                width="stretch",
            )

        with details_col:
            if st.button(
                    "Voir détails",
                    key="view_details",
                    type="secondary",
                    width="stretch",
                    disabled=details_page is None,
            ):
                st.switch_page(details_page)



def display_inspection_details(home_page) -> None:
    """Page dédiée à la visualisation et aux détails techniques."""

    if st.button("Retour à l'analyse"):
        st.switch_page(home_page)

    inspection = st.session_state.get("last_inspection")

    if inspection is None:
        st.info("Aucune inspection disponible.")
        return

    result = inspection["result"]
    image = inspection["image"]

    st.title("Détails de l'inspection")

    st.caption(model_label(result))

    _, overlay, _ = make_overlay(
        image,
        result.anomaly_map,
        result.threshold,
    )

    left, right = st.columns(2)

    with left:
        st.image(
            image,
            caption="Image analysée",
            width="stretch",
        )

    with right:
        st.image(
            overlay,
            caption="Heatmap superposée",
            width="stretch",
        )

    with st.expander("Détails techniques"):
        st.image(
            _normalized_raw_map(result),
            caption="Carte d'anomalie brute normalisée",
            width="stretch",
        )

        st.json(result.to_dict())
