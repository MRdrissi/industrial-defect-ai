"""Affichage des résultats d'inférence dans Streamlit.

US-06-02 / US-06-03 : décision, indicateurs, carte d'anomalie et détails.
US-06-04 (IDL-77)   : heatmap superposée à l'image, alignée sur sa taille
                      d'origine, et identification du modèle (version).
"""

import numpy as np
import streamlit as st
from PIL import Image

from app.overlay import make_overlay
from src.inference.inference_contract import InferenceResult

METHOD_LABELS = {"cae": "CAE", "patchcore": "PatchCore"}

SCALE_CAPTIONS = {
    "threshold": "Couleurs relatives au seuil : vert = niveau du seuil, rouge = au-dessus.",
    "relative": "Seuil non calibré : couleurs relatives à cette image (zone la plus forte en rouge).",
}


def model_label(result: InferenceResult) -> str:
    """Identification lisible du modèle utilisé."""
    method = METHOD_LABELS.get(result.method, result.method)
    mode = "simulation" if result.simulated else "modèle réel"
    return (
        f"Modèle : {result.category.capitalize()} · {method} · "
        f"version {result.version} ({mode})"
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


def display_result(result: InferenceResult, image: Image.Image | None = None) -> None:
    """Affiche le résultat d'une inspection (simulée ou réelle)."""

    st.subheader("Résultat de l'inspection")

    # 1. Décision
    if result.decision == "non_calibree":
        st.info(
            "Inférence réelle effectuée. "
            "Décision indisponible : seuil non calibré."
        )
    elif result.decision == "normal":
        st.success("Décision : NORMAL")
    else:
        st.warning("Décision : SUSPECT")

    # 2. Indicateurs et identification du modèle
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Score brut", f"{result.score:.4f}")

    with col2:
        st.metric(
            "Seuil",
            "Non calibré"
            if result.threshold is None
            else f"{result.threshold:.4f}",
        )

    with col3:
        st.metric("Latence", f"{result.latency_ms:.1f} ms")

    st.caption(f"{model_label(result)} · décision : {decision_label(result)}")

    # 3. Heatmap superposée à l'image (taille d'origine)
    if image is not None:
        _, overlay, scale = make_overlay(image, result.anomaly_map, result.threshold)

        left, right = st.columns(2)
        with left:
            st.image(image, caption="Image analysée", width="stretch")
        with right:
            st.image(overlay, caption="Heatmap superposée", width="stretch")

        st.caption(SCALE_CAPTIONS[scale])

    # 4. Carte brute et métadonnées
    with st.expander("Détails techniques"):
        st.image(
            _normalized_raw_map(result),
            caption=(
                "Carte d'écarts simulée"
                if result.simulated
                else "Carte d'anomalie réelle (échelle visuelle normalisée)"
            ),
            width="stretch",
        )
        st.json(result.to_dict())

    # 5. Avertissement
    if result.simulated:
        st.info(
            "Simulation uniquement : aucune véritable "
            "détection de défaut n'a été effectuée."
        )
    elif result.threshold is None:
        st.caption(
            "Score brut non calibré : aucune classification "
            "NORMAL/SUSPECT n'est établie."
        )
