"""Affichage du comparatif CAE / PatchCore — US-06-05 (IDL-79).

Les deux résultats sont présentés côte à côte (heatmap, décision, seuil).
Les scores bruts sont affichés en texte, chacun avec son propre seuil et ce
qu'il mesure, sans indicateur numérique côte à côte qui inviterait à les
comparer directement (critère 3).
"""

import streamlit as st
from PIL import Image

from app.comparison import (
    DECISION_LABELS,
    METHOD_LABELS,
    METHODS,
    SCORE_MEANINGS,
    SCORE_WARNING,
    ComparisonResult,
    decisions_summary,
)
from app.overlay import make_overlay
from src.inference.inference_contract import InferenceResult


def _threshold_text(result: InferenceResult) -> str:
    return "non calibré" if result.threshold is None else f"{result.threshold:.4f}"


def _decision_box(result: InferenceResult) -> None:
    label = DECISION_LABELS.get(result.decision, result.decision.upper())
    if result.decision == "normal":
        st.success(f"Décision : {label}")
    elif result.decision == "suspect":
        st.warning(f"Décision : {label}")
    else:
        st.info(f"Décision : {label} (seuil absent)")


def _method_column(result: InferenceResult, image: Image.Image) -> None:
    method = result.method
    mode = "simulation" if result.simulated else "modèle réel"
    st.markdown(f"#### {METHOD_LABELS[method]}")
    st.caption(f"Version {result.version} ({mode})")

    _decision_box(result)

    _, overlay, _ = make_overlay(image, result.anomaly_map, result.threshold)
    st.image(overlay, caption=f"Heatmap {METHOD_LABELS[method]}", width="stretch")

    st.caption(
        f"Score brut : {result.score:.4f} · seuil {METHOD_LABELS[method]} : "
        f"{_threshold_text(result)}"
    )
    st.caption(f"Ce score mesure : {SCORE_MEANINGS[method]}.")


def display_comparison(comparison: ComparisonResult, image: Image.Image) -> None:
    """Affiche CAE et PatchCore côte à côte pour la même image."""

    st.subheader("Comparaison CAE / PatchCore")
    st.caption(
        f"Même image ({image.width} x {image.height} px) analysée par les deux méthodes · "
        f"catégorie {comparison.category.capitalize()}"
    )

    st.markdown(f"**{decisions_summary(comparison)}**")

    columns = st.columns(2)
    for column, method in zip(columns, METHODS):
        with column:
            _method_column(comparison[method], image)

    st.info(SCORE_WARNING)
    st.caption(
        "Couleurs : relatives au seuil de chaque méthode s'il existe, "
        "sinon relatives à l'image. Les deux heatmaps n'ont pas la même échelle."
    )

    with st.expander("Détails techniques"):
        st.json({method: comparison[method].to_dict() for method in METHODS})
