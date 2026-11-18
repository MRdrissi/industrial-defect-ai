"""Synthèse et détails de comparaison, avec les composants de l'inspection."""

import streamlit as st
from PIL import Image

from app.comparison import METHODS, METHOD_LABELS, SCORE_WARNING, ComparisonResult
from app.comparison_report import comparison_report_filename, comparison_report_to_json
from app.overlay import make_overlay
from app.result_view import (
    display_details_header,
    display_result_card,
    display_report_actions,
    display_summary_title,
    display_technical_details,
    show_anomaly_maps,
)

COMPARISON_RESULT_TITLE_SIZE_PX = 24
COMPARISON_DETAILS_TITLE_SIZE_PX = 30


def display_comparison(
    comparison: ComparisonResult,
    image: Image.Image | None = None,
    report: dict | None = None,
    details_page=None,
) -> None:
    """Synthèse compacte ; chaque score reste associé au seuil de son modèle."""
    display_summary_title("Résultat de la comparaison", COMPARISON_RESULT_TITLE_SIZE_PX)
    with st.container(key="result_cards"):
        for column, method in zip(st.columns(2), METHODS):
            with column:
                display_result_card(comparison[method], METHOD_LABELS[method], compact=True)
    if report is not None:
        display_report_actions(
            comparison_report_to_json(report), comparison_report_filename(report), details_page,
        )


def show_comparison_anomaly_maps(comparison: ComparisonResult, image: Image.Image) -> None:
    overlays = []
    for method in METHODS:
        result = comparison[method]
        _, overlay, _ = make_overlay(image, result.anomaly_map, result.threshold)
        overlays.append((METHOD_LABELS[method], overlay))
    show_anomaly_maps(overlays)


def display_comparison_details(home_page) -> None:
    stored = st.session_state.get("last_comparison")
    if stored is None:
        st.info("Aucune comparaison disponible.")
        st.page_link(home_page, label="Retour à l'analyse", width="content")
        return
    comparison = stored["result"]
    if display_details_header(
        home_page, "Détails de la comparaison", COMPARISON_DETAILS_TITLE_SIZE_PX,
        "Voir cartes d'anomalie",
    ):
        show_comparison_anomaly_maps(comparison, stored["image"])
    for column, method in zip(st.columns(2), METHODS):
        with column:
            with st.container(border=True, key=f"comparison_details_{method}"):
                display_technical_details(comparison[method], METHOD_LABELS[method], compact=True)
    st.caption(SCORE_WARNING)
