"""Affichage des résultats d'inférence dans Streamlit.

US-06-02 / US-06-03 : décision, indicateurs, carte d'anomalie et détails.
US-06-04 (IDL-77)   : heatmap superposée à l'image, alignée sur sa taille
                      d'origine, et identification du modèle (version).
US-06-04 (IDL-78)   : téléchargement du rapport d'inspection JSON.
"""

from base64 import b64encode
from pathlib import Path
import streamlit as st
from PIL import Image

from app.overlay import make_overlay
from app.report import report_filename, report_to_json
from src.inference.inference_contract import InferenceResult

METHOD_LABELS = {"cae": "Convolutional AutoEncoder", "patchcore": "PatchCore"}

DETAILS_TITLE_SIZE_PX = 30

_BACK_SVG = (
        Path(__file__).parent / "assets/icons/chevron-left.svg"
).read_text(encoding="utf-8")

_BACK_ICON_DATA = b64encode(
    _BACK_SVG.replace(
        'stroke="currentColor"',
        'stroke="#262626"',
    ).encode("utf-8")
).decode("ascii")

_BACK_LABEL = (
    f"![Retour à l'analyse]"
    f"(data:image/svg+xml;base64,{_BACK_ICON_DATA})"
)


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


@st.dialog("Heatmap superposée", width="medium", dismissible=True)
def show_anomaly_map(overlay: Image.Image) -> None:
    """Affiche la heatmap superposée dans une fenêtre modale."""

    st.html(
        """
        <style>
            .st-key-anomaly_modal [data-testid="stImage"] {
                display: flex;
                justify-content: center;
                align-items: center;
            }

            .st-key-anomaly_modal img {
                width: auto;
                max-width: 100%;
                max-height: min(520px, 60dvh);
                object-fit: contain;
            }
        </style>
        """
    )

    with st.container(key="anomaly_modal"):
        st.image(overlay, width="stretch")


def display_inspection_details(home_page) -> None:
    """Page dédiée aux détails d'une inspection."""

    st.html(
        f"""
        <style>
            [data-testid="stMainBlockContainer"] {{
                padding-top: 2rem;
            }}
            
            .st-key-details_header {{
                padding-top: 2rem;
                padding-bottom: 0.5rem;
                gap: 0rem !important;
                align-items: center !important;
            }}

            .st-key-details_header h1 {{
                font-size: {DETAILS_TITLE_SIZE_PX}px !important;
                line-height: 1.2;
                padding: 0.25rem 0;
                margin: 0;
            }}

            .st-key-details_back button {{
                border: none;
                background: transparent;
                padding: 0 !important;
                width: 32px;
                height: 32px;
                min-height: 32px;
                display: flex;
                align-items: center;
                justify-content: center;
            }}

            .st-key-details_back button img {{
                width: 26px;
                height: 26px;
                max-height: 26px;
            }}
        </style>
        """
    )

    # 1. En-tête : chevron de retour + titre
    with st.container(
            key="details_header",
            horizontal=True,
            horizontal_alignment="left",
            vertical_alignment="center",
            gap="small",
    ):
        if st.button(
                _BACK_LABEL,
                key="details_back",
                type="tertiary",
                width="content",
        ):
            st.switch_page(home_page)

        st.title("Détails de l'inspection", anchor=False)

    # 2. Récupérer l'inspection
    inspection = st.session_state.get("last_inspection")

    if inspection is None:
        st.info("Aucune inspection disponible.")
        return

    result = inspection["result"]
    image = inspection["image"]

    st.caption(model_label(result))

    # 3. Bouton heatmap et détails techniques sur une ligne
    heatmap_col, technical_col = st.columns(
        2,
        vertical_alignment="center",
        gap="medium",
    )

    with heatmap_col:
        if st.button(
                "Voir carte d'anomalie",
                key="view_anomaly_map",
                width="stretch",
        ):
            _, overlay, _ = make_overlay(
                image,
                result.anomaly_map,
                result.threshold,
            )
            show_anomaly_map(overlay)

    with technical_col:
        with st.expander("Détails techniques"):
            st.json(result.to_dict())

