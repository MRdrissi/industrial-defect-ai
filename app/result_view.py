"""Affichage des résultats d'inférence dans Streamlit."""

import numpy as np
import streamlit as st

from src.inference.inference_contract import InferenceResult


def display_result(result: InferenceResult) -> None:
    """Affiche le résultat d'une inspection simulée."""

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


    # 2. Indicateurs
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


    # 3. Carte d'anomalie
    raw_map = result.anomaly_map.detach().cpu().numpy()

    visual_map: np.ndarray

    if result.simulated:
        visual_map = np.asarray(
            np.clip(raw_map * 2, 0.0, 1.0),
            dtype=np.float32,
        )
    else:
        minimum = float(raw_map.min())
        maximum = float(raw_map.max())

        if maximum > minimum:
            visual_map = np.asarray(
                (raw_map - minimum) / (maximum - minimum),
                dtype=np.float32,
                )
        else:
            visual_map = np.zeros_like(
                raw_map,
                dtype=np.float32,
            )

    st.image(
        visual_map,
        caption=(
            "Carte d'écarts simulée"
            if result.simulated
            else "Carte d'anomalie réelle (échelle visuelle normalisée)"
        ),
        width="stretch",
    )


    # 4. Métadonnées
    with st.expander("Détails techniques"):
        st.json(result.to_dict())


    # 5. Avertissement
    if result.simulated:
        st.info(
            "Simulation uniquement : aucune véritable "
            "détection de défaut n'a été effectuée."
        )
    else:
        st.caption(
            "Score brut non calibré : aucune classification "
            "NORMAL/SUSPECT n'est établie."
        )
