"""Affichage des résultats d'inférence dans Streamlit."""

import numpy as np
import streamlit as st

from src.inference.inference_contract import InferenceResult


def display_result(result: InferenceResult) -> None:
    """Affiche le résultat d'une inspection simulée."""

    st.subheader("Résultat de l'inspection")

    # 1. Décision
    if result.decision == "normal":
        st.success("Décision simulée : NORMAL")
    else:
        st.warning("Décision simulée : SUSPECT")

    # 2. Indicateurs
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Score brut", f"{result.score:.4f}")

    with col2:
        st.metric("Seuil démo", f"{result.threshold:.2f}")

    with col3:
        st.metric("Latence", f"{result.latency_ms:.1f} ms")

    # 3. Carte d'anomalie simulée
    visual_map = np.clip(
        result.anomaly_map.numpy() * 2,
        0,
        1,
        )

    st.image(
        visual_map,
        caption="Carte d'écarts simulée (non validée)",
        width="stretch",
    )

    # 4. Métadonnées
    with st.expander("Détails techniques"):
        st.json(result.to_dict())

    # 5. Avertissement
    st.info(
        "Simulation uniquement : aucune véritable "
        "détection de défaut n'a été effectuée."
    )
