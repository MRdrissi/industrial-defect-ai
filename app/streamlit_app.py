"""Application web IndustrialDefect AI — US-06-02."""

from pathlib import Path
from typing import cast

import streamlit as st

from app.comparison import METHOD_LABELS, METHODS, available_versions, run_comparison
from app.comparison_view import display_comparison
from app.image_utils import decode_image, image_to_tensor
from app.report import build_report
from app.result_view import display_result

from src.inference.model_registry import get_model, load_registry
from src.inference.inference_contract import (
    InferenceRequest,
    run_inference,
)


ROOT = Path(__file__).resolve().parents[1]
REAL_REGISTRY_PATH = ROOT / "configs/real_model_registry.json"

APP_TITLE_SIZE_PX = 32
APP_TAGLINE_SIZE_PX = 20

INFERENCE_ERRORS = (
    ValueError,
    TypeError,
    KeyError,
    NotImplementedError,
    OSError,
    ImportError,
    RuntimeError,
)


def main():
    st.set_page_config(
        page_title="DefectVision",
        layout="wide",
    )

    st.html(
        f"""
        <style>
            [data-testid="stMainBlockContainer"] {{
                padding-top: 2rem;
            }}
            .st-key-app_title h1 {{
                font-size: {APP_TITLE_SIZE_PX}px;
            }}
            .st-key-app_title [data-testid="stCaptionContainer"] p {{
                font-size: {APP_TAGLINE_SIZE_PX}px;
            }}
        </style>
        """
    )

    with st.container(key="app_title"):
        st.title("DefectVision", anchor=False, text_alignment="left")
        st.caption("Analysez vos pièces, détectez leurs défauts.")

    # 1. Charger le registre des modèles réels
    try:
        registry = load_registry(REAL_REGISTRY_PATH)
    except (OSError, ValueError) as exc:
        st.error(f"Impossible de charger le registre : {exc}")
        st.stop()

    # 2. Sélectionner la catégorie
    categories = sorted({key[0] for key in registry})
    category = cast(str, st.selectbox("Catégorie", categories))

    # 3. Mode comparaison CAE / PatchCore (US-06-05)
    compare_mode = st.checkbox(
        "Comparer CAE et PatchCore sur la même image",
        value=False,
        key="compare_mode",
    )

    if compare_mode:
        # 4 bis. Une version par méthode
        try:
            available = available_versions(registry, category)
        except ValueError as exc:
            st.error(str(exc))
            st.stop()

        compare_versions = {
            name: cast(str, st.selectbox(
                f"Version {METHOD_LABELS[name]}",
                available[name],
            ))
            for name in METHODS
        }

    else:
        # 3. Sélectionner la méthode
        methods = sorted({
            key[1] for key in registry
            if key[0] == category
        })

        method = cast(str, st.selectbox(
            "Méthode",
            methods,
            format_func=lambda value: {
                "cae": "CAE",
                "patchcore": "PatchCore",
            }.get(value, value),
        ))

        # 4. Sélectionner la version
        versions = sorted({
            key[2] for key in registry
            if key[0] == category and key[1] == method
        })

        version = cast(str, st.selectbox("Version", versions))

    # 5. Importer l'image
    uploaded = st.file_uploader(
        "Importer une image",
        type=["png", "jpg", "jpeg"],
        accept_multiple_files=False,
    )

    image = None

    if uploaded is not None:
        try:
            image = decode_image(uploaded.getvalue())
            st.image(
                image,
                caption="Image importée",
                width="stretch",
            )
        except ValueError as exc:
            st.error(str(exc))
            st.stop()

    # 6. Lancer l'analyse
    if st.button("Analyser l'image", type="primary"):

        if image is None:
            st.warning("Importe d'abord une image.")
            return

        if compare_mode:
            try:
                comparison = run_comparison(
                    image_to_tensor(image),
                    category,
                    compare_versions,
                    registry,
                )
            except INFERENCE_ERRORS as exc:
                st.error(f"Erreur d'inférence : {exc}")
                return

            display_comparison(comparison, image)
            return

        try:
            request = InferenceRequest(
                category=category,
                method=method,
                version=version,
                image=image_to_tensor(image),
            )

            result = run_inference(request, registry)

        except INFERENCE_ERRORS as exc:
            st.error(f"Erreur d'inférence : {exc}")
            return

        # 7. Préparer le rapport d'inspection (US-06-04)
        report = build_report(
            result,
            get_model(registry, category, method, version),
            image_bytes=uploaded.getvalue(),
            image_name=uploaded.name,
            image_size=image.size,
        )

        # 8. Afficher les résultats
        display_result(result, image, report)


if __name__ == "__main__":
    main()
