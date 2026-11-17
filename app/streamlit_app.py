"""Application web IndustrialDefect AI — US-06-02."""

from hashlib import sha256
from contextlib import contextmanager
import json
from pathlib import Path
from typing import cast

import streamlit as st

from app.comparison import METHODS, available_versions, run_comparison
from app.comparison_report import build_comparison_report
from app.comparison_view import display_comparison, display_comparison_details
from app.image_preview import image_preview_button
from app.image_utils import MAX_FILE_SIZE, decode_image, image_to_tensor
from app.report import build_report
from app.result_view import (
    display_result,
    display_inspection_details,
)

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


@contextmanager
def analysis_button_loader():
    """Anime le bouton uniquement pendant l'inférence, même si elle échoue."""
    indicator = st.empty()
    indicator.html("""
        <style>
            @keyframes analysis-button-spin {
                to { transform: rotate(360deg); }
            }
            .st-key-analyze_image button {
                pointer-events: none;
            }
            .st-key-analyze_image button::before {
                visibility: visible !important;
                animation: analysis-button-spin 0.75s linear infinite;
            }
        </style>
        <span role="status" aria-live="polite"
              style="position:absolute;width:1px;height:1px;overflow:hidden;clip-path:inset(50%)">
            Analyse en cours
        </span>
    """)
    try:
        yield
    finally:
        indicator.empty()


def clear_last_inspection():
    uploaded = st.session_state.get("inspection_image")
    image_digest = sha256(uploaded.getvalue()).hexdigest() if uploaded else None
    for key in ("last_inspection", "last_comparison"):
        inspection = st.session_state.get(key)
        if inspection is not None and inspection.get("image_digest") != image_digest:
            st.session_state.pop(key, None)


def render_image_uploader():
    """Garde l'importeur natif associé au point d'entrée, indépendamment de la page."""
    return st.file_uploader(
        "Importer une image",
        type=["png", "jpg", "jpeg"],
        accept_multiple_files=False,
        max_upload_size=MAX_FILE_SIZE // (1024 * 1024),
        key="inspection_image",
        label_visibility="collapsed",
        on_change=clear_last_inspection,
    )


def main():
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
            .st-key-compare_controls [data-testid="stCheckbox"] label > div:first-of-type {{
                position: relative;
                border-radius: 50%;
                box-sizing: border-box;
                padding: 0;
                background-color: transparent;
            }}
            .st-key-compare_controls [data-testid="stCheckbox"] label[data-selected] > div:first-of-type::after {{
                content: "";
                position: absolute;
                top: 50%;
                left: 50%;
                transform: translate(-50%, -50%);
                width: 8px;
                height: 8px;
                box-sizing: border-box;
                border: 4px solid;
                border-color: inherit;
                border-radius: 50%;
            }}
            .st-key-compare_controls [data-testid="stCheckbox"] label > div:first-of-type svg {{
                display: none;
            }}
            .st-key-method_selector [data-testid="stWidgetLabel"] {{
                color: inherit;
            }}
            .st-key-inspection_image button[aria-label="Add files"] {{
                display: none;
            }}
            .st-key-upload_panel {{
                background-color: color-mix(in srgb, currentColor 5%, transparent);
                border-radius: 0.5rem;
                padding: 1rem;
            }}
            .st-key-inspection_image [data-testid="stFileUploaderDropzone"] {{
                background-color: transparent;
                border: none;
                padding: 0;
                min-height: 2.5rem;
                align-items: center;
                pointer-events: none;
            }}
            .st-key-inspection_image [data-testid="stFileUploaderDropzone"] button {{
                pointer-events: auto;
            }}
            .st-key-inspection_image [data-testid="stFileUploaderDropzone"] > span button p {{
                font-size: 0;
            }}
            .st-key-inspection_image [data-testid="stFileUploaderDropzone"] > span button p::after {{
                content: "Import";
                font-size: 0.875rem;
            }}
            .st-key-analyze_image button:disabled,
            .st-key-analyze_image button:disabled:hover {{
                background-color: #f8c7ca;
                border-color: #f8c7ca;
                color: #8b343a;
                opacity: 1;
            }}
            .st-key-analyze_image button {{
                padding-inline: 1.25rem !important;
                gap: 0.75rem;
            }}
            .st-key-analyze_image button::before,
            .st-key-analyze_image button::after {{
                content: "";
                display: inline-block;
                flex-shrink: 0;
                box-sizing: border-box;
                width: 0.85rem;
                height: 0.85rem;
                border: 2px solid transparent;
                border-radius: 50%;
                visibility: hidden;
            }}
            .st-key-analyze_image button::before {{
                border-color: currentColor;
                border-right-color: transparent;
            }}
            .st-key-view_details button,
            .st-key-view_details a {{
                background-color: #2563eb !important;
                border-color: #2563eb !important;
                color: white !important;
            }}
            .st-key-view_details a,
            .st-key-view_details a * {{
                color: #ffffff !important;
            }}
            .st-key-view_details button:hover,
            .st-key-view_details a:hover {{
                background-color: #1d4ed8 !important;
                border-color: #1d4ed8 !important;
                color: white !important;
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

    # 2. Sélectionner la catégorie et la méthode sur la même ligne
    category_column, method_column = st.columns(2)
    categories = sorted({key[0] for key in registry})
    category = cast(str, category_column.selectbox(
        "Catégorie",
        categories,
        format_func=str.capitalize,
        key="category_selector",
        persist_state="session",
    ))

    # 3. Mode comparaison CAE / PatchCore (US-06-05)
    with st.container(
        key="compare_controls",
        horizontal=True,
        horizontal_alignment="right",
    ):
        compare_mode = st.checkbox(
            "Comparer CAE et PatchCore sur la même image",
            value=False,
            key="compare_mode",
            persist_state="session",
        )

    methods = sorted({
        key[1] for key in registry
        if key[0] == category
    })
    method = cast(str, method_column.selectbox(
        "Méthode",
        ["both"] if compare_mode else methods,
        format_func=lambda value: {
            "cae": "Convolutional AutoEncoder",
            "patchcore": "PatchCore",
            "both": "Les deux méthodes",
        }.get(value, value),
        disabled=compare_mode,
        key="method_selector",
        persist_state="session",
    ))

    if compare_mode:
        # 4 bis. Une version par méthode
        try:
            available = available_versions(registry, category)
            if any(len(versions) != 1 for versions in available.values()):
                raise ValueError("Une seule version par catégorie et méthode est attendue.")
        except ValueError as exc:
            st.error(str(exc))
            st.stop()

        compare_versions = {
            name: available[name][0]
            for name in METHODS
        }
        # Le registre fait partie de l'identité : seuil, checkpoint et configuration.
        comparison_selection = (
            category,
            tuple((name, compare_versions[name], json.dumps(
                get_model(registry, category, name, compare_versions[name]), sort_keys=True,
            )) for name in METHODS),
        )

    else:
        # 4. Utiliser l'unique version disponible
        versions = sorted({
            key[2] for key in registry
            if key[0] == category and key[1] == method
        })

        if len(versions) != 1:
            st.error("Une seule version par catégorie et méthode est attendue.")
            st.stop()
        version = versions[0]

    selection = comparison_selection if compare_mode else (category, method, version)
    state_key = "last_comparison" if compare_mode else "last_inspection"

    # 5. Importer et valider l'image avant d'autoriser l'analyse
    image = None
    image_error = None

    st.markdown("Importer votre image")
    with st.container(key="upload_panel"):
        upload_column, analyze_column = st.columns([4, 1], vertical_alignment="center")
        with upload_column.container(
            key="upload_controls",
            horizontal=True,
            horizontal_alignment="left",
            vertical_alignment="center",
        ):
            uploaded = render_image_uploader()
            image_digest = sha256(uploaded.getvalue()).hexdigest() if uploaded else None

            if uploaded is not None:
                try:
                    image = decode_image(uploaded.getvalue())
                except ValueError as exc:
                    image_error = str(exc)

                if image is not None:
                    with st.container(key="preview_control", width="content"):
                        image_preview_button(image, uploaded.name)

        inspection = st.session_state.get(state_key)
        already_analyzed = (
            inspection is not None
            and inspection.get("image_digest") == image_digest
            and inspection["selection"] == selection
        )
        with analyze_column.container(horizontal=True, horizontal_alignment="right"):
            analyze_clicked = st.button(
                "Analyser l'image",
                type="primary",
                disabled=image is None or already_analyzed,
                width="content",
                key="analyze_image",
            )

    if image_error is not None:
        st.error(image_error)

    # 6. Lancer l'analyse
    if analyze_clicked and image is not None:
        if compare_mode:
            try:
                with analysis_button_loader():
                    comparison = run_comparison(
                        image_to_tensor(image),
                        category,
                        compare_versions,
                        registry,
                        infer=run_inference,
                    )
                    report = build_comparison_report(
                        comparison, registry, uploaded.getvalue(), uploaded.name, image.size,
                    )
            except INFERENCE_ERRORS as exc:
                st.error(f"Erreur d'inférence : {exc}")
                return

            st.session_state["last_comparison"] = {
                "result": comparison,
                "image": image.copy(),
                "report": report,
                "selection": selection,
                "image_digest": image_digest,
            }
            st.rerun()

        try:
            with analysis_button_loader():
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


        # 8. Mémoriser le résultat pour les deux pages
        st.session_state["last_inspection"] = {
            "result": result,
            "image": image.copy(),
            "report": report,
            "selection": (category, method, version),
            "image_digest": image_digest,
        }
        st.rerun()

    return {
        "compare_mode": compare_mode,
        "selection": selection,
        "image_digest": image_digest,
    }


def show_inspection_summary():
    """Affiche la synthèse de la page Analyse sans relancer l'inférence."""
    if inspection_context is None:
        return
    compare_mode = inspection_context["compare_mode"]
    inspection = st.session_state.get("last_comparison" if compare_mode else "last_inspection")
    if (
        inspection is not None
        and inspection.get("image_digest") == inspection_context["image_digest"]
        and inspection["selection"] == inspection_context["selection"]
    ):
        display = display_comparison if compare_mode else display_result
        display(
            inspection["result"],
            inspection["image"],
            inspection["report"],
            comparison_details_page if compare_mode else details_page,
        )


if __name__ == "__main__":
    st.set_page_config(
        page_title="DefectVision",
        layout="wide",
        initial_sidebar_state="collapsed",
    )

    home_page = st.Page(
        show_inspection_summary,
        title="Analyse",
        default=True,
    )

    details_page = st.Page(
        lambda: display_inspection_details(home_page),
        title="Détails de l'inspection",
        url_path="details",
    )

    comparison_details_page = st.Page(
        lambda: display_comparison_details(home_page),
        title="Détails de la comparaison",
        url_path="comparison-details",
    )

    selected_page = st.navigation(
        [home_page, details_page, comparison_details_page],
        position="hidden",
    )
    # Les contrôles restent dans le cadre commun, au même emplacement DOM.
    # Sur Détails, ils sont masqués sans démonter l'importeur natif.
    controls_display = "flex" if selected_page.url_path == home_page.url_path else "none"
    st.html(
        f"<style>.st-key-inspection_controls {{ display: {controls_display}; }}</style>"
    )
    with st.container(key="inspection_controls"):
        inspection_context = main()
    selected_page.run()
