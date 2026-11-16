"""Bouton Lucide et aperçu de l'image importée dans une modale Streamlit."""

from base64 import b64encode
from pathlib import Path

import streamlit as st
from PIL import Image


# Source : https://github.com/lucide-icons/lucide/blob/main/icons/eye.svg
# Licence : assets/icons/LICENSE.
_EYE_SVG = (Path(__file__).parent / "assets/icons/eye.svg").read_text(encoding="utf-8")
_EYE_DATA = b64encode(
    _EYE_SVG.replace('stroke="currentColor"', 'stroke="#64748b"').encode("utf-8")
).decode("ascii")
_PREVIEW_LABEL = f"![Afficher l’aperçu](data:image/svg+xml;base64,{_EYE_DATA})"
PREVIEW_ICON_SIZE_PX = 42


@st.dialog("Aperçu de l’image", width="medium", dismissible=True)
def _show_image_preview(image: Image.Image, image_name: str) -> None:
    """La croix native en haut à droite ferme la modale."""
    st.html(
        """
        <style>
            [data-testid="stDialog"]:has(.st-key-image_preview_content) > div {
                transform: translateY(46px);
            }
            .st-key-image_preview_content [data-testid="stImage"] {
                display: flex;
                flex-direction: column;
                align-items: center;
            }
            .st-key-image_preview_content img {
                width: auto;
                max-width: 100%;
                max-height: min(520px, 60dvh, calc(100dvh - 12rem));
                object-fit: contain;
            }
            .st-key-image_preview_content [data-testid="stImageCaption"] {
                max-width: 100%;
                white-space: nowrap;
                overflow: hidden;
                text-overflow: ellipsis;
            }
        </style>
        """
    )
    with st.container(key="image_preview_content"):
        st.image(image, caption=image_name, width="stretch")


@st.fragment
def image_preview_button(image: Image.Image, image_name: str) -> None:
    """Ouvre l'aperçu uniquement lorsque l'utilisateur clique sur l'œil."""
    st.html(
        f"""
        <style>
            .st-key-upload_controls {{
                justify-content: flex-start;
                gap: 6px;
            }}
            .st-key-inspection_image:has([data-testid="stFileChip"]) {{
                flex: 0 1 auto;
                width: max-content;
                max-width: calc(100% - 40px);
                min-width: 0;
                margin: 0;
            }}
            .st-key-inspection_image:has([data-testid="stFileChip"]) [data-testid="stFileUploaderDropzone"] {{
                flex-direction: row;
                justify-content: flex-start;
            }}
            .st-key-preview_control {{
                flex: 0 0 auto;
                margin: 0;
            }}
            .st-key-preview_image button {{
                padding: 0;
            }}
            .st-key-preview_image button img {{
                width: {PREVIEW_ICON_SIZE_PX}px;
                height: {PREVIEW_ICON_SIZE_PX}px;
                max-height: {PREVIEW_ICON_SIZE_PX}px;
                transition: filter 150ms ease;
            }}
            .st-key-preview_image button:hover img,
            .st-key-preview_image button:focus-visible img {{
                filter: brightness(0.55);
            }}
        </style>
        """
    )
    if st.button(
        _PREVIEW_LABEL,
        key="preview_image",
        type="tertiary",
        width="content",
    ):
        _show_image_preview(image, image_name)
