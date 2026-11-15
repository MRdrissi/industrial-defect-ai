"""Tests de l'écran de comparaison CAE / PatchCore — US-06-05 (IDL-79)."""

from io import BytesIO
from pathlib import Path

import pytest
from PIL import Image
from streamlit.testing.v1 import AppTest

from app.comparison import SCORE_WARNING

ROOT = Path(__file__).resolve().parents[2]
APP = ROOT / "app/streamlit_app.py"
CATEGORY_SIZES = {"bottle": 900, "hazelnut": 1024, "transistor": 1024}


def make_png(side, color=(128, 128, 128)):
    buffer = BytesIO()
    Image.new("RGB", (side, side), color).save(buffer, format="PNG")
    return buffer.getvalue()


def texts(elements):
    return " | ".join(str(element.value) for element in elements)


def compare_app():
    at = AppTest.from_file(str(APP)).run()
    at.checkbox(key="compare_mode").check().run()
    return at


def test_default_mode_is_unchanged():
    at = AppTest.from_file(str(APP)).run()
    assert not at.exception
    assert [box.label for box in at.selectbox] == ["Catégorie", "Méthode", "Version"]


def test_compare_mode_asks_one_version_per_method():
    at = compare_app()
    assert not at.exception
    assert [box.label for box in at.selectbox] == ["Catégorie", "Version CAE", "Version PatchCore"]


@pytest.mark.parametrize("category, side", CATEGORY_SIZES.items())
def test_side_by_side_comparison_for_each_category(category, side):
    at = compare_app()
    at.selectbox[0].set_value(category).run()
    at.file_uploader[0].upload(f"{category}.png", make_png(side), "image/png").run()
    at.button[0].click().run()

    captions = texts(at.caption)
    assert not at.exception
    assert "Comparaison CAE / PatchCore" in texts(at.subheader)
    assert f"Même image ({side} x {side} px)" in captions
    assert captions.count("Score brut") == 2            # un score par méthode
    assert "seuil CAE : 0.2000" in captions             # seuils affichés
    assert "seuil PatchCore : 0.2000" in captions
    assert len(at.success) == 2                          # deux décisions NORMAL
    assert "Les deux méthodes concluent : NORMAL." in texts(at.markdown)


def test_raw_scores_are_not_presented_as_comparable():
    at = compare_app()
    at.file_uploader[0].upload("img.png", make_png(64), "image/png").run()
    at.button[0].click().run()

    assert not at.exception
    assert SCORE_WARNING in texts(at.info)
    assert len(at.metric) == 0                           # pas d'indicateurs côte à côte
    assert "Ce score mesure : erreur de reconstruction" in texts(at.caption)
    assert "Ce score mesure : distance au patch normal" in texts(at.caption)


def test_suspect_decisions_in_comparison():
    at = compare_app()
    at.file_uploader[0].upload("black.png", make_png(64, (0, 0, 0)), "image/png").run()
    at.button[0].click().run()

    assert not at.exception
    assert len(at.warning) == 2
    assert "Les deux méthodes concluent : SUSPECT." in texts(at.markdown)


def _real_uncalibrated_comparison():
    """Script autonome : deux résultats réels sans seuil, cartes 256 x 256, image 1024."""
    import torch
    from PIL import Image

    from app.comparison import ComparisonResult
    from app.comparison_view import display_comparison
    from src.inference.inference_contract import InferenceResult

    def result(method, score, version):
        return InferenceResult(
            category="transistor", method=method, version=version, score=score,
            threshold=None, anomaly_map=torch.rand(256, 256), decision="non_calibree",
            latency_ms=10.0, simulated=False,
        )

    comparison = ComparisonResult(
        category="transistor",
        image_shape=(3, 1024, 1024),
        results={"cae": result("cae", 0.0251, "exp-001"),
                 "patchcore": result("patchcore", 40.13, "exp-001")},
    )
    display_comparison(comparison, Image.new("RGB", (1024, 1024)))


def test_real_uncalibrated_comparison():
    at = AppTest.from_function(_real_uncalibrated_comparison).run()

    captions = texts(at.caption)
    assert not at.exception
    assert "aucune conclusion commune" in texts(at.markdown)
    assert "seuil CAE : non calibré" in captions
    assert "seuil PatchCore : non calibré" in captions
    assert "Version exp-001 (modèle réel)" in captions
    assert texts(at.info).count("NON CALIBRÉE") == 2
