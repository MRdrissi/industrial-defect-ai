"""Tests de l'écran de résultat — US-06-04 (IDL-77).

Vérifie que l'écran affiche la heatmap superposée, le score, le seuil, la
décision et la version, en mode simulé (pour les trois catégories) et pour
un résultat de modèle réel non calibré (carte 256 x 256, seuil absent).
"""

from io import BytesIO
from pathlib import Path

import pytest
import torch
from PIL import Image
from streamlit.testing.v1 import AppTest

from app.result_view import decision_label, model_label
from src.inference.inference_contract import InferenceResult

ROOT = Path(__file__).resolve().parents[2]
APP = ROOT / "app/streamlit_app.py"

# Taille d'origine des images MVTec AD (US-02-01)
CATEGORY_SIZES = {"bottle": 900, "hazelnut": 1024, "transistor": 1024}


def make_png(side, color=(128, 128, 128)):
    buffer = BytesIO()
    Image.new("RGB", (side, side), color).save(buffer, format="PNG")
    return buffer.getvalue()


def make_result(**overrides):
    values = dict(
        category="transistor",
        method="patchcore",
        version="exp-001",
        score=40.13,
        threshold=None,
        anomaly_map=torch.zeros(256, 256),
        decision="non_calibree",
        latency_ms=12.5,
        simulated=False,
    )
    values.update(overrides)
    return InferenceResult(**values)


def captions(at):
    return " | ".join(element.value for element in at.caption)


def test_model_label_shows_category_method_and_version():
    label = model_label(make_result())
    assert "Transistor" in label
    assert "PatchCore" in label
    assert "version exp-001" in label
    assert "modèle réel" in label

    assert "simulation" in model_label(make_result(simulated=True, method="cae"))


@pytest.mark.parametrize(
    "decision, expected",
    [("normal", "NORMAL"), ("suspect", "SUSPECT"), ("non_calibree", "NON CALIBRÉE")],
)
def test_decision_label(decision, expected):
    assert decision_label(make_result(decision=decision)) == expected


@pytest.mark.parametrize("category, side", CATEGORY_SIZES.items())
def test_simulated_result_screen_for_each_category(category, side):
    at = AppTest.from_file(str(APP)).run()
    at.selectbox[0].set_value(category).run()
    at.file_uploader[0].upload(f"{category}.png", make_png(side), "image/png").run()
    at.button[0].click().run()

    text = captions(at)
    assert not at.exception
    assert len(at.metric) == 3                       # score, seuil, latence
    assert f"Modèle : {category.capitalize()}" in text
    assert "version demo-v1" in text                 # version visible
    assert "décision : NORMAL" in text               # décision visible
    assert "Couleurs relatives au seuil" in text     # heatmap superposée affichée


def test_suspect_decision_is_shown_with_overlay():
    at = AppTest.from_file(str(APP)).run()
    at.file_uploader[0].upload("black.png", make_png(64, (0, 0, 0)), "image/png").run()
    at.button[0].click().run()

    assert not at.exception
    assert "décision : SUSPECT" in captions(at)
    assert "Couleurs relatives au seuil" in captions(at)


def _real_uncalibrated_screen():
    """Script Streamlit autonome : résultat réel non calibré, carte 256 x 256, image 1024."""
    import torch
    from PIL import Image

    from app.result_view import display_result
    from src.inference.inference_contract import InferenceResult

    anomaly_map = torch.zeros(256, 256)
    anomaly_map[60:70, 180:200] = 1.0
    result = InferenceResult(
        category="transistor",
        method="patchcore",
        version="exp-001",
        score=40.13,
        threshold=None,
        anomaly_map=anomaly_map,
        decision="non_calibree",
        latency_ms=12.5,
        simulated=False,
    )
    display_result(result, Image.new("RGB", (1024, 1024), (90, 90, 90)))


def test_real_uncalibrated_result_uses_relative_colors():
    at = AppTest.from_function(_real_uncalibrated_screen).run()

    text = captions(at)
    assert not at.exception
    assert len(at.info) == 1                          # décision indisponible
    assert at.metric[1].value == "Non calibré"        # seuil absent affiché
    assert "version exp-001 (modèle réel)" in text
    assert "décision : NON CALIBRÉE" in text
    assert "couleurs relatives à cette image" in text
