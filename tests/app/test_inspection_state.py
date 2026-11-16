"""Régressions de navigation et d'état, sans validation visuelle ni inférence réelle."""

from io import BytesIO
from pathlib import Path

import pytest
import torch
from PIL import Image
from streamlit.testing.v1 import AppTest

from src.inference.inference_contract import InferenceResult


APP = Path(__file__).resolve().parents[2] / "app/streamlit_app.py"


def png(color="gray"):
    buffer = BytesIO()
    Image.new("RGB", (8, 8), color).save(buffer, format="PNG")
    return buffer.getvalue()


def follow_page_link(app):
    # AppTest.switch_page ne gère que les pages basées sur des fichiers.
    # Simuler ici le lien natif vers nos pages définies par des fonctions.
    app._page_hash = app.get("page_link")[0].proto.page_script_hash
    app.run()


def element_path(node, target):
    """Emplacement structurel, pour détecter le déplacement/démontage du widget."""
    if node is target:
        return ()
    for index, child in getattr(node, "children", {}).items():
        path = element_path(child, target)
        if path is not None:
            return (index, *path)
    return None


@pytest.fixture
def inspection_app(monkeypatch):
    calls = []

    def infer(request, registry):
        calls.append(request)
        return InferenceResult(
            category=request.category,
            method=request.method,
            version=request.version,
            score=0.1,
            threshold=None,
            anomaly_map=torch.zeros(8, 8),
            decision="non_calibree",
            latency_ms=1.0,
            simulated=False,
        )

    monkeypatch.setattr("src.inference.inference_contract.run_inference", infer)
    app = AppTest.from_file(str(APP), default_timeout=20).run()
    assert not app.exception
    return app, calls


def test_image_selection_and_result_survive_details_round_trip(inspection_app):
    app, calls = inspection_app
    app.selectbox(key="category_selector").set_value("transistor").run()
    app.selectbox(key="method_selector").set_value("patchcore").run()
    content = png()
    app.file_uploader[0].upload("piece.png", content, "image/png").run()
    original_uploader_id = app.file_uploader[0].id
    original_uploader_path = element_path(app._tree, app.file_uploader[0])
    assert original_uploader_path is not None
    app.button(key="analyze_image").click().run()

    assert not app.exception
    assert len(calls) == 1
    assert app.button(key="analyze_image").disabled
    assert app.subheader[0].proto.hide_anchor
    assert [metric.label for metric in app.metric] == [
        "Score brut", "Seuil", "Latence", "Verdict",
    ]
    original_report = app.session_state["last_inspection"]["report"]

    for _ in range(3):
        follow_page_link(app)
        assert not app.exception
        assert app.file_uploader[0].value.getvalue() == content
        assert element_path(app._tree, app.file_uploader[0]) == original_uploader_path
        assert app.session_state["last_inspection"]["report"] == original_report
        follow_page_link(app)
        assert not app.exception
        assert element_path(app._tree, app.file_uploader[0]) == original_uploader_path
        assert len(app.metric) == 4
        assert app.session_state["last_inspection"]["report"] == original_report

    assert not app.exception
    assert app.file_uploader[0].id == original_uploader_id
    assert app.file_uploader[0].value.getvalue() == content
    assert app.file_uploader[0].value.name == "piece.png"
    assert app.selectbox(key="category_selector").value == "transistor"
    assert app.selectbox(key="method_selector").value == "patchcore"
    assert app.button(key="preview_image") is not None
    assert app.button(key="analyze_image").disabled
    assert len(app.metric) == 4
    assert len(calls) == 1

    app.button(key="preview_image").click().run()
    assert not app.exception
    assert len(calls) == 1
    assert app.button(key="analyze_image").disabled

    app.file_uploader[0].clear().run()
    assert not app.exception
    assert app.button(key="analyze_image").disabled
    assert len(app.metric) == 0
    assert app.file_uploader[0].value is None
    app.file_uploader[0].upload("new.png", png("white"), "image/png").run()
    assert not app.exception
    assert not app.button(key="analyze_image").disabled


def test_analysis_reenabled_for_changed_model_or_image(inspection_app):
    app, calls = inspection_app
    assert app.button(key="analyze_image").disabled
    app.file_uploader[0].upload("piece.png", png(), "image/png").run()
    assert not app.button(key="analyze_image").disabled
    app.button(key="analyze_image").click().run()
    assert app.button(key="analyze_image").disabled

    app.selectbox(key="method_selector").set_value("patchcore").run()
    assert not app.button(key="analyze_image").disabled
    assert len(app.metric) == 0
    app.button(key="analyze_image").click().run()
    assert app.button(key="analyze_image").disabled

    app.file_uploader[0].upload("piece.png", png("white"), "image/png").run()
    assert not app.button(key="analyze_image").disabled
    assert len(app.metric) == 0
    app.button(key="analyze_image").click().run()
    assert app.button(key="analyze_image").disabled
    assert len(calls) == 3

    app.selectbox(key="category_selector").set_value("hazelnut").run()
    assert not app.button(key="analyze_image").disabled
    app.file_uploader[0].clear().run()
    assert not app.exception
    assert app.button(key="analyze_image").disabled
    assert len(app.metric) == 0


def test_same_image_content_does_not_enable_another_analysis(inspection_app):
    app, calls = inspection_app
    content = png()
    app.file_uploader[0].upload("piece.png", content, "image/png").run()
    app.button(key="analyze_image").click().run()
    app.file_uploader[0].upload("renamed.png", content, "image/png").run()
    assert not app.exception
    assert app.button(key="analyze_image").disabled
    assert len(app.metric) == 4
    assert len(calls) == 1


def test_failed_analysis_can_be_retried(inspection_app, monkeypatch):
    app, calls = inspection_app

    def fail(request, registry):
        raise RuntimeError("Échec temporaire du modèle")

    monkeypatch.setattr("src.inference.inference_contract.run_inference", fail)
    app.file_uploader[0].upload("piece.png", png(), "image/png").run()
    app.button(key="analyze_image").click().run()
    assert not app.exception
    assert len(app.error) == 1
    assert not app.button(key="analyze_image").disabled
    assert len(app.metric) == 0


def test_invalid_image_does_not_enable_analysis(inspection_app):
    app, calls = inspection_app
    app.file_uploader[0].upload("invalid.png", b"invalid image", "image/png").run()
    assert not app.exception
    assert len(app.error) == 1
    assert app.button(key="analyze_image").disabled
    assert not calls


def test_native_remove_allows_reimporting_the_same_image(inspection_app):
    app, calls = inspection_app
    content = png()
    app.file_uploader[0].upload("piece.png", content, "image/png").run()
    app.file_uploader[0].clear().run()
    assert not app.exception
    assert app.file_uploader[0].value is None
    assert app.button(key="analyze_image").disabled

    app.file_uploader[0].upload("piece.png", content, "image/png").run()
    assert not app.exception
    assert not app.button(key="analyze_image").disabled
    assert app.file_uploader[0].value.getvalue() == content
    assert not calls
