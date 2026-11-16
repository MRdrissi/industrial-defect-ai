"""Régressions de comparaison : synthèse, détails, modale et persistance native."""

from copy import deepcopy
from dataclasses import replace
from io import BytesIO

import pytest
from PIL import Image

from app.comparison import METHODS, SCORE_WARNING
from app.overlay import make_overlay
from src.inference.model_registry import load_registry


def png(color="gray"):
    buffer = BytesIO()
    Image.new("RGB", (8, 8), color).save(buffer, format="PNG")
    return buffer.getvalue()


def follow_link(app):
    # Pages par fonctions : AppTest.switch_page ne sait gérer que les fichiers.
    app._page_hash = app.get("page_link")[0].proto.page_script_hash
    app.run()


def element_path(node, target):
    if node is target:
        return ()
    for index, child in getattr(node, "children", {}).items():
        path = element_path(child, target)
        if path is not None:
            return (index, *path)
    return None


def analyze(app):
    app.checkbox(key="compare_mode").check().run()
    app.file_uploader[0].upload("piece.png", png(), "image/png").run()
    app.button(key="analyze_image").click().run()
    assert not app.exception


@pytest.mark.parametrize("category", ["bottle", "hazelnut", "transistor"])
def test_compact_summary(inference_app, category):
    app, calls = inference_app
    app.selectbox(key="category_selector").set_value(category).run()
    analyze(app)
    assert len(calls) == 2
    assert [request.method for request in calls] == list(METHODS)
    assert [box.label for box in app.selectbox] == ["Catégorie", "Méthode"]
    assert app.selectbox(key="method_selector").disabled
    assert app.selectbox(key="method_selector").options == ["Les deux méthodes"]
    assert app.subheader[0].value == "Résultat de la comparaison"
    assert app.subheader[0].proto.hide_anchor
    assert [metric.label for metric in app.metric] == ["Score brut", "Seuil", "Latence", "Verdict"] * 2
    assert app.metric[3].value == ":green-badge[NORMAL]"
    assert app.metric[7].value == ":red-badge[SUSPECT]"
    assert len(app.download_button) == 1
    assert app.get("page_link")[0].label == "Voir détails"
    assert not app.get("image")
    assert not app.info
    assert not app.json
    assert not app.expander
    assert app.button(key="analyze_image").disabled
    report = app.session_state["last_comparison"]["report"]
    assert set(report["reports"]) == set(METHODS)
    assert report["category"] == category


def test_round_trips_preserve_native_upload_preview_and_results(inference_app):
    app, calls = inference_app
    analyze(app)
    uploader_id = app.file_uploader[0].id
    path = element_path(app._tree, app.file_uploader[0])
    report = deepcopy(app.session_state["last_comparison"]["report"])
    image = app.session_state["last_comparison"]["image"].tobytes()
    for _ in range(3):
        follow_link(app)
        assert not app.exception
        assert app.title[-1].value == "Détails de la comparaison"
        assert app.title[-1].proto.hide_anchor
        assert element_path(app._tree, app.file_uploader[0]) == path
        assert not app.get("image")
        assert not app.json
        assert not app.expander
        assert [s.value for s in app.subheader] == ["CAE", "PatchCore"]
        text = " ".join(m.value for m in app.markdown)
        for field in ("Catégorie", "Méthode", "Version", "Score brut", "Seuil", "Latence"):
            assert text.count(f"**{field} :**") == 2
        assert SCORE_WARNING in [c.value for c in app.caption]
        assert "simulation" not in text.lower()
        follow_link(app)
        app.run()
        assert not app.exception
        assert len(app.metric) == 8
        assert app.file_uploader[0].id == uploader_id
        assert element_path(app._tree, app.file_uploader[0]) == path
        assert app.file_uploader[0].value.getvalue() == png()
        assert app.file_uploader[0].value.name == "piece.png"
        assert app.button(key="preview_image") is not None
        assert app.button(key="analyze_image").disabled
        assert app.session_state["last_comparison"]["report"] == report
        assert app.session_state["last_comparison"]["image"].tobytes() == image
    app.button(key="preview_image").click().run()
    assert not app.exception
    assert len(calls) == 2


@pytest.mark.parametrize("uncalibrated", [False, True])
def test_modal_overlays_use_separate_maps_and_scales(inference_app, monkeypatch, uncalibrated):
    app, calls = inference_app
    analyze(app)
    stored = app.session_state["last_comparison"]
    if uncalibrated:
        stored["result"].results["cae"] = replace(
            stored["result"]["cae"], threshold=None, decision="non_calibree",
        )
    overlays = []

    def overlay(image, anomaly_map, threshold):
        overlays.append((anomaly_map, threshold))
        return make_overlay(image, anomaly_map, threshold)

    monkeypatch.setattr("app.comparison_view.make_overlay", overlay)
    follow_link(app)
    assert not overlays
    app.button(key="view_anomaly_map").click().run()
    assert not app.exception
    assert len(app.get("image")) == 2
    assert len(overlays) == 2
    for (anomaly_map, threshold), method in zip(overlays, METHODS):
        assert anomaly_map is stored["result"][method].anomaly_map
        assert threshold == stored["result"][method].threshold
    captions = " ".join(c.value for c in app.caption)
    if uncalibrated:
        assert "seuil non calibré" in captions
        assert "NON CALIBRÉ" in " ".join(m.value for m in app.markdown)
    else:
        assert captions.count("Couleurs relatives au seuil de ce modèle.") == 2
    assert len(calls) == 2
    follow_link(app)
    assert len(app.metric) == 8
    assert app.button(key="analyze_image").disabled
    assert len(calls) == 2


def test_inputs_and_mode_have_separate_cached_results(inference_app):
    app, calls = inference_app
    app.file_uploader[0].upload("piece.png", png(), "image/png").run()
    app.button(key="analyze_image").click().run()
    individual_report = deepcopy(app.session_state["last_inspection"]["report"])
    analyze(app)
    assert len(calls) == 3
    app.checkbox(key="compare_mode").uncheck().run()
    assert len(app.metric) == 4
    assert app.session_state["last_inspection"]["report"] == individual_report
    assert app.button(key="analyze_image").disabled
    app.checkbox(key="compare_mode").check().run()
    assert len(app.metric) == 8
    assert app.button(key="analyze_image").disabled
    app.file_uploader[0].upload("renamed.png", png(), "image/png").run()
    assert app.button(key="analyze_image").disabled
    app.selectbox(key="category_selector").set_value("hazelnut").run()
    assert not app.button(key="analyze_image").disabled
    assert not app.metric
    app.button(key="analyze_image").click().run()
    assert len(calls) == 5
    app.file_uploader[0].upload("new.png", png("white"), "image/png").run()
    assert not app.button(key="analyze_image").disabled
    assert not app.metric
    assert "last_comparison" not in app.session_state
    assert "last_inspection" not in app.session_state
    app.file_uploader[0].clear().run()
    assert app.file_uploader[0].value is None
    assert app.button(key="analyze_image").disabled
    app.file_uploader[0].upload("piece.png", png(), "image/png").run()
    assert not app.button(key="analyze_image").disabled
    assert len(calls) == 5


@pytest.mark.parametrize("change", ["version", "threshold", "sha256"])
def test_registry_change_reenables_comparison(inference_app, monkeypatch, change):
    app, calls = inference_app
    analyze(app)

    def changed_registry(path):
        registry = load_registry(path)
        key = next(key for key in registry if key[:2] == ("bottle", "cae"))
        config = registry.pop(key)
        config[change] = {"version": "new-version", "threshold": 0.123, "sha256": "a" * 64}[change]
        registry[(key[0], key[1], config["version"])] = config
        return registry

    monkeypatch.setattr("src.inference.model_registry.load_registry", changed_registry)
    app.run()
    assert not app.exception
    assert not app.button(key="analyze_image").disabled
    assert not app.metric
    assert len(calls) == 2
    app.button(key="analyze_image").click().run()
    assert not app.exception
    assert app.button(key="analyze_image").disabled
    assert len(calls) == 4


def test_partial_failure_is_not_cached_and_can_be_retried(inference_app, monkeypatch):
    app, calls = inference_app
    from src.inference.inference_contract import run_inference

    def fail_second(request, registry):
        if request.method == "patchcore":
            raise RuntimeError("Échec temporaire")
        return run_inference(request, registry)

    monkeypatch.setattr("src.inference.inference_contract.run_inference", fail_second)
    analyze(app)
    assert app.error
    assert not app.metric
    assert "last_comparison" not in app.session_state
    assert not app.button(key="analyze_image").disabled
    monkeypatch.setattr("src.inference.inference_contract.run_inference", run_inference)
    app.button(key="analyze_image").click().run()
    assert not app.exception
    assert len(app.metric) == 8
    assert app.button(key="analyze_image").disabled
    assert len(calls) == 3
