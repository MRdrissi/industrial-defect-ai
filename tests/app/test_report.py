"""Tests du rapport d'inspection JSON — US-06-04 (IDL-78).

Vérifie que le rapport contient la date, la configuration et la décision,
qu'il permet de retrouver l'image et le modèle exacts (traçabilité),
et qu'il est proposé au téléchargement dans l'application.
"""

import hashlib
import json
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path

import pytest
import torch
from PIL import Image
from streamlit.testing.v1 import AppTest

from app.report import build_report, report_filename, report_to_json
from src.inference.inference_contract import InferenceResult
from src.inference.model_registry import get_model, load_registry

ROOT = Path(__file__).resolve().parents[2]
APP = ROOT / "app/streamlit_app.py"
REAL_REGISTRY = ROOT / "configs/real_model_registry.json"
FIXED_DATE = datetime(2026, 11, 15, 17, 30, 5, tzinfo=timezone.utc)


def make_png(side=32, color=(128, 128, 128)):
    buffer = BytesIO()
    Image.new("RGB", (side, side), color).save(buffer, format="PNG")
    return buffer.getvalue()


def make_result(**overrides):
    values = dict(
        category="transistor",
        method="patchcore",
        version="exp-001",
        score=40.13337707519531,
        threshold=None,
        anomaly_map=torch.zeros(256, 256),
        decision="non_calibree",
        latency_ms=123.456789,
        simulated=False,
    )
    values.update(overrides)
    return InferenceResult(**values)


@pytest.fixture
def real_config():
    """Entrée réelle du registre US-06-03 (PatchCore Transistor)."""
    return get_model(load_registry(REAL_REGISTRY), "transistor", "patchcore", "exp-001")


def test_report_contains_date_configuration_and_decision(real_config):
    image_bytes = make_png(1024)
    report = build_report(
        make_result(), real_config, image_bytes, "transistor.png", (1024, 1024), FIXED_DATE
    )

    assert report["generated_at_utc"] == "2026-11-15T17:30:05Z"
    assert report["configuration"]["threshold"] is None
    assert report["configuration"]["threshold_status"] == "non calibré"
    assert "seuil" in report["configuration"]["decision_rule"]
    assert "patchcore" in report["configuration"]["preprocessing"]
    assert report["result"]["decision"] == "non_calibree"
    assert report["result"]["score"] == 40.13337707519531
    assert report["result"]["anomaly_map_shape"] == [256, 256]


def test_report_traces_image_and_exact_model(real_config):
    image_bytes = make_png(1024)
    report = build_report(make_result(), real_config, image_bytes, "transistor.png", (1024, 1024), FIXED_DATE)

    assert report["image"] == {
        "name": "transistor.png",
        "width": 1024,
        "height": 1024,
        "sha256": hashlib.sha256(image_bytes).hexdigest(),
    }
    assert report["model"]["version"] == "exp-001"
    assert report["model"]["simulated"] is False
    assert report["model"]["artifact_path"] == real_config["artifact_path"]
    assert report["model"]["checkpoint_sha256"] == real_config["sha256"]


def test_calibrated_threshold_is_reported():
    config = {"category": "bottle", "method": "cae", "version": "demo-v1", "backend": "fake",
              "threshold": 0.2, "artifact_path": None}
    result = make_result(category="bottle", method="cae", version="demo-v1",
                         score=0.31, threshold=0.2, decision="suspect", simulated=True)
    report = build_report(result, config, make_png(), "x.png", (32, 32), FIXED_DATE)

    assert report["configuration"]["threshold"] == 0.2
    assert report["configuration"]["threshold_status"] == "calibré"
    assert report["result"]["decision"] == "suspect"
    assert report["model"]["checkpoint_sha256"] is None


def test_json_round_trip_and_filename(real_config):
    report = build_report(make_result(), real_config, make_png(), "é.png", (32, 32), FIXED_DATE)
    data = report_to_json(report)

    assert json.loads(data.decode("utf-8")) == report
    assert "é.png" in data.decode("utf-8")  # accents conservés
    assert report_filename(report) == "inspection_transistor_patchcore_exp-001_20261115T173005Z.json"


def test_report_rejects_mismatched_model(real_config):
    with pytest.raises(ValueError):
        build_report(make_result(category="bottle"), real_config, make_png(), "x.png", (32, 32))


def test_report_rejects_empty_image_and_naive_date(real_config):
    with pytest.raises(ValueError):
        build_report(make_result(), real_config, b"", "x.png", (32, 32))
    with pytest.raises(ValueError):
        build_report(make_result(), real_config, make_png(), "x.png", (32, 32), datetime(2026, 11, 15))


def test_incomplete_report_is_not_serialized():
    with pytest.raises(ValueError):
        report_to_json({"image": {}})


def test_download_button_is_offered_after_analysis():
    at = AppTest.from_file(str(APP)).run()
    at.file_uploader[0].upload("normal.png", make_png(), "image/png").run()
    at.button[0].click().run()

    assert not at.exception
    assert len(at.download_button) == 1
    assert at.download_button[0].label == "Télécharger le rapport JSON"
    assert any("Rapport du" in caption.value for caption in at.caption)


def test_no_download_button_before_analysis():
    at = AppTest.from_file(str(APP)).run()
    assert not at.exception
    assert len(at.download_button) == 0
