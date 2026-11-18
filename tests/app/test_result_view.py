"""Contrat de la synthèse individuelle, sans vérification visuelle."""

import pytest
import torch
from streamlit.testing.v1 import AppTest

from app.result_view import decision_label, model_label
from src.inference.inference_contract import InferenceResult


def make_result(**overrides):
    values = dict(
        category="transistor", method="patchcore", version="exp-001", score=40.13,
        threshold=None, anomaly_map=torch.zeros(8, 8), decision="non_calibree",
        latency_ms=12.5, simulated=False,
    )
    values.update(overrides)
    return InferenceResult(**values)


def test_model_label():
    assert model_label(make_result()) == "Modèle : Transistor · PatchCore"
    assert "Convolutional AutoEncoder" in model_label(make_result(method="cae"))


@pytest.mark.parametrize("decision, expected", [
    ("normal", "NORMAL"), ("suspect", "SUSPECT"), ("non_calibree", "NON CALIBRÉE"),
])
def test_decision_label(decision, expected):
    assert decision_label(make_result(decision=decision)) == expected


def summary_script(result):
    from app.result_view import display_result
    display_result(result)


@pytest.mark.parametrize("threshold, decision, badge", [
    (None, "non_calibree", ":gray-badge[NON CALIBRÉ]"),
    (50.0, "normal", ":green-badge[NORMAL]"),
    (20.0, "suspect", ":red-badge[SUSPECT]"),
])
def test_summary_metrics_and_no_inline_details(threshold, decision, badge):
    at = AppTest.from_function(summary_script, args=(make_result(
        threshold=threshold, decision=decision,
    ),)).run()
    assert not at.exception
    assert at.subheader[0].value == "Résultat de l'inspection"
    assert at.subheader[0].proto.hide_anchor
    assert [metric.label for metric in at.metric] == ["Score brut", "Seuil", "Latence", "Verdict"]
    assert at.metric[3].value == badge
    assert at.metric[1].value == ("Non calibré" if threshold is None else f"{threshold:.4f}")
    assert not at.get("image")
    assert not at.json
    assert not at.expander
