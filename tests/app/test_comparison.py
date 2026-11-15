"""Tests de la logique de comparaison CAE / PatchCore — US-06-05 (IDL-79)."""

from pathlib import Path

import pytest
import torch

from app.comparison import (
    METHODS,
    SCORE_MEANINGS,
    SCORE_WARNING,
    available_versions,
    decisions_summary,
    run_comparison,
)
from src.inference.inference_contract import InferenceResult
from src.inference.model_registry import load_registry

ROOT = Path(__file__).resolve().parents[2]
DEMO_REGISTRY = load_registry(ROOT / "configs/model_registry.json")
REAL_REGISTRY = load_registry(ROOT / "configs/real_model_registry.json")


def fake_result(method, decision="normal", score=0.1, threshold=0.2, category="transistor", version="v"):
    return InferenceResult(
        category=category,
        method=method,
        version=version,
        score=score,
        threshold=threshold,
        anomaly_map=torch.zeros(8, 8),
        decision=decision,
        latency_ms=1.0,
        simulated=True,
    )


def comparison_of(cae_decision, patchcore_decision):
    def infer(request, _registry):
        decision = cae_decision if request.method == "cae" else patchcore_decision
        return fake_result(request.method, decision, version=request.version)

    return run_comparison(torch.rand(3, 8, 8), "transistor", {"cae": "v", "patchcore": "v"}, {}, infer)


@pytest.mark.parametrize("category", ["bottle", "hazelnut", "transistor"])
def test_both_methods_available_for_each_category(category):
    assert available_versions(DEMO_REGISTRY, category) == {"cae": ["demo-v1"], "patchcore": ["demo-v1"]}
    real = available_versions(REAL_REGISTRY, category)
    assert set(real) == set(METHODS)
    assert all(len(versions) >= 1 for versions in real.values())


def test_missing_method_is_rejected():
    only_cae = {key: value for key, value in DEMO_REGISTRY.items() if key[1] == "cae"}
    with pytest.raises(ValueError, match="PatchCore"):
        available_versions(only_cae, "bottle")


def test_both_methods_receive_the_same_image():
    image = torch.rand(3, 32, 48)
    received = {}

    def infer(request, _registry):
        received[request.method] = request.image
        return fake_result(request.method, version=request.version)

    comparison = run_comparison(image, "transistor", {"cae": "a", "patchcore": "b"}, {}, infer)

    assert list(received) == ["cae", "patchcore"]
    assert received["cae"] is image and received["patchcore"] is image
    assert comparison.image_shape == (3, 32, 48)
    assert comparison["cae"].version == "a" and comparison["patchcore"].version == "b"


def test_simulated_comparison_end_to_end():
    image = torch.full((3, 16, 16), 0.5)
    comparison = run_comparison(
        image, "hazelnut", {"cae": "demo-v1", "patchcore": "demo-v1"}, DEMO_REGISTRY
    )

    for method in METHODS:
        result = comparison[method]
        assert result.method == method
        assert result.category == "hazelnut"
        assert tuple(result.anomaly_map.shape) == (16, 16)
        assert result.decision == "normal"


def test_versions_must_cover_both_methods():
    with pytest.raises(ValueError):
        run_comparison(torch.rand(3, 8, 8), "bottle", {"cae": "demo-v1"}, DEMO_REGISTRY)


def test_unexpected_result_is_rejected():
    def infer(request, _registry):
        return fake_result("cae", version=request.version)  # toujours CAE

    with pytest.raises(RuntimeError):
        run_comparison(torch.rand(3, 8, 8), "transistor", {"cae": "v", "patchcore": "v"}, {}, infer)


@pytest.mark.parametrize(
    "cae, patchcore, expected",
    [
        ("normal", "normal", "Les deux méthodes concluent : NORMAL."),
        ("suspect", "suspect", "Les deux méthodes concluent : SUSPECT."),
        ("normal", "suspect", "CAE → NORMAL, PatchCore → SUSPECT."),
        ("non_calibree", "suspect", "aucune conclusion commune"),
    ],
)
def test_decisions_summary(cae, patchcore, expected):
    assert expected in decisions_summary(comparison_of(cae, patchcore))


def test_summary_never_compares_raw_scores():
    summary = decisions_summary(comparison_of("normal", "suspect"))
    assert "score" not in summary.lower()
    assert "plus" not in summary.lower()


def test_warning_and_score_meanings_are_defined():
    assert "pas comparables" in SCORE_WARNING
    assert set(SCORE_MEANINGS) == set(METHODS)
