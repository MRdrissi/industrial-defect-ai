"""Tests du contrat d'inférence réel — US-06-03."""

from types import SimpleNamespace

import pytest
import torch

from src.inference import inference_contract
from src.inference.inference_contract import (
    InferenceRequest,
    run_inference,
)


@pytest.mark.parametrize(
    "category",
    ["bottle", "hazelnut", "transistor"],
)
@pytest.mark.parametrize("method", ["cae", "patchcore"])
def test_real_inference_contract(monkeypatch, category, method):
    """Les six combinaisons passent par le moteur réel."""

    version = "exp-test"

    registry = {
        (category, method, version): {
            "backend": method,
            "threshold": None,
        }
    }

    class StubEngine:
        def predict(self, image):
            assert image.shape == (3, 256, 256)
            assert image.dtype == torch.float32
            assert torch.isfinite(image).all()

            if method == "cae":
                assert image.min() >= 0
                assert image.max() <= 1
            else:
                # La normalisation ImageNet produit ici
                # des valeurs négatives.
                assert image.min() < 0

            return SimpleNamespace(
                score=0.42,
                anomaly_map=torch.ones(256, 256),
            )

    def fake_loader(reg, cat, meth, ver):
        assert (cat, meth, ver) == (category, method, version)
        return StubEngine()

    monkeypatch.setattr(
        inference_contract,
        "load_real_engine",
        fake_loader,
    )

    request = InferenceRequest(
        category=category,
        method=method,
        version=version,
        image=torch.zeros(3, 32, 32),
    )

    result = run_inference(request, registry)

    assert result.simulated is False
    assert result.score == pytest.approx(0.42)
    assert result.threshold is None
    assert result.decision == "non_calibree"
    assert result.anomaly_map.shape == (256, 256)
    assert result.anomaly_map.device.type == "cpu"
    assert result.latency_ms >= 0


def test_real_checkpoint_error(monkeypatch):
    """Une erreur de restauration doit être propagée."""

    def broken_loader(*args):
        raise FileNotFoundError("Checkpoint introuvable")

    monkeypatch.setattr(
        inference_contract,
        "load_real_engine",
        broken_loader,
    )

    registry = {
        ("bottle", "cae", "exp-test"): {
            "backend": "cae",
            "threshold": None,
        }
    }

    request = InferenceRequest(
        category="bottle",
        method="cae",
        version="exp-test",
        image=torch.zeros(3, 256, 256),
    )

    with pytest.raises(FileNotFoundError):
        run_inference(request, registry)
