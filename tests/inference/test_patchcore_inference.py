"""Tests unitaires du moteur PatchCore — US-04-05."""

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from src.inference.patchcore_inference import (
    PatchCoreInference,
    load_patchcore,
)


class FakePatchCore(nn.Module):
    """Simule PatchCore sans charger de backbone réel."""

    def __init__(self):
        super().__init__()

        # Paramètre factice pour identifier le device.
        self.anchor = nn.Parameter(torch.zeros(1))
        self.inference_mode_used = False

    def forward(self, images):
        self.inference_mode_used = torch.is_inference_mode_enabled()

        # Un score brut par image : [B].
        scores = images.abs().mean(dim=(1, 2, 3))

        # Une carte par image : [B, 1, H, W].
        maps = images.abs().mean(dim=1, keepdim=True)

        return SimpleNamespace(
            pred_score=scores,
            anomaly_map=maps,
        )


@pytest.fixture
def engine():
    """Crée un moteur d'inférence factice pour Bottle."""
    return PatchCoreInference(
        model=FakePatchCore(),
        category="bottle",
    )


def test_predict_single_image(engine):
    image = torch.ones(3, 256, 256)

    result = engine.predict(image)

    assert result.category == "bottle"
    assert result.model_family == "patchcore"
    assert result.model_name == "patchcore"
    assert result.score == pytest.approx(1.0)

    assert result.anomaly_map.shape == (256, 256)
    assert result.anomaly_map.dtype == torch.float32
    assert result.anomaly_map.device.type == "cpu"
    assert torch.isfinite(result.anomaly_map).all()


def test_predict_batch(engine):
    images = torch.stack([
        torch.zeros(3, 256, 256),
        torch.ones(3, 256, 256),
    ])

    results = engine.predict_batch(images)

    assert len(results) == 2
    assert results[0].score == pytest.approx(0.0)
    assert results[1].score == pytest.approx(1.0)

    for result in results:
        assert result.anomaly_map.shape == (256, 256)


def test_inference_mode(engine):
    image = torch.ones(3, 256, 256)

    engine.predict(image)

    assert engine.model.inference_mode_used
    assert not engine.model.training


def test_result_serialization(engine):
    result = engine.predict(torch.ones(3, 256, 256))

    output = result.to_dict()

    assert output["category"] == "bottle"
    assert output["model_family"] == "patchcore"
    assert output["model_name"] == "patchcore"
    assert output["anomaly_map_shape"] == [256, 256]
    assert isinstance(output["score"], float)


@pytest.mark.parametrize(
    "images",
    [
        torch.zeros(3, 256, 256),
        torch.zeros(1, 1, 256, 256),
        torch.zeros(0, 3, 256, 256),
    ],
)
def test_invalid_dimensions(engine, images):
    with pytest.raises(ValueError):
        engine.predict_batch(images)


@pytest.mark.parametrize("value", [float("nan"), float("inf")])
def test_invalid_values(engine, value):
    images = torch.ones(1, 3, 256, 256)
    images[0, 0, 0, 0] = value

    with pytest.raises(ValueError):
        engine.predict_batch(images)


def test_invalid_dtype(engine):
    images = torch.zeros(1, 3, 256, 256, dtype=torch.uint8)

    with pytest.raises(TypeError):
        engine.predict_batch(images)


def test_missing_checkpoint(tmp_path):
    checkpoint_path = tmp_path / "missing.pt"

    with pytest.raises(FileNotFoundError):
        load_patchcore(checkpoint_path)
