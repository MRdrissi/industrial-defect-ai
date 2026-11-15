"""Tests des formats de checkpoints PatchCore — US-06-03."""

import pytest
import torch

from src.inference.patchcore_inference import extract_patchcore_checkpoint


@pytest.fixture
def weights():
    return {"memory_bank": torch.zeros(2, 3)}


def test_original_format(weights):
    """Format Bottle et Transistor."""

    checkpoint = {
        "model_state_dict": weights,
        "backbone": "wide_resnet50_2",
        "layers": ["layer2", "layer3"],
        "num_neighbors": 1,
    }

    restored_weights, config = extract_patchcore_checkpoint(checkpoint)

    assert restored_weights is weights
    assert config["backbone"] == "wide_resnet50_2"


def test_corrected_hazelnut_format(weights):
    """Format Hazelnut EXP-002."""

    checkpoint = {
        "state_dict": weights,
        "config": {
            "category": "hazelnut",
            "pooling_corrected": True,
            "backbone": "wide_resnet50_2",
            "layers": ["layer2", "layer3"],
            "num_neighbors": 1,
        },
    }

    restored_weights, config = extract_patchcore_checkpoint(checkpoint)

    assert restored_weights is weights
    assert config["pooling_corrected"] is True


def test_uncorrected_hazelnut_rejected(weights):
    checkpoint = {
        "state_dict": weights,
        "config": {
            "category": "hazelnut",
            "pooling_corrected": False,
            "backbone": "wide_resnet50_2",
            "layers": ["layer2", "layer3"],
            "num_neighbors": 1,
        },
    }

    with pytest.raises(ValueError, match="Pooling"):
        extract_patchcore_checkpoint(checkpoint)


def test_invalid_checkpoint():
    with pytest.raises(ValueError):
        extract_patchcore_checkpoint({"unknown": "data"})
