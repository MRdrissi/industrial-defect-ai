"""Tests du contrat et du registre — US-06-01."""

import json
from pathlib import Path

import pytest
import torch

from src.inference.model_registry import load_registry, get_model
from src.inference.inference_contract import (
    InferenceRequest,
    load_fake_model,
    run_inference,
)

ROOT = Path(__file__).resolve().parents[2]
REGISTRY_PATH = ROOT / "configs/model_registry.json"


def registry():
    return load_registry(REGISTRY_PATH)


def request(image):
    return InferenceRequest(
        category="bottle",
        method="cae",
        version="demo-v1",
        image=image,
    )


def test_registry():
    models = registry()

    assert len(models) == 6
    assert get_model(models, "bottle", "cae", "demo-v1")["backend"] == "fake"


def test_unknown_model():
    with pytest.raises(KeyError):
        get_model(registry(), "unknown", "cae", "demo-v1")


def test_missing_registry():
    with pytest.raises(FileNotFoundError):
        load_registry("missing_registry.json")


def test_invalid_configuration(tmp_path):
    path = tmp_path / "invalid.json"
    path.write_text(
        json.dumps({"models": [{"category": "bottle"}]}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError):
        load_registry(path)


def test_duplicate_models(tmp_path):
    model = get_model(registry(), "bottle", "cae", "demo-v1")
    path = tmp_path / "duplicates.json"
    path.write_text(
        json.dumps({"models": [model, model]}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError):
        load_registry(path)


def test_invalid_threshold(tmp_path):
    model = get_model(registry(), "bottle", "cae", "demo-v1").copy()
    model["threshold"] = -1

    path = tmp_path / "invalid_threshold.json"
    path.write_text(
        json.dumps({"models": [model]}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError):
        load_registry(path)


def test_cpu_loading():
    model = load_fake_model()

    assert not model.training
    assert next(model.buffers()).device.type == "cpu"

    with pytest.raises(ValueError):
        load_fake_model("cuda")


def test_normal_result():
    image = torch.full((3, 8, 8), 0.5)
    result = run_inference(request(image), registry())

    assert result.score == pytest.approx(0.0)
    assert result.decision == "normal"
    assert result.threshold == pytest.approx(0.2)
    assert result.simulated is True
    assert result.latency_ms >= 0
    assert result.anomaly_map.shape == (8, 8)
    assert result.anomaly_map.device.type == "cpu"
    assert result.anomaly_map.dtype == torch.float32

    output = result.to_dict()
    assert output["anomaly_map_shape"] == [8, 8]
    json.dumps(output)


def test_suspect_result():
    image = torch.zeros(3, 8, 8)
    result = run_inference(request(image), registry())

    assert result.score == pytest.approx(0.5)
    assert result.decision == "suspect"


def test_invalid_dimensions():
    image = torch.zeros(1, 8, 8)

    with pytest.raises(ValueError):
        run_inference(request(image), registry())


def test_invalid_values():
    image = torch.zeros(3, 8, 8)
    image[0, 0, 0] = float("nan")

    with pytest.raises(ValueError):
        run_inference(request(image), registry())


def test_invalid_dtype():
    image = torch.zeros(3, 8, 8, dtype=torch.uint8)

    with pytest.raises(TypeError):
        run_inference(request(image), registry())


def test_unsupported_backend():
    models = registry()
    key = ("bottle", "cae", "demo-v1")
    models[key] = {**models[key], "backend": "checkpoint"}

    with pytest.raises(NotImplementedError):
        run_inference(
            request(torch.zeros(3, 8, 8)),
            models,
        )
