"""Tests du registre réel — US-06-03."""

import json
from pathlib import Path

import pytest

from src.inference.model_registry import load_registry, get_model


ROOT = Path(__file__).resolve().parents[2]

FAKE_REGISTRY = ROOT / "configs/model_registry.json"
REAL_REGISTRY = ROOT / "configs/real_model_registry.json"


def test_registries_are_independent():
    """Les deux registres restent indépendants."""

    fake = load_registry(FAKE_REGISTRY)
    real = load_registry(REAL_REGISTRY)

    assert len(fake) == 6
    assert len(real) == 6

    assert all(m["backend"] == "fake" for m in fake.values())
    assert all(m["backend"] != "fake" for m in real.values())


def test_six_real_models():
    """Les six combinaisons attendues sont présentes."""

    registry = load_registry(REAL_REGISTRY)

    expected = {
        ("bottle", "cae", "exp-002"),
        ("hazelnut", "cae", "exp-001"),
        ("transistor", "cae", "exp-001"),
        ("bottle", "patchcore", "exp-002"),
        ("hazelnut", "patchcore", "exp-002"),
        ("transistor", "patchcore", "exp-001"),
    }

    assert set(registry.keys()) == expected


def test_real_model_configuration():
    """Chaque modèle possède les métadonnées nécessaires."""

    registry = load_registry(REAL_REGISTRY)

    for model in registry.values():
        assert model["backend"] == model["method"]
        assert model["threshold"] is None
        assert model["artifact_path"].startswith("artifacts/")
        assert len(model["sha256"]) == 64

        if model["method"] == "cae":
            assert model["config_path"] == (
                "experiments/configs/cae_baseline.json"
            )


def test_get_real_model():
    """Recherche d'un modèle réel dans le registre."""

    registry = load_registry(REAL_REGISTRY)

    model = get_model(
        registry,
        "bottle",
        "cae",
        "exp-002",
    )

    assert model["backend"] == "cae"
    assert model["version"] == "exp-002"


@pytest.mark.parametrize(
    "changes",
    [
        {"backend": "unknown"},
        {"backend": "patchcore"},
        {"threshold": -1},
        {"artifact_path": None},
        {"config_path": ""},
        {"sha256": "invalid"},
    ],
)
def test_invalid_real_configuration(tmp_path, changes):
    """Les configurations incorrectes sont refusées."""

    registry = load_registry(REAL_REGISTRY)

    model = get_model(
        registry, "bottle", "cae", "exp-002"
    ).copy()

    model.update(changes)

    path = tmp_path / "invalid_registry.json"
    path.write_text(
        json.dumps({"models": [model]}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError):
        load_registry(path)
