"""Tests du chargement et du cache — US-06-03."""

import hashlib

import pytest

from src.inference import model_loader


@pytest.fixture(autouse=True)
def reset_cache():
    model_loader.clear_model_cache()
    yield
    model_loader.clear_model_cache()


def make_registry(checkpoint, category="bottle"):
    return {
        (category, "patchcore", "exp-002"): {
            "category": category,
            "method": "patchcore",
            "version": "exp-002",
            "backend": "patchcore",
            "artifact_path": str(checkpoint),
            "sha256": hashlib.sha256(
                checkpoint.read_bytes()
            ).hexdigest(),
        }
    }


def test_lazy_loading_and_cache(monkeypatch, tmp_path):
    """Le modèle est chargé une seule fois pour deux appels."""

    checkpoint = tmp_path / "patchcore_model.pt"
    checkpoint.write_bytes(b"test")

    registry = make_registry(checkpoint)

    calls = []
    fake_engine = object()

    def fake_loader(**kwargs):
        calls.append(kwargs)
        return fake_engine

    monkeypatch.setattr(
        model_loader.PatchCoreInference,
        "from_checkpoint",
        fake_loader,
    )

    # Aucun chargement avant la première demande.
    assert len(calls) == 0

    first = model_loader.load_real_engine(
        registry, "bottle", "patchcore", "exp-002"
    )

    second = model_loader.load_real_engine(
        registry, "bottle", "patchcore", "exp-002"
    )

    assert first is second
    assert len(calls) == 1
    assert calls[0]["device"] == "cpu"


def test_cache_limited_to_one_model(monkeypatch, tmp_path):
    """Changer de modèle remplace l'entrée du cache."""

    checkpoint = tmp_path / "patchcore_model.pt"
    checkpoint.write_bytes(b"test")

    calls = []

    def fake_loader(**kwargs):
        calls.append(kwargs)
        return object()

    monkeypatch.setattr(
        model_loader.PatchCoreInference,
        "from_checkpoint",
        fake_loader,
    )

    bottle = make_registry(checkpoint, "bottle")
    hazelnut = make_registry(checkpoint, "hazelnut")

    model_loader.load_real_engine(
        bottle, "bottle", "patchcore", "exp-002"
    )

    model_loader.load_real_engine(
        hazelnut, "hazelnut", "patchcore", "exp-002"
    )

    # Bottle a été évincé du cache.
    model_loader.load_real_engine(
        bottle, "bottle", "patchcore", "exp-002"
    )

    assert len(calls) == 3


def test_missing_checkpoint(tmp_path):
    """Un checkpoint absent doit être refusé."""

    checkpoint = tmp_path / "missing.pt"

    registry = {
        ("bottle", "patchcore", "exp-002"): {
            "backend": "patchcore",
            "artifact_path": str(checkpoint),
            "sha256": "0" * 64,
        }
    }

    with pytest.raises(FileNotFoundError):
        model_loader.load_real_engine(
            registry, "bottle", "patchcore", "exp-002"
        )


def test_fake_backend_rejected():
    """Le chargeur réel ne doit pas accepter la simulation."""

    registry = {
        ("bottle", "cae", "demo-v1"): {
            "backend": "fake",
        }
    }

    with pytest.raises(ValueError):
        model_loader.load_real_engine(
            registry, "bottle", "cae", "demo-v1"
        )


def test_invalid_cae_hash(tmp_path):
    """Un fichier CAE dont l'empreinte est fausse est refusé."""

    checkpoint = tmp_path / "cae_model.pt"
    checkpoint.write_bytes(b"wrong-content")

    registry = {
        ("bottle", "cae", "exp-002"): {
            "backend": "cae",
            "artifact_path": str(checkpoint),
            "config_path": "experiments/configs/cae_baseline.json",
            "sha256": "0" * 64,
        }
    }

    with pytest.raises(ValueError, match="SHA-256"):
        model_loader.load_real_engine(
            registry, "bottle", "cae", "exp-002"
        )
