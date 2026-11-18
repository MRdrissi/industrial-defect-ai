"""Seuils calibrés dans le registre réel — IDL-96.

Les six seuils de ``configs/real_model_registry.json`` doivent être exactement
ceux de la calibration US-05-02 (``experiments/calibration/six-thresholds/
thresholds.json`` sur ``epic/05``), pour le même checkpoint, et la décision
doit suivre la règle figée : anomalie si score > seuil (égalité = normal).
"""

from pathlib import Path

import pytest
import torch

from src.inference.inference_contract import InferenceRequest, run_inference
from src.inference.model_registry import load_registry

ROOT = Path(__file__).resolve().parents[2]
REAL_REGISTRY = ROOT / "configs/real_model_registry.json"
THRESHOLDS_REGISTRY_SHA256 = "f3128a255482f55cb416312780bc4832f4bfe24b017c9a2036089128ade68b54"

# (catégorie, méthode) -> (seuil, représentation hexadécimale exacte, SHA-256 du checkpoint)
CALIBRATED = {
    ("bottle", "cae"): (0.01955269752070307, "0x1.4059f51999997p-6",
                        "12501c4d125e8dd0e6512062f4cc76cfa4287770974bd170af7e3a9528fa88e8"),
    ("hazelnut", "cae"): (0.008434063522145145, "0x1.145e0d7fffffcp-7",
                          "34ab16490dc5e75003971a54e53bbda61b2f6be9428f63ace322b06c64d7e811"),
    ("transistor", "cae"): (0.027364758774638176, "0x1.c0581e0000000p-6",
                            "7849dab459cf2ae0c2a6dd764a2864998a298a6ad0e5498d6ee5882326cc0a2d"),
    ("bottle", "patchcore"): (31.620044136047362, "0x1.f9ebb36666666p+4",
                              "dcf0f8df3dc9b4be9ee1bcb897d3ec6f24a3191240dcfd4e68afd598acd77cde"),
    ("hazelnut", "patchcore"): (48.83736801147461, "0x1.86b2ee0000000p+5",
                                "7b4c3784434d69724e8e7fb74d587e4c8a059faa24819dd556c224d03e580a93"),
    ("transistor", "patchcore"): (40.95855598449707, "0x1.47ab1f6666666p+5",
                                  "7917810678f656324236a0df6db639d7de93176a7a8ae9ef06e72f3bf0a57191"),
}


@pytest.fixture(scope="module")
def registry():
    return load_registry(REAL_REGISTRY)


def test_six_models_have_a_calibrated_threshold(registry):
    assert {(key[0], key[1]) for key in registry} == set(CALIBRATED)


@pytest.mark.parametrize("instance", sorted(CALIBRATED))
def test_threshold_matches_calibration_exactly(registry, instance):
    category, method = instance
    expected, expected_hex, checkpoint_sha256 = CALIBRATED[instance]
    (model,) = [m for key, m in registry.items() if key[:2] == instance]

    assert model["sha256"] == checkpoint_sha256          # même checkpoint
    assert model["threshold"] == expected                # même valeur, sans arrondi
    assert float(model["threshold"]).hex() == expected_hex
    assert model["threshold_source"]["threshold_hex"] == expected_hex
    assert model["threshold_source"]["registry_sha256"] == THRESHOLDS_REGISTRY_SHA256
    assert model["threshold_source"]["decision_rule"] == "score > threshold"


def fake_registry(threshold):
    return {
        ("bottle", "cae", "demo"): {
            "category": "bottle", "method": "cae", "version": "demo",
            "backend": "fake", "threshold": threshold, "artifact_path": None,
        }
    }


def decide(image_value, threshold):
    image = torch.full((3, 8, 8), image_value)
    return run_inference(InferenceRequest("bottle", "cae", "demo", image), fake_registry(threshold))


def test_score_equal_to_threshold_is_normal():
    # Image uniforme à 0,5 : score factice exactement 0,0 ; seuil 0,0.
    result = decide(0.5, 0.0)
    assert result.score == 0.0
    assert result.decision == "normal"


def test_score_above_threshold_is_suspect():
    result = decide(0.75, 0.0)  # score 0,25 > 0
    assert result.decision == "suspect"


def test_score_below_threshold_is_normal():
    result = decide(0.75, 0.3)  # score 0,25 < 0,3
    assert result.decision == "normal"
