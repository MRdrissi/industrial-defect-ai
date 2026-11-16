"""Traçabilité des deux modèles et conservation exacte du contrat individuel."""

import hashlib
import json
from dataclasses import replace
from datetime import datetime, timezone

import pytest
import torch

from app.comparison import METHODS, SCORE_WARNING, ComparisonResult
from app.comparison_report import (
    build_comparison_report, comparison_report_filename, comparison_report_to_json,
)
from app.report import build_report, report_to_json
from src.inference.inference_contract import InferenceResult


@pytest.fixture
def comparison_data():
    results, registry = {}, {}
    for method, score, threshold, decision in (
        ("cae", 0.03, 0.02, "suspect"), ("patchcore", 39.0, 40.0, "normal"),
    ):
        version = f"{method}-v1"
        results[method] = InferenceResult(
            category="bottle", method=method, version=version, score=score,
            threshold=threshold, decision=decision, anomaly_map=torch.zeros(8, 8),
            latency_ms=12.34567, simulated=False,
        )
        registry[("bottle", method, version)] = {
            "category": "bottle", "method": method, "version": version,
            "backend": method, "threshold": threshold,
            "artifact_path": f"models/{method}.pt", "sha256": method[0] * 64,
            "config_path": f"configs/{method}.yaml",
        }
    return ComparisonResult("bottle", (3, 8, 8), results), registry


def test_comparison_embeds_unchanged_individual_reports(comparison_data):
    comparison, registry = comparison_data
    moment = datetime(2026, 11, 16, 12, 0, tzinfo=timezone.utc)
    content = b"source image bytes"
    report = build_comparison_report(comparison, registry, content, "pièce.png", (8, 8), moment)
    assert report["image"]["sha256"] == hashlib.sha256(content).hexdigest()
    assert report["score_warning"] == SCORE_WARNING
    assert report["comparison_schema_version"] == "1.0"
    assert json.loads(comparison_report_to_json(report)) == report
    assert comparison_report_filename(report) == "comparaison_bottle_20261116T120000Z.json"
    for method in METHODS:
        result = comparison[method]
        individual = build_report(
            result, registry[("bottle", method, result.version)], content, "pièce.png", (8, 8), moment,
        )
        assert report["reports"][method] == individual
        assert json.loads(report_to_json(report["reports"][method])) == individual
        assert individual["image"] == report["image"]
        assert individual["generated_at_utc"] == report["generated_at_utc"]
        assert individual["model"]["checkpoint_sha256"] == method[0] * 64
        assert individual["configuration"]["threshold"] == result.threshold
        assert individual["result"]["score"] == result.score


def test_uncalibrated_result_preserved(comparison_data):
    comparison, registry = comparison_data
    comparison.results["cae"] = replace(comparison["cae"], threshold=None, decision="non_calibree")
    report = build_comparison_report(comparison, registry, b"image", "x.jpg", (8, 8))
    decoded = json.loads(comparison_report_to_json(report))
    assert decoded["reports"]["cae"]["configuration"]["threshold"] is None
    assert decoded["reports"]["cae"]["result"]["decision"] == "non_calibree"
    assert decoded["reports"]["patchcore"]["configuration"]["threshold"] == 40.0


@pytest.mark.parametrize("invalid", ["missing", "category", "method", "version"])
def test_mismatched_or_partial_results_rejected(comparison_data, invalid):
    comparison, registry = comparison_data
    if invalid == "missing":
        comparison.results.pop("patchcore")
    else:
        comparison.results["cae"] = replace(comparison["cae"], **{invalid: "incorrect"})
    with pytest.raises((ValueError, KeyError)):
        build_comparison_report(comparison, registry, b"image", "x.jpg", (8, 8))
