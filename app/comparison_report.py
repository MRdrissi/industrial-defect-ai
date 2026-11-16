"""Rapport de comparaison : deux rapports individuels au contrat inchangé."""

import json
from datetime import datetime, timezone

from app.comparison import METHODS, SCORE_WARNING, ComparisonResult
from app.report import APPLICATION, build_report
from src.inference.model_registry import get_model


def build_comparison_report(
    comparison: ComparisonResult,
    registry: dict,
    image_bytes: bytes,
    image_name: str,
    image_size: tuple[int, int],
    generated_at: datetime | None = None,
) -> dict:
    moment = generated_at or datetime.now(timezone.utc)
    if set(comparison.results) != set(METHODS):
        raise ValueError("La comparaison doit contenir CAE et PatchCore.")
    reports = {}
    for method in METHODS:
        result = comparison[method]
        if result.category != comparison.category or result.method != method:
            raise ValueError("Le résultat ne correspond pas à la comparaison.")
        reports[method] = build_report(
            result,
            get_model(registry, comparison.category, method, result.version),
            image_bytes, image_name, image_size, generated_at=moment,
        )
    return {
        "comparison_schema_version": "1.0",
        "application": APPLICATION,
        "generated_at_utc": reports["cae"]["generated_at_utc"],
        "category": comparison.category,
        "image": reports["cae"]["image"].copy(),
        "score_warning": SCORE_WARNING,
        "reports": reports,
    }


def comparison_report_to_json(report: dict) -> bytes:
    return (json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")


def comparison_report_filename(report: dict) -> str:
    stamp = report["generated_at_utc"].replace("-", "").replace(":", "")
    return f"comparaison_{report['category']}_{stamp}.json"
