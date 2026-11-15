"""Rapport d'inspection JSON téléchargeable — US-06-04 (IDL-78).

Le rapport rend une inspection traçable : on peut retrouver, à partir du seul
fichier, l'image analysée (empreinte SHA-256), le modèle exact (catégorie,
méthode, version, checkpoint et son SHA-256), la configuration de décision
(seuil, règle), le résultat (score, décision, latence) et le moment de l'analyse.

Fonctions pures : aucune dépendance à Streamlit, testables seules.
"""

from __future__ import annotations

import hashlib
import json
import math
from datetime import datetime, timezone

from src.inference.inference_contract import InferenceResult

REPORT_SCHEMA_VERSION = "1.0"
APPLICATION = "IndustrialDefect AI"

# Règle appliquée par run_inference (US-06-01 / US-06-03).
DECISION_RULE = "suspect si score >= seuil ; non calibrée si seuil absent"

PREPROCESSING = {
    "cae": "RGB, redimensionnement 256 x 256, valeurs dans [0, 1] (MVTecTransform, profil cae)",
    "patchcore": "RGB, redimensionnement 256 x 256, normalisation ImageNet (MVTecTransform, profil patchcore)",
}

REQUIRED_KEYS = ("report_schema_version", "generated_at_utc", "image", "model", "configuration", "result")


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _iso_utc(moment: datetime) -> str:
    if moment.tzinfo is None:
        raise ValueError("La date du rapport doit avoir un fuseau horaire.")
    return moment.astimezone(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def build_report(
    result: InferenceResult,
    model_config: dict,
    image_bytes: bytes,
    image_name: str,
    image_size: tuple[int, int],
    generated_at: datetime | None = None,
) -> dict:
    """Assemble le rapport d'une inspection.

    ``model_config`` est l'entrée du registre utilisée par run_inference
    (get_model). ``image_size`` = (largeur, hauteur) de l'image importée.
    """
    identity = (result.category, result.method, result.version)
    config_identity = (model_config.get("category"), model_config.get("method"), model_config.get("version"))
    if identity != config_identity:
        raise ValueError(f"Le résultat {identity} ne correspond pas au modèle {config_identity}.")
    if not image_bytes:
        raise ValueError("Image vide : impossible de tracer l'inspection.")
    if not math.isfinite(result.score):
        raise ValueError("Score non fini.")

    width, height = image_size
    calibrated = result.threshold is not None

    return {
        "report_schema_version": REPORT_SCHEMA_VERSION,
        "application": APPLICATION,
        "generated_at_utc": _iso_utc(generated_at or _utc_now()),
        "image": {
            "name": image_name,
            "width": int(width),
            "height": int(height),
            "sha256": hashlib.sha256(image_bytes).hexdigest(),
        },
        "model": {
            "category": result.category,
            "method": result.method,
            "version": result.version,
            "backend": model_config.get("backend"),
            "simulated": result.simulated,
            "artifact_path": model_config.get("artifact_path"),
            "checkpoint_sha256": model_config.get("sha256"),
            "config_path": model_config.get("config_path"),
        },
        "configuration": {
            "threshold": result.threshold,
            "threshold_status": "calibré" if calibrated else "non calibré",
            "decision_rule": DECISION_RULE,
            "preprocessing": PREPROCESSING.get(result.method, "inconnu"),
            "device": "cpu",
        },
        "result": {
            "score": result.score,
            "decision": result.decision,
            "latency_ms": round(result.latency_ms, 3),
            "anomaly_map_shape": list(result.anomaly_map.shape),
        },
    }


def report_to_json(report: dict) -> bytes:
    """Sérialise le rapport (UTF-8, indenté, accents conservés)."""
    missing = [key for key in REQUIRED_KEYS if key not in report]
    if missing:
        raise ValueError(f"Rapport incomplet : {missing}")
    return (json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")


def report_filename(report: dict) -> str:
    """Nom de fichier lisible : inspection_<categorie>_<methode>_<version>_<date>.json."""
    model = report["model"]
    stamp = report["generated_at_utc"].replace("-", "").replace(":", "")
    return f"inspection_{model['category']}_{model['method']}_{model['version']}_{stamp}.json"
