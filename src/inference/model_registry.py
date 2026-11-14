"""Registre configurable des modèles — US-06-01."""

import json
import math
from pathlib import Path


def load_registry(path: str | Path) -> dict:
    """Charge et valide le registre JSON."""
    path = Path(path)

    if not path.is_file():
        raise FileNotFoundError(f"Registre introuvable : {path}")

    data = json.loads(path.read_text(encoding="utf-8"))

    if (
            not isinstance(data, dict)
            or not isinstance(data.get("models"), list)
            or not data["models"]
    ):
        raise ValueError("Le registre doit contenir une liste 'models' non vide.")

    registry = {}

    for model in data["models"]:
        if not isinstance(model, dict):
            raise ValueError("Entrée de modèle invalide.")

        required = (
            "category", "method", "version",
            "backend", "threshold", "artifact_path",
        )

        if any(key not in model for key in required):
            raise ValueError("Configuration de modèle incomplète.")

        if any(
                not isinstance(model[key], str) or not model[key].strip()
                for key in required[:4]
        ):
            raise ValueError("Identifiant ou backend de modèle invalide.")

        threshold = model["threshold"]
        if (
                type(threshold) not in (int, float)
                or not math.isfinite(threshold)
                or threshold < 0
        ):
            raise ValueError("Seuil de modèle invalide.")

        artifact = model["artifact_path"]
        if artifact is not None and (
                not isinstance(artifact, str) or not artifact.strip()
        ):
            raise ValueError("Chemin d'artefact invalide.")

        key = (
            model["category"],
            model["method"],
            model["version"],
        )

        if key in registry:
            raise ValueError(f"Modèle dupliqué : {key}")

        registry[key] = model

    return registry


def get_model(registry: dict, category: str, method: str, version: str) -> dict:
    """Recherche un modèle par son identifiant unique."""
    key = (category, method, version)

    if key not in registry:
        raise KeyError(f"Modèle introuvable : {key}")

    return registry[key]
