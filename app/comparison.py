"""Comparaison CAE / PatchCore sur une même image — US-06-05 (IDL-79).

Logique pure, sans Streamlit :

- ``available_versions`` : versions disponibles de chaque méthode pour une catégorie ;
- ``run_comparison``     : lance les deux méthodes sur le MÊME tenseur image ;
- ``decisions_summary``  : résume l'accord ou le désaccord des deux décisions.

Les scores bruts ne sont pas comparables entre méthodes (critère 3) :
- CAE       : erreur de reconstruction (maximum de la carte lissée), ordre de 0,01 ;
- PatchCore : distance au patch normal le plus proche, ordre de 10 à 50.
On compare donc chaque score à SON seuil, jamais les deux scores entre eux.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import torch

from src.inference.inference_contract import InferenceRequest, InferenceResult, run_inference

METHODS = ("cae", "patchcore")
METHOD_LABELS = {"cae": "CAE", "patchcore": "PatchCore"}
DECISION_LABELS = {"normal": "NORMAL", "suspect": "SUSPECT", "non_calibree": "NON CALIBRÉE"}

SCORE_MEANINGS = {
    "cae": "erreur de reconstruction (maximum de la carte lissée)",
    "patchcore": "distance au patch normal le plus proche",
}

SCORE_WARNING = (
    "Les scores bruts ne sont pas comparables entre les deux méthodes : ils n'ont ni la même "
    "unité ni la même échelle. Comparez chaque score uniquement à son propre seuil."
)


@dataclass(frozen=True)
class ComparisonResult:
    """Résultats des deux méthodes pour une même image et une même catégorie."""

    category: str
    image_shape: tuple[int, ...]
    results: dict[str, InferenceResult]

    def __getitem__(self, method: str) -> InferenceResult:
        return self.results[method]


def available_versions(registry: dict, category: str) -> dict[str, list[str]]:
    """Versions triées de chaque méthode ; erreur si une méthode manque pour la catégorie."""
    versions = {
        method: sorted({key[2] for key in registry if key[0] == category and key[1] == method})
        for method in METHODS
    }
    missing = [METHOD_LABELS[m] for m, found in versions.items() if not found]
    if missing:
        raise ValueError(f"Comparaison impossible pour {category} : aucun modèle {', '.join(missing)}.")
    return versions


def run_comparison(
    image: torch.Tensor,
    category: str,
    versions: dict[str, str],
    registry: dict,
    infer: Callable[[InferenceRequest, dict], InferenceResult] = run_inference,
) -> ComparisonResult:
    """Exécute CAE puis PatchCore sur le même tenseur ``image`` [3, H, W]."""
    if set(versions) != set(METHODS):
        raise ValueError(f"Une version par méthode est attendue : {METHODS}.")

    results = {}
    for method in METHODS:
        result = infer(InferenceRequest(category, method, versions[method], image), registry)
        if (result.category, result.method, result.version) != (category, method, versions[method]):
            raise RuntimeError(f"Résultat inattendu pour {METHOD_LABELS[method]}.")
        results[method] = result

    return ComparisonResult(category=category, image_shape=tuple(image.shape), results=results)


def decisions_summary(comparison: ComparisonResult) -> str:
    """Phrase de synthèse, sans comparer les scores bruts."""
    decisions = {method: comparison[method].decision for method in METHODS}

    if "non_calibree" in decisions.values():
        return (
            "Au moins une méthode n'a pas de seuil calibré : "
            "aucune conclusion commune n'est possible."
        )
    if decisions["cae"] == decisions["patchcore"]:
        return f"Les deux méthodes concluent : {DECISION_LABELS[decisions['cae']]}."
    return (
        "Les deux méthodes ne concluent pas la même chose : "
        f"CAE → {DECISION_LABELS[decisions['cae']]}, "
        f"PatchCore → {DECISION_LABELS[decisions['patchcore']]}."
    )
