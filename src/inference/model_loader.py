"""Chargement à la demande des modèles réels — US-06-03."""

import hashlib
from functools import lru_cache
from pathlib import Path

from src.inference.model_registry import get_model
from src.inference.cae_inference import CAEInference
from src.inference.patchcore_inference import PatchCoreInference


ROOT = Path(__file__).resolve().parents[2]


@lru_cache(maxsize=1)
def _load_cached(
        category: str,
        method: str,
        version: str,
        checkpoint_path: str,
        config_path: str | None,
        sha256: str,
):
    """Restaure un modèle sur CPU et conserve une seule instance."""

    checkpoint = Path(checkpoint_path)

    if not checkpoint.is_file():
        raise FileNotFoundError(
            f"Checkpoint introuvable : {checkpoint}"
        )

    if method == "cae":
        if config_path is None:
            raise ValueError("Configuration CAE manquante.")

        # Vérification d'intégrité avant restauration.
        digest = hashlib.sha256()

        with checkpoint.open("rb") as file:
            for chunk in iter(lambda: file.read(1024 * 1024), b""):
                digest.update(chunk)

        if digest.hexdigest() != sha256:
            raise ValueError("SHA-256 du checkpoint CAE incorrect.")

        return CAEInference.from_checkpoint(
            category=category,
            config_path=config_path,
            checkpoint_path=checkpoint,
            device="cpu",
        )

    if method == "patchcore":
        return PatchCoreInference.from_checkpoint(
            category=category,
            checkpoint_path=checkpoint,
            device="cpu",
            expected_sha256=sha256,
        )

    raise ValueError(f"Méthode inconnue : {method}")


def load_real_engine(
        registry: dict,
        category: str,
        method: str,
        version: str,
):
    """Sélectionne et charge le moteur demandé dans le registre."""

    config = get_model(registry, category, method, version)

    if (
            config["backend"] not in ("cae", "patchcore")
            or config["backend"] != method
    ):
        raise ValueError("Ce modèle n'est pas un backend réel valide.")

    checkpoint = ROOT / config["artifact_path"]

    config_path = None

    if method == "cae":
        config_path = str(
            (ROOT / config["config_path"]).resolve()
        )

    return _load_cached(
        category=category,
        method=method,
        version=version,
        checkpoint_path=str(checkpoint.resolve()),
        config_path=config_path,
        sha256=config["sha256"].lower(),
    )


def clear_model_cache():
    """Libère la référence conservée par le cache."""

    _load_cached.cache_clear()
