"""Restauration et inférence PatchCore — US-04-05."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from anomalib.models.image.patchcore.torch_model import PatchcoreModel


def load_patchcore(
        checkpoint_path: str | Path,
        device: str | torch.device = "cpu",
        expected_sha256: str | None = None,
) -> PatchcoreModel:
    """Restaure PatchCore sans reconstruire la Memory Bank."""

    checkpoint_path = Path(checkpoint_path)

    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"Checkpoint introuvable : {checkpoint_path}"
        )

    # 1. Vérifier l'intégrité du checkpoint si demandée.
    if expected_sha256 is not None:
        digest = hashlib.sha256()

        with checkpoint_path.open("rb") as file:
            for chunk in iter(lambda: file.read(1024 * 1024), b""):
                digest.update(chunk)

        if digest.hexdigest() != expected_sha256:
            raise ValueError("SHA-256 du checkpoint incorrect")

    # 2. Lire les poids et métadonnées sur CPU.
    saved = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True,
    )

    required = {
        "model_state_dict",
        "backbone",
        "layers",
        "num_neighbors",
    }

    if not isinstance(saved, dict) or not required.issubset(saved):
        raise ValueError("Format de checkpoint PatchCore invalide")

    # 3. Reconstruire l'architecture sans télécharger de poids.
    from anomalib.models.image.patchcore.torch_model import PatchcoreModel

    model = PatchcoreModel(
        backbone=saved["backbone"],
        layers=saved["layers"],
        pre_trained=False,
        num_neighbors=saved["num_neighbors"],
    )

    # 4. Restaurer exactement les poids et la Memory Bank.
    model.load_state_dict(
        saved["model_state_dict"],
        strict=True,
    )

    # 5. Préparer le modèle pour l'inférence.
    model.feature_extractor.requires_grad_(False)
    model = model.to(device)
    model.eval()

    bank = model.memory_bank

    if bank.ndim != 2 or bank.shape[0] == 0:
        raise ValueError("Memory Bank vide ou invalide")

    if not torch.isfinite(bank).all():
        raise ValueError("Memory Bank non finie")

    return model



@dataclass
class PatchCoreInferenceResult:
    """Résultat d'inférence pour une seule image."""

    category: str
    model_name: str
    anomaly_map: torch.Tensor
    score: float

    model_family: str = "patchcore"
    extras: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Métadonnées sérialisables sans le tenseur de la carte."""
        return {
            "category": self.category,
            "model_family": self.model_family,
            "model_name": self.model_name,
            "score": self.score,
            "anomaly_map_shape": list(self.anomaly_map.shape),
            **self.extras,
        }



class PatchCoreInference:
    """Moteur d'inférence PatchCore pour une catégorie."""

    def __init__(
            self,
            model: PatchcoreModel,
            category: str,
            model_name: str = "patchcore",
    ):
        self.model = model.eval()
        self.category = category
        self.model_name = model_name
        self.device = next(model.parameters()).device

    @classmethod
    def from_checkpoint(
            cls,
            category: str,
            checkpoint_path: str | Path,
            device: str | torch.device = "cpu",
            expected_sha256: str | None = None,
    ) -> "PatchCoreInference":
        model = load_patchcore(
            checkpoint_path=checkpoint_path,
            device=device,
            expected_sha256=expected_sha256,
        )

        return cls(
            model=model,
            category=category,
        )

    @torch.inference_mode()
    def predict_batch(
            self,
            images: torch.Tensor,
    ) -> list[PatchCoreInferenceResult]:
        """Inférence sur un batch [B, 3, H, W]."""

        if not isinstance(images, torch.Tensor):
            raise TypeError("images doit être un Tensor PyTorch")

        if (
                images.ndim != 4
                or images.shape[0] == 0
                or images.shape[1] != 3
        ):
            raise ValueError(
                "Entrée attendue : [B, 3, H, W], avec B >= 1"
            )

        if not torch.is_floating_point(images):
            raise TypeError("Les images doivent être flottantes")

        if not torch.isfinite(images).all():
            raise ValueError("Les images contiennent NaN ou Inf")

        images = images.to(
            device=self.device,
            dtype=torch.float32,
        )

        # Le modèle est déjà en mode évaluation.
        prediction = self.model(images)

        scores = prediction.pred_score.reshape(-1)
        maps = prediction.anomaly_map

        # Anomalib : [B, 1, H, W] -> [B, H, W]
        if maps.ndim == 4 and maps.shape[1] == 1:
            maps = maps[:, 0]

        expected_shape = (
            images.shape[0],
            images.shape[2],
            images.shape[3],
        )

        if (
                scores.numel() != images.shape[0]
                or tuple(maps.shape) != expected_shape
        ):
            raise RuntimeError(
                "Dimensions des sorties PatchCore inattendues"
            )

        if not torch.isfinite(scores).all():
            raise RuntimeError("Scores non finis")

        if not torch.isfinite(maps).all():
            raise RuntimeError("Cartes non finies")

        results = []

        for i in range(images.shape[0]):
            results.append(
                PatchCoreInferenceResult(
                    category=self.category,
                    model_name=self.model_name,
                    score=float(scores[i].item()),
                    anomaly_map=maps[i]
                    .detach()
                    .to("cpu", dtype=torch.float32)
                    .clone(),
                    )
            )

        return results

    def predict(
            self,
            image: torch.Tensor,
    ) -> PatchCoreInferenceResult:
        """Inférence sur une image [3, H, W]."""

        if image.ndim != 3:
            raise ValueError("Entrée attendue : [3, H, W]")

        return self.predict_batch(
            image.unsqueeze(0)
        )[0]
