"""Contrat d'inférence commun et modèle factice — US-06-01."""

from dataclasses import dataclass
from time import perf_counter

import torch
from torch import nn

from src.inference.model_registry import get_model


@dataclass(frozen=True)
class InferenceRequest:
    """Entrée : modèle choisi et image RGB [3, H, W]."""

    category: str
    method: str
    version: str
    image: torch.Tensor


@dataclass(frozen=True)
class InferenceResult:
    """Sortie commune, indépendante de la méthode."""

    category: str
    method: str
    version: str
    score: float
    threshold: float
    anomaly_map: torch.Tensor
    decision: str
    latency_ms: float
    simulated: bool

    def to_dict(self) -> dict:
        """Métadonnées sérialisables sans le tenseur complet."""
        return {
            "category": self.category,
            "method": self.method,
            "version": self.version,
            "score": self.score,
            "threshold": self.threshold,
            "anomaly_map_shape": list(self.anomaly_map.shape),
            "decision": self.decision,
            "latency_ms": self.latency_ms,
            "simulated": self.simulated,
        }


class FakeModel(nn.Module):
    """Modèle déterministe pour tester le contrat."""

    def __init__(self):
        super().__init__()
        self.register_buffer(
            "reference",
            torch.tensor(0.5, dtype=torch.float32),
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        # Écart moyen à une valeur de référence fictive.
        # Image [3, H, W] -> carte [H, W].
        return (image - self.reference).abs().mean(dim=0)


def load_fake_model(device: str = "cpu") -> FakeModel:
    """Charge le modèle factice uniquement sur CPU."""
    if device != "cpu":
        raise ValueError("Le modèle factice utilise uniquement le CPU.")

    return FakeModel().to("cpu").eval()


def run_inference(request: InferenceRequest, registry: dict) -> InferenceResult:
    """Exécute une inférence simulée conforme au contrat."""

    config = get_model(
        registry,
        request.category,
        request.method,
        request.version,
    )

    if config["backend"] != "fake":
        raise NotImplementedError(
            "Ce backend sera intégré dans US-06-03."
        )

    image = request.image

    if not isinstance(image, torch.Tensor):
        raise TypeError("L'image doit être un tenseur PyTorch.")

    if (
            image.ndim != 3
            or image.shape[0] != 3
            or image.shape[1] == 0
            or image.shape[2] == 0
    ):
        raise ValueError("Image RGB attendue : [3, H, W].")

    if not torch.is_floating_point(image):
        raise TypeError("L'image doit être de type flottant.")

    if not torch.isfinite(image).all():
        raise ValueError("L'image contient NaN ou Inf.")

    if image.min() < 0 or image.max() > 1:
        raise ValueError("Les pixels doivent être dans [0, 1].")

    start = perf_counter()

    model = load_fake_model("cpu")

    with torch.inference_mode():
        anomaly_map = model(
            image.to("cpu", dtype=torch.float32)
        ).detach().cpu()

        score = float(anomaly_map.mean().item())

    latency_ms = (perf_counter() - start) * 1000
    threshold = float(config["threshold"])

    return InferenceResult(
        category=request.category,
        method=request.method,
        version=request.version,
        score=score,
        threshold=threshold,
        anomaly_map=anomaly_map,
        decision="suspect" if score >= threshold else "normal",
        latency_ms=latency_ms,
        simulated=True,
    )
