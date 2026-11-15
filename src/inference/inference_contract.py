"""Contrat d'inférence commun et modèle factice — US-06-01."""

import math

from PIL import Image
from torchvision.transforms.functional import to_pil_image

from dataclasses import dataclass
from time import perf_counter

import torch
from torch import nn

from src.inference.model_registry import get_model
from src.inference.model_loader import load_real_engine
from src.data.transforms import MVTecTransform


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
    threshold: float| None
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


def prepare_real_image(
        image: torch.Tensor,
        method: str,
) -> torch.Tensor:
    """Réutilise MVTecTransform sans modifier son implémentation."""

    if method not in ("cae", "patchcore"):
        raise ValueError(f"Méthode inconnue : {method}")

    # 1. Convertir le tenseur RGB en image PIL.
    pixels = (image.detach().cpu() * 255).round()
    pil_image = to_pil_image(
        pixels.clamp(0, 255).to(torch.uint8)
    )

    # 2. Créer un masque vide temporaire.
    dummy_mask = Image.new(
        "L",
        pil_image.size,
        color=0,
    )

    # 3. Utiliser le prétraitement existant.
    transform = MVTecTransform(
        size=(256, 256),
        profile="cae" if method == "cae" else "patchcore",
    )

    processed_image, _ = transform(
        pil_image,
        dummy_mask,
    )

    return processed_image


def run_inference(
        request: InferenceRequest,
        registry: dict,
) -> InferenceResult:
    """Exécute une inférence simulée ou réelle sur CPU."""

    config = get_model(
        registry,
        request.category,
        request.method,
        request.version,
    )

    backend = config["backend"]

    if backend not in ("fake", "cae", "patchcore"):
        raise NotImplementedError(
            f"Backend non pris en charge : {backend}"
        )

    # 1. Valider l'image avant tout chargement.
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

    # 2. Mesurer la durée de l'opération.
    start = perf_counter()

    if backend == "fake":
        # Conserver exactement le comportement US-06-01.
        model = load_fake_model("cpu")

        with torch.inference_mode():
            anomaly_map = model(
                image.to("cpu", dtype=torch.float32)
            ).detach().cpu()

            score = float(anomaly_map.mean().item())

    else:
        # 3. Charger le modèle réel (disque ou cache).
        engine = load_real_engine(
            registry,
            request.category,
            request.method,
            request.version,
        )

        # 4. Préparer l'image selon la famille du modèle.
        processed = prepare_real_image(image, backend)

        # 5. Réutiliser le moteur d'inférence existant.
        prediction = engine.predict(processed)

        score = float(prediction.score)

        anomaly_map = prediction.anomaly_map.detach().to(
            device="cpu",
            dtype=torch.float32,
        )

    # 6. Vérifier les sorties communes.
    if not math.isfinite(score):
        raise RuntimeError("Score d'anomalie non fini.")

    if (
            anomaly_map.ndim != 2
            or not torch.isfinite(anomaly_map).all()
    ):
        raise RuntimeError("Carte d'anomalie invalide.")

    # 7. Utiliser un seuil seulement s'il existe.
    raw_threshold = config["threshold"]

    threshold = (
        None
        if raw_threshold is None
        else float(raw_threshold)
    )

    if threshold is None:
        decision = "non_calibree"
    else:
        decision = (
            "suspect" if score >= threshold else "normal"
        )

    latency_ms = (perf_counter() - start) * 1000

    return InferenceResult(
        category=request.category,
        method=request.method,
        version=request.version,
        score=score,
        threshold=threshold,
        anomaly_map=anomaly_map,
        decision=decision,
        latency_ms=latency_ms,
        simulated=(backend == "fake"),
    )