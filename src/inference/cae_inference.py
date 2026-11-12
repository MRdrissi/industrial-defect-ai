"""
Inférence du Convolutional Autoencoder (US-03-05, ST-03-05-01 / IDL-35).

Principe : le CAE a appris à reconstruire uniquement des pièces normales.
Sur une pièce défectueuse, il reconstruit mal la zone du défaut. On mesure
donc l'écart entre l'image et sa reconstruction :

    image ──► CAE ──► reconstruction
      └────── écart pixel par pixel ──────► carte d'anomalie [H, W]
                                               └─► score image (un nombre)

Le score est BRUT : aucun seuil n'est appliqué ici. La décision
« normal / défaut » sera calibrée sur la validation en US-05-02.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path

import torch
from torchvision.transforms.functional import gaussian_blur

from src.models.convolutional_autoencoder import (
    CAEConfig,
    ConvolutionalAutoencoder,
)

MODEL_FAMILY = "cae"
SCORE_METHODS = ("max", "mean")


@dataclass(frozen=True)
class CAEInferenceConfig:
    """Réglages du calcul de la carte et du score (sans calibration)."""

    # Lissage gaussien de la carte : réduit le bruit isolé de reconstruction
    # (contours fins, reflets) pour que le score reflète des zones, pas des pixels.
    # 0 désactive le lissage.
    smoothing_sigma: float = 4.0

    # "max"  : score = valeur maximale de la carte lissée (sensible aux petits défauts)
    # "mean" : score = MSE moyenne de l'image (identique à la perte d'entraînement)
    score_method: str = "max"

    def __post_init__(self):
        if self.smoothing_sigma < 0:
            raise ValueError("smoothing_sigma doit être >= 0.")
        if self.score_method not in SCORE_METHODS:
            raise ValueError(f"score_method doit être dans {SCORE_METHODS}.")


@dataclass
class CAEInferenceResult:
    """
    Sortie d'inférence pour UNE image.

    Champs communs à tous les modèles (contrat d'inférence partagé, EPIC-06) :
      - category, model_family, model_name
      - anomaly_map : carte [H, W], float32, CPU, valeurs >= 0
      - score       : score brut de l'image (float, non calibré)
    Champ propre au CAE :
      - reconstruction : image reconstruite [3, H, W] dans [0, 1]
    """

    category: str
    model_name: str
    anomaly_map: torch.Tensor
    score: float
    reconstruction: torch.Tensor
    model_family: str = MODEL_FAMILY
    extras: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Version sérialisable (rapport JSON) sans les tenseurs lourds."""
        return {
            "category": self.category,
            "model_family": self.model_family,
            "model_name": self.model_name,
            "score": self.score,
            "anomaly_map_shape": list(self.anomaly_map.shape),
            **self.extras,
        }


def extract_state_dict(checkpoint: dict) -> dict:
    """
    Renvoie les poids du modèle, quel que soit le format du checkpoint.

    Deux formats coexistent dans le projet :
      - poids seuls : torch.save(model.state_dict(), ...)            (transistor)
      - dictionnaire complet : poids sous "model_state_dict" + métadonnées
        (époque, val MSE, commit…)                                    (bottle, hazelnut)
    """
    if not isinstance(checkpoint, dict):
        raise TypeError(f"Checkpoint inattendu : {type(checkpoint).__name__}")
    if "model_state_dict" in checkpoint:
        return checkpoint["model_state_dict"]
    if checkpoint and all(isinstance(v, torch.Tensor) for v in checkpoint.values()):
        return checkpoint
    raise KeyError(
        "Format de checkpoint non reconnu (ni poids seuls, ni clé 'model_state_dict'). "
        f"Clés trouvées : {sorted(checkpoint)[:10]}"
    )


def load_cae(
    config_path: str | Path,
    checkpoint_path: str | Path,
    device: str | torch.device = "cpu",
) -> tuple[ConvolutionalAutoencoder, str]:
    """
    Recrée le CAE à partir du fichier de configuration et charge ses poids.

    Accepte les deux formats de checkpoint du projet (voir extract_state_dict).
    Renvoie (modèle en mode eval, nom du modèle).
    Lève une erreur si le checkpoint ne correspond pas à l'architecture.
    """
    config_path, checkpoint_path = Path(config_path), Path(checkpoint_path)
    if not config_path.is_file():
        raise FileNotFoundError(f"Configuration introuvable : {config_path}")
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint introuvable : {checkpoint_path}")

    config_file = json.loads(config_path.read_text("utf-8"))
    params = dict(config_file["config"])
    params["image_size"] = tuple(params["image_size"])
    params["encoder_channels"] = tuple(params["encoder_channels"])

    model = ConvolutionalAutoencoder(CAEConfig(**params))
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(extract_state_dict(checkpoint))  # strict : toutes les couches doivent correspondre
    model.to(device).eval()

    return model, config_file.get("name", "cae")


class CAEInference:
    """Moteur d'inférence CAE spécialisé pour UNE catégorie."""

    def __init__(
        self,
        model: ConvolutionalAutoencoder,
        category: str,
        model_name: str = "cae",
        config: CAEInferenceConfig = CAEInferenceConfig(),
    ):
        self.model = model.eval()
        self.category = category
        self.model_name = model_name
        self.config = config
        self.device = next(model.parameters()).device

    @classmethod
    def from_checkpoint(
        cls,
        category: str,
        config_path: str | Path,
        checkpoint_path: str | Path,
        device: str | torch.device = "cpu",
        config: CAEInferenceConfig = CAEInferenceConfig(),
    ) -> "CAEInference":
        model, model_name = load_cae(config_path, checkpoint_path, device)
        return cls(model, category, model_name, config)

    @torch.inference_mode()
    def predict_batch(self, images: torch.Tensor) -> list[CAEInferenceResult]:
        """
        images : [B, 3, H, W], float dans [0, 1] (transform MVTec profil "cae").
        Renvoie un CAEInferenceResult par image.
        """
        if images.ndim != 4:
            raise ValueError("Entrée attendue : [B, 3, H, W]")
        if images.min() < 0 or images.max() > 1:
            raise ValueError(
                "Les images doivent être dans [0, 1] (profil 'cae', sans normalisation)."
            )

        images = images.to(self.device, dtype=torch.float32)
        reconstructions = self.model(images)

        # 1. Erreur quadratique par pixel, moyennée sur les 3 canaux -> [B, H, W]
        raw_maps = ((images - reconstructions) ** 2).mean(dim=1)

        # 2. Lissage gaussien (optionnel) -> carte d'anomalie
        anomaly_maps = self._smooth(raw_maps)

        # 3. Score image (brut, non calibré)
        if self.config.score_method == "max":
            scores = anomaly_maps.flatten(1).max(dim=1).values
        else:
            scores = raw_maps.flatten(1).mean(dim=1)
        mean_mse = raw_maps.flatten(1).mean(dim=1)

        results = []
        for i in range(images.shape[0]):
            results.append(
                CAEInferenceResult(
                    category=self.category,
                    model_name=self.model_name,
                    anomaly_map=anomaly_maps[i].cpu(),
                    score=float(scores[i]),
                    reconstruction=reconstructions[i].cpu(),
                    extras={
                        "score_method": self.config.score_method,
                        "smoothing_sigma": self.config.smoothing_sigma,
                        "mean_mse": float(mean_mse[i]),
                    },
                )
            )
        return results

    def predict(self, image: torch.Tensor) -> CAEInferenceResult:
        """Inférence sur une seule image [3, H, W]."""
        if image.ndim != 3:
            raise ValueError("Entrée attendue : [3, H, W]")
        return self.predict_batch(image.unsqueeze(0))[0]

    def _smooth(self, maps: torch.Tensor) -> torch.Tensor:
        sigma = self.config.smoothing_sigma
        if sigma == 0:
            return maps
        kernel_size = 2 * int(4 * sigma + 0.5) + 1  # taille impaire couvrant ±4 sigma
        return gaussian_blur(
            maps.unsqueeze(1), kernel_size=[kernel_size, kernel_size], sigma=[sigma, sigma]
        ).squeeze(1)
