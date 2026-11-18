from dataclasses import dataclass
from typing import Literal

import torch
from PIL import Image
from torchvision.transforms.functional import pil_to_tensor, normalize


@dataclass(frozen=True)
class MVTecTransform:
    """
    Transformations communes aux images et aux masques MVTec AD.

    size : (hauteur, largeur)
    profile : "cae" ou "patchcore"
    """

    size: tuple[int, int] = (256, 256)
    profile: Literal["cae", "patchcore"] = "cae"

    # Valeurs ImageNet par défaut, configurables selon le backbone.
    mean: tuple[float, float, float] = (0.485, 0.456, 0.406)
    std: tuple[float, float, float] = (0.229, 0.224, 0.225)

    def __post_init__(self):
        if (
                len(self.size) != 2
                or any(type(v) is not int or v <= 0 for v in self.size)
        ):
            raise ValueError("size doit être (hauteur, largeur) positive")

        if self.profile not in ("cae", "patchcore"):
            raise ValueError("Profil inconnu")

        if len(self.mean) != 3 or len(self.std) != 3:
            raise ValueError("mean et std doivent contenir 3 valeurs")

        if any(value <= 0 for value in self.std):
            raise ValueError("Les écarts-types doivent être positifs")

    def __call__(
            self,
            image: Image.Image,
            mask: Image.Image,
    ) -> tuple[torch.Tensor, torch.Tensor]:

        if image.size != mask.size:
            raise ValueError("Image et masque de dimensions différentes")

        height, width = self.size

        # L'image RGB est interpolée en bilinéaire.
        image = image.convert("RGB").resize(
            (width, height),
            resample=Image.Resampling.BILINEAR
        )

        # Le masque utilise le plus proche voisin.
        mask = mask.convert("L").resize(
            (width, height),
            resample=Image.Resampling.NEAREST
        )

        # Image float32 dans [0, 1].
        image_tensor = pil_to_tensor(image).float() / 255.0

        # Masque booléen : 0 = normal, 1 = anomalie.
        mask_tensor = pil_to_tensor(mask) > 127

        if self.profile == "patchcore":
            image_tensor = normalize(
                image_tensor,
                mean=self.mean,
                std=self.std,
            )

        return image_tensor, mask_tensor
