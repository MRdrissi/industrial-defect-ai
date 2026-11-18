"""Validation et conversion des images pour l'application."""

from io import BytesIO

import numpy as np
import torch
from PIL import Image, UnidentifiedImageError


MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 Mo


def decode_image(data: bytes) -> Image.Image:
    """Vérifie et convertit une image PNG/JPEG en RGB."""

    if not data:
        raise ValueError("Le fichier est vide.")

    if len(data) > MAX_FILE_SIZE:
        raise ValueError("L'image dépasse 10 Mo.")

    try:
        with Image.open(BytesIO(data)) as image:
            if image.format not in ("PNG", "JPEG"):
                raise ValueError("Format attendu : PNG ou JPEG.")

            if image.width * image.height > 16_000_000:
                raise ValueError("Dimensions de l'image trop grandes.")

            return image.convert("RGB")

    except (UnidentifiedImageError, OSError) as exc:
        raise ValueError(
            "Fichier image invalide ou endommagé."
        ) from exc


def image_to_tensor(image: Image.Image) -> torch.Tensor:
    """Convertit une image RGB en tenseur [3, H, W] dans [0, 1]."""

    pixels = np.asarray(image, dtype=np.float32) / 255.0

    return torch.from_numpy(
        pixels
    ).permute(2, 0, 1).contiguous()
