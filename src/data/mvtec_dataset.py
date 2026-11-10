import csv
from pathlib import Path
from typing import Callable

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms.functional import pil_to_tensor

from src.data.split_dataset import MANIFEST_COLUMNS


class MVTecDataset(Dataset):
    """Dataset MVTec AD basé sur les manifestes figés de US-02-03."""

    CATEGORIES = ("bottle", "transistor", "hazelnut")
    SPLITS = ("train", "val", "test")

    def __init__(
            self,
            dataset_root: str | Path,
            manifest_path: str | Path,
            category: str,
            split: str,
            transform: Callable | None = None,
    ):
        if category not in self.CATEGORIES:
            raise ValueError(f"Catégorie invalide : {category}")

        if split not in self.SPLITS:
            raise ValueError(f"Split invalide : {split}")

        self.dataset_root = Path(dataset_root)
        self.manifest_path = Path(manifest_path)
        self.category = category
        self.split = split
        self.transform = transform

        self.samples = self._load_manifest()

    def _load_manifest(self):
        samples = []

        with self.manifest_path.open(
                "r", encoding="utf-8", newline=""
        ) as file:
            reader = csv.DictReader(file)

            if tuple(reader.fieldnames or ()) != tuple(MANIFEST_COLUMNS):
                raise ValueError("Schéma du manifeste invalide")

            for line_number, row in enumerate(reader, start=2):
                if row["category"] != self.category:
                    raise ValueError(
                        f"Ligne {line_number}: catégorie inattendue"
                    )

                if row["split"] not in self.SPLITS:
                    raise ValueError(
                        f"Ligne {line_number}: split invalide"
                    )

                if row["label"] not in ("0", "1"):
                    raise ValueError(
                        f"Ligne {line_number}: label invalide"
                    )

                label = int(row["label"])

                if row["split"] in ("train", "val") and label != 0:
                    raise ValueError(
                        f"Ligne {line_number}: anomalie hors test"
                    )

                if label == 1 and not row["mask_path"]:
                    raise ValueError(
                        f"Ligne {line_number}: masque manquant"
                    )

                if label == 0 and row["mask_path"]:
                    raise ValueError(
                        f"Ligne {line_number}: masque inattendu"
                    )

                if row["split"] == self.split:
                    samples.append({
                        **row,
                        "label": label,
                    })

        if not samples:
            raise ValueError(
                f"Aucun échantillon pour {self.category}/{self.split}"
            )

        return samples

    def __len__(self):
        return len(self.samples)

    def _resolve_path(self, relative_path: str) -> Path:
        if not relative_path or "\\" in relative_path:
            raise ValueError("Chemin de manifeste invalide")

        path = Path(relative_path)

        if path.is_absolute() or ".." in path.parts:
            raise ValueError("Chemin hors du dataset interdit")

        root = self.dataset_root.resolve()
        resolved = (root / path).resolve()

        if not resolved.is_relative_to(root):
            raise ValueError("Chemin hors de la racine MVTec")

        return resolved

    def __getitem__(self, index):
        sample = self.samples[index]

        image_path = self._resolve_path(sample["image_path"])

        with Image.open(image_path) as source:
            image = source.convert("RGB")

        if sample["label"] == 1:
            mask_path = self._resolve_path(sample["mask_path"])

            with Image.open(mask_path) as source:
                if source.size != image.size:
                    raise ValueError(
                        f"Dimensions incompatibles : {mask_path}"
                    )

                mask = source.convert("L")
                mask = mask.point(
                    lambda pixel: 255 if pixel > 127 else 0
                )
        else:
            mask = Image.new("L", image.size, color=0)

        if self.transform is not None:
            image_tensor, mask_tensor = self.transform(image, mask)
        else:
            image_tensor = pil_to_tensor(image).float() / 255.0
            mask_tensor = pil_to_tensor(mask)

        if not isinstance(image_tensor, torch.Tensor):
            raise TypeError("La transformation doit retourner un Tensor")

        if not isinstance(mask_tensor, torch.Tensor):
            raise TypeError("Le masque doit être un Tensor")

        if image_tensor.ndim != 3 or image_tensor.shape[0] != 3:
            raise ValueError("Image attendue : [3, H, W]")

        if mask_tensor.ndim != 3 or mask_tensor.shape[0] != 1:
            raise ValueError("Masque attendu : [1, H, W]")

        if image_tensor.shape[1:] != mask_tensor.shape[1:]:
            raise ValueError("Image et masque non alignés en dimensions")

        if not torch.all(
                (mask_tensor == 0) | (mask_tensor == 1) |
                (mask_tensor == 255)
        ):
            raise ValueError("Masque non binaire")

        mask_tensor = mask_tensor.bool()

        return {
            "image": image_tensor,
            "mask": mask_tensor,
            "label": sample["label"],
            "category": sample["category"],
            "defect_type": sample["defect_type"],
            "image_path": sample["image_path"],
        }
