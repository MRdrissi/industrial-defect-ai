import csv
import random
from collections.abc import Iterable
from pathlib import Path


DEFAULT_SEED = 42
DEFAULT_VALIDATION_RATIO = 0.20


def split_normal_images(
        image_paths: Iterable[str],
        seed: int = DEFAULT_SEED,
        validation_ratio: float = DEFAULT_VALIDATION_RATIO,
) -> dict[str, list[str]]:
    """
    Sépare des images normales en Train et Validation.

    Le découpage est déterministe pour une même liste
    de chemins, une même seed et un même ratio.
    """

    # 1. Trier les chemins pour fixer l'ordre initial
    images = sorted(image_paths)

    # 2. Valider les paramètres
    if not 0 < validation_ratio < 1:
        raise ValueError("validation_ratio doit être entre 0 et 1")

    if not images:
        raise ValueError("La liste des images est vide")

    if len(images) != len(set(images)):
        raise ValueError("La liste contient des doublons")

    # 3. Mélanger avec une seed fixe
    shuffled = images.copy()
    rng = random.Random(seed)
    rng.shuffle(shuffled)

    # 4. Calculer le nombre d'images de validation
    n_val = round(len(shuffled) * validation_ratio)

    # 5. Construire les deux ensembles
    val_images = sorted(shuffled[:n_val])
    train_images = sorted(shuffled[n_val:])

    return {
        "train": train_images,
        "val": val_images,
    }


MANIFEST_COLUMNS = [
    "category",
    "split",
    "image_path",
    "label",
    "defect_type",
    "mask_path",
]


def build_manifest(
        dataset_root: Path,
        category: str,
        seed: int = DEFAULT_SEED,
        validation_ratio: float = DEFAULT_VALIDATION_RATIO,
) -> list[dict]:
    """Construit le manifeste d'une catégorie MVTec AD."""

    dataset_root = Path(dataset_root)
    category_dir = dataset_root / category

    train_dir = category_dir / "train" / "good"
    test_dir = category_dir / "test"

    if not train_dir.is_dir() or not test_dir.is_dir():
        raise FileNotFoundError(
            f"Structure MVTec AD introuvable : {category}"
        )

    def relative_path(path: Path) -> str:
        return path.relative_to(dataset_root).as_posix()

    # 1. Récupérer les images normales originales
    normal_images = [
        relative_path(path)
        for path in train_dir.glob("*.png")
    ]

    # 2. Générer les mêmes splits que Kaggle
    splits = split_normal_images(
        normal_images,
        seed=seed,
        validation_ratio=validation_ratio,
    )

    rows = []

    # 3. Ajouter Train et Validation
    for split in ["train", "val"]:
        for image_path in splits[split]:
            rows.append({
                "category": category,
                "split": split,
                "image_path": image_path,
                "label": 0,
                "defect_type": "good",
                "mask_path": "",
            })

    # 4. Ajouter le test officiel, sans le modifier
    for defect_dir in sorted(test_dir.iterdir()):
        if not defect_dir.is_dir():
            continue

        defect_type = defect_dir.name
        is_anomaly = defect_type != "good"

        for image_file in sorted(defect_dir.glob("*.png")):
            mask_path = ""

            if is_anomaly:
                mask_file = (
                        category_dir
                        / "ground_truth"
                        / defect_type
                        / f"{image_file.stem}_mask.png"
                )

                if not mask_file.is_file():
                    raise FileNotFoundError(
                        f"Masque manquant : {mask_file}"
                    )

                mask_path = relative_path(mask_file)

            rows.append({
                "category": category,
                "split": "test",
                "image_path": relative_path(image_file),
                "label": int(is_anomaly),
                "defect_type": defect_type,
                "mask_path": mask_path,
            })

    return rows


def save_manifest(rows: list[dict], output_path: Path) -> None:
    """Écrit un manifeste CSV de façon déterministe."""

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open(
            "w",
            newline="",
            encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=MANIFEST_COLUMNS,
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)
