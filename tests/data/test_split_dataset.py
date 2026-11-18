import csv
import hashlib
import json
import tempfile
import unittest
from collections import Counter
from pathlib import Path

from src.data.split_dataset import (
    MANIFEST_COLUMNS,
    build_manifest,
    save_manifest,
    split_normal_images,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
REFERENCE_DIR = REPO_ROOT / "docs" / "data" / "us-02-03"
CATEGORIES = ("bottle", "hazelnut", "transistor")

EXPECTED_HASHES = {
    "bottle": "32b3c0998b3ebb1305aef61d7153bb80dc14b9c9ab4975241435dd7c8b322064",
    "hazelnut": "8f0e0d5b5bb767cd2d2e6f8a0ab10f65a959a97a38bb14bc5d9aae1d30c81fb6",
    "transistor": "eff9f514691610284c902020e62931189ffae80d20fabff1f4a6ab702d8ef728",
}

EXPECTED_COUNTS = {
    "bottle": {"train": 167, "val": 42, "test": 83},
    "hazelnut": {"train": 313, "val": 78, "test": 110},
    "transistor": {"train": 170, "val": 43, "test": 100},
}


class TestSplitDataset(unittest.TestCase):

    @staticmethod
    def touch_file(root: Path, relative_path: str):
        path = root / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()

    # --------------------------------------------------
    # 1. Déterminisme et ordre d'entrée
    # --------------------------------------------------
    def test_deterministic_split(self):
        paths = [
            f"bottle/train/good/{i:03d}.png"
            for i in range(100)
        ]

        first = split_normal_images(paths, seed=42)
        second = split_normal_images(
            list(reversed(paths)), seed=42
        )

        self.assertEqual(first, second)
        self.assertEqual(len(first["train"]), 80)
        self.assertEqual(len(first["val"]), 20)
        self.assertEqual(
            set(first["train"]) | set(first["val"]),
            set(paths),
            )
        self.assertTrue(
            set(first["train"]).isdisjoint(first["val"])
        )

    # --------------------------------------------------
    # 2. Influence de la seed
    # --------------------------------------------------
    def test_seed_changes_split(self):
        paths = [f"image_{i:03d}.png" for i in range(100)]

        first = split_normal_images(paths, seed=42)
        second = split_normal_images(paths, seed=43)

        self.assertNotEqual(first["val"], second["val"])

    # --------------------------------------------------
    # 3. Paramètres invalides
    # --------------------------------------------------
    def test_invalid_validation_ratio(self):
        paths = [f"{i}.png" for i in range(10)]

        for ratio in (-0.1, 0, 1, 1.5, float("nan")):
            with self.subTest(ratio=ratio):
                with self.assertRaises(ValueError):
                    split_normal_images(
                        paths, validation_ratio=ratio
                    )

    # --------------------------------------------------
    # 4. Entrées vides et doublons
    # --------------------------------------------------
    def test_empty_and_duplicate_inputs(self):
        with self.assertRaises(ValueError):
            split_normal_images([])

        with self.assertRaises(ValueError):
            split_normal_images(["001.png", "001.png"])

    # --------------------------------------------------
    # 5. Structure MVTec manquante
    # --------------------------------------------------
    def test_missing_dataset_structure(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(FileNotFoundError):
                build_manifest(Path(tmp), "bottle")

    # --------------------------------------------------
    # 6. Masque manquant
    # --------------------------------------------------
    def test_missing_anomaly_mask(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)

            self.touch_file(
                root, "bottle/train/good/000.png"
            )
            self.touch_file(
                root, "bottle/test/broken_large/000.png"
            )

            with self.assertRaises(FileNotFoundError):
                build_manifest(root, "bottle")

    # --------------------------------------------------
    # 7. Génération d'un manifeste fictif complet
    # --------------------------------------------------
    def test_manifest_generation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "mvtec"

            for i in range(10):
                self.touch_file(
                    root, f"bottle/train/good/{i:03d}.png"
                )

            self.touch_file(
                root, "bottle/test/good/000.png"
            )
            self.touch_file(
                root, "bottle/test/broken_large/001.png"
            )
            self.touch_file(
                root,
                "bottle/ground_truth/broken_large/001_mask.png",
            )

            rows = build_manifest(root, "bottle")

            counts = Counter(r["split"] for r in rows)

            self.assertEqual(
                dict(counts),
                {"train": 8, "val": 2, "test": 2},
            )

            anomaly = next(r for r in rows if r["label"] == 1)

            self.assertEqual(
                anomaly["mask_path"],
                "bottle/ground_truth/broken_large/001_mask.png",
            )

            output = Path(tmp) / "manifest.csv"
            save_manifest(rows, output)

            with output.open(
                    encoding="utf-8", newline=""
            ) as f:
                reader = csv.DictReader(f)
                self.assertEqual(
                    reader.fieldnames, MANIFEST_COLUMNS
                )
                saved = list(reader)

            self.assertEqual(len(saved), 12)
            self.assertEqual(
                len({r["image_path"] for r in saved}),
                12,
            )

    # --------------------------------------------------
    # 8. Configuration du protocole
    # --------------------------------------------------
    def test_protocol_configuration(self):
        path = REFERENCE_DIR / "split_protocol.json"

        protocol = json.loads(path.read_text(encoding="utf-8"))

        self.assertEqual(protocol["seed"], 42)
        self.assertEqual(protocol["validation_ratio"], 0.20)
        self.assertEqual(
            set(protocol["categories"]), set(CATEGORIES)
        )
        self.assertEqual(
            protocol["manifest_columns"], MANIFEST_COLUMNS
        )

        for category in CATEGORIES:
            self.assertEqual(
                protocol["manifests"][category]["sha256"],
                EXPECTED_HASHES[category],
            )

    # --------------------------------------------------
    # 9. Reproduction exacte des manifestes Kaggle
    # --------------------------------------------------
    def test_reproduce_all_kaggle_manifests(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "mvtec"
            output_dir = Path(tmp) / "generated"
            total_images = 0

            # Reconstruire l'arborescence à partir des
            # chemins officiels des manifestes.
            for category in CATEGORIES:
                reference = (
                        REFERENCE_DIR / f"{category}_manifest.csv"
                )

                with reference.open(
                        encoding="utf-8", newline=""
                ) as f:
                    reader = csv.DictReader(f)
                    self.assertEqual(
                        reader.fieldnames, MANIFEST_COLUMNS
                    )
                    rows = list(reader)

                paths = [r["image_path"] for r in rows]

                self.assertEqual(len(paths), len(set(paths)))

                for row in rows:
                    self.assertEqual(row["category"], category)
                    self.assertIn(
                        row["split"], {"train", "val", "test"}
                    )

                    image_path = row["image_path"]

                    if row["split"] in {"train", "val"}:
                        self.assertTrue(image_path.startswith(
                            f"{category}/train/good/"
                        ))
                        self.assertEqual(row["label"], "0")
                        self.assertEqual(row["mask_path"], "")
                    else:
                        self.assertTrue(image_path.startswith(
                            f"{category}/test/"
                        ))

                    self.touch_file(root, image_path)

                    if row["mask_path"]:
                        self.assertEqual(row["split"], "test")
                        self.assertEqual(row["label"], "1")
                        self.touch_file(
                            root, row["mask_path"]
                        )

                total_images += len(rows)

            self.assertEqual(total_images, 1106)

            # Régénérer les CSV avec le module local.
            for category in CATEGORIES:
                with self.subTest(category=category):

                    rows = build_manifest(
                        root,
                        category,
                        seed=42,
                        validation_ratio=0.20,
                    )

                    counts = Counter(
                        row["split"] for row in rows
                    )

                    self.assertEqual(
                        dict(counts), EXPECTED_COUNTS[category]
                    )

                    all_paths = [
                        row["image_path"] for row in rows
                    ]

                    self.assertEqual(
                        len(all_paths), len(set(all_paths))
                    )

                    train = {
                        r["image_path"] for r in rows
                        if r["split"] == "train"
                    }
                    val = {
                        r["image_path"] for r in rows
                        if r["split"] == "val"
                    }
                    test = {
                        r["image_path"] for r in rows
                        if r["split"] == "test"
                    }

                    self.assertTrue(train.isdisjoint(val))
                    self.assertTrue(train.isdisjoint(test))
                    self.assertTrue(val.isdisjoint(test))

                    generated = (
                            output_dir / f"{category}_manifest.csv"
                    )

                    save_manifest(rows, generated)

                    generated_hash = hashlib.sha256(
                        generated.read_bytes()
                    ).hexdigest()

                    reference = (
                            REFERENCE_DIR
                            / f"{category}_manifest.csv"
                    )
                    reference_hash = hashlib.sha256(
                        reference.read_bytes()
                    ).hexdigest()

                    self.assertEqual(
                        reference_hash, EXPECTED_HASHES[category]
                    )
                    self.assertEqual(
                        generated_hash, reference_hash
                    )


if __name__ == "__main__":
    unittest.main()
