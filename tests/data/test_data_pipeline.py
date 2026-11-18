import csv
import tempfile
import unittest
from pathlib import Path

import torch
from PIL import Image, ImageDraw

from src.data.mvtec_dataset import MVTecDataset
from src.data.split_dataset import MANIFEST_COLUMNS
from src.data.transforms import MVTecTransform
from src.data.dataloaders import create_dataloader


class TestMVTecDataset(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)

        self.root = Path(self.temp_dir.name)
        self.dataset_root = self.root / "mvtec"
        self.manifest_path = self.root / "bottle_manifest.csv"

        normal_path = self.dataset_root / "bottle/train/good/000.png"
        abnormal_path = self.dataset_root / "bottle/test/crack/001.png"
        mask_path = self.dataset_root / "bottle/ground_truth/crack/001_mask.png"

        for path in (normal_path, abnormal_path, mask_path):
            path.parent.mkdir(parents=True, exist_ok=True)

        Image.new("RGB", (8, 6), (255, 0, 0)).save(normal_path)
        Image.new("RGB", (8, 6), (0, 255, 0)).save(abnormal_path)

        mask = Image.new("L", (8, 6), 0)
        ImageDraw.Draw(mask).rectangle((2, 1, 4, 3), fill=255)
        mask.save(mask_path)

        rows = [
            {
                "category": "bottle",
                "split": "train",
                "image_path": "bottle/train/good/000.png",
                "label": "0",
                "defect_type": "good",
                "mask_path": "",
            },
            {
                "category": "bottle",
                "split": "test",
                "image_path": "bottle/test/crack/001.png",
                "label": "1",
                "defect_type": "crack",
                "mask_path": "bottle/ground_truth/crack/001_mask.png",
            },
        ]

        with self.manifest_path.open(
                "w", encoding="utf-8", newline=""
        ) as file:
            writer = csv.DictWriter(
                file, fieldnames=MANIFEST_COLUMNS
            )
            writer.writeheader()
            writer.writerows(rows)

        self.mask_path = mask_path

    def dataset(self, split):
        return MVTecDataset(
            dataset_root=self.dataset_root,
            manifest_path=self.manifest_path,
            category="bottle",
            split=split,
        )

    def test_normal_image(self):
        sample = self.dataset("train")[0]

        self.assertEqual(sample["label"], 0)
        self.assertEqual(sample["image"].shape, (3, 6, 8))
        self.assertEqual(sample["mask"].shape, (1, 6, 8))

        self.assertEqual(sample["image"].dtype, torch.float32)
        self.assertEqual(sample["mask"].dtype, torch.bool)
        self.assertEqual(sample["mask"].sum().item(), 0)

        self.assertTrue(
            torch.all(sample["image"][0] == 1.0)
        )

    def test_anomalous_image(self):
        sample = self.dataset("test")[0]

        self.assertEqual(sample["label"], 1)
        self.assertEqual(sample["defect_type"], "crack")
        self.assertEqual(sample["mask"].sum().item(), 9)

        self.assertTrue(sample["mask"][0, 2, 3].item())
        self.assertFalse(sample["mask"][0, 0, 0].item())

        self.assertTrue(
            torch.all(sample["image"][1] == 1.0)
        )

    def test_missing_anomaly_mask(self):
        self.mask_path.unlink()

        with self.assertRaises(FileNotFoundError):
            _ = self.dataset("test")[0]

    def test_incompatible_mask_dimensions(self):
        Image.new("L", (4, 4), 255).save(self.mask_path)

        with self.assertRaises(ValueError):
            _ = self.dataset("test")[0]


    def test_cae_resize_and_mask_alignment(self):
        transform = MVTecTransform(
            size=(12, 16),
            profile="cae",
        )

        dataset = MVTecDataset(
            dataset_root=self.dataset_root,
            manifest_path=self.manifest_path,
            category="bottle",
            split="test",
            transform=transform,
        )

        sample = dataset[0]

        self.assertEqual(sample["image"].shape, (3, 12, 16))
        self.assertEqual(sample["mask"].shape, (1, 12, 16))

        # Rectangle original 3x3, agrandi par facteur 2.
        self.assertEqual(sample["mask"].sum().item(), 36)

        self.assertTrue(sample["mask"][0, 4, 6].item())
        self.assertFalse(sample["mask"][0, 0, 0].item())

        self.assertEqual(sample["mask"].dtype, torch.bool)

        # Le CAE travaille ici avec RGB dans [0,1].
        self.assertGreaterEqual(sample["image"].min().item(), 0)
        self.assertLessEqual(sample["image"].max().item(), 1)

    def test_patchcore_normalization(self):
        transform = MVTecTransform(
            size=(6, 8),
            profile="patchcore",
        )

        dataset = MVTecDataset(
            dataset_root=self.dataset_root,
            manifest_path=self.manifest_path,
            category="bottle",
            split="test",
            transform=transform,
        )

        sample = dataset[0]

        # L'image synthétique est entièrement verte (0,255,0).
        expected = torch.tensor([
            (0.0 - 0.485) / 0.229,
            (1.0 - 0.456) / 0.224,
            (0.0 - 0.406) / 0.225,
            ])

        torch.testing.assert_close(
            sample["image"][:, 0, 0],
            expected,
            rtol=1e-5,
            atol=1e-5,
        )

        # La normalisation ne doit pas modifier le masque.
        self.assertEqual(sample["mask"].sum().item(), 9)
        self.assertEqual(sample["mask"].dtype, torch.bool)

    def test_invalid_transform_config(self):
        with self.assertRaises(ValueError):
            MVTecTransform(size=(0, 256))

        with self.assertRaises(ValueError):
            MVTecTransform(profile="unknown")

        with self.assertRaises(ValueError):
            MVTecTransform(std=(0.229, 0, 0.225))


    def test_dataloader_batch_contract(self):
        transform = MVTecTransform(
            size=(12, 16),
            profile="cae",
        )

        dataset = MVTecDataset(
            dataset_root=self.dataset_root,
            manifest_path=self.manifest_path,
            category="bottle",
            split="test",
            transform=transform,
        )

        loader = create_dataloader(
            dataset,
            batch_size=2,
            seed=42,
        )

        batch = next(iter(loader))

        # Le test synthétique contient une seule anomalie.
        self.assertEqual(batch["image"].shape, (1, 3, 12, 16))
        self.assertEqual(batch["mask"].shape, (1, 1, 12, 16))

        self.assertEqual(batch["image"].dtype, torch.float32)
        self.assertEqual(batch["mask"].dtype, torch.bool)

        self.assertEqual(batch["label"].tolist(), [1])
        self.assertEqual(batch["category"], ["bottle"])

        self.assertEqual(
            batch["image_path"],
            ["bottle/test/crack/001.png"],
        )

    def test_dataloader_split_order_and_seed(self):
        from torch.utils.data import Dataset

        class IndexedDataset(Dataset):
            def __init__(self, split):
                self.split = split

            def __len__(self):
                return 12

            def __getitem__(self, index):
                return {"index": index}

        def get_indices(loader):
            return [
                index
                for batch in loader
                for index in batch["index"].tolist()
            ]

        # Même seed : même ordre de mélange pour train.
        train_dataset = IndexedDataset("train")

        first = get_indices(
            create_dataloader(train_dataset, batch_size=4, seed=42)
        )

        second = get_indices(
            create_dataloader(train_dataset, batch_size=4, seed=42)
        )

        self.assertEqual(first, second)
        self.assertEqual(sorted(first), list(range(12)))

        # Validation et test : ordre inchangé.
        for split in ("val", "test"):
            with self.subTest(split=split):
                dataset = IndexedDataset(split)

                indices = get_indices(
                    create_dataloader(
                        dataset,
                        batch_size=5,
                        seed=123,
                    )
                )

                self.assertEqual(indices, list(range(12)))




if __name__ == "__main__":
    unittest.main()
