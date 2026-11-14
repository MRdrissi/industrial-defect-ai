"""
Preuves de l'expérience patchcore-transistor/EXP-001 (US-04-03, IDL-48 / IDL-49).

Vérifie, sans dataset, GPU, Anomalib ni checkpoint :
  - la Memory Bank n'a été construite qu'avec les 170 normales Train du manifeste ;
  - les rapports GPU / CPU, la fiche d'expérience et la configuration concordent ;
  - la prédiction de référence (NPZ) correspond au rapport ;
  - la restauration CPU respecte les tolérances déclarées.

Lancement : python -m unittest discover -s tests/experiments -p "test_*.py" -v
"""

import csv
import hashlib
import json
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = ROOT / "docs/data/us-04-03"
MANIFEST = ROOT / "docs/data/us-02-03/transistor_manifest.csv"
CONFIG = ROOT / "experiments/configs/patchcore_transistor.json"
RUN = ROOT / "experiments/runs/patchcore-transistor/EXP-001.json"

CHECKPOINT_SHA256 = "7917810678f656324236a0df6db639d7de93176a7a8ae9ef06e72f3bf0a57191"
GIT_COMMIT = "76589696aabc801f346a0759f5d4cd04efcd0d70"


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def sha256(path):
    """Empreinte du contenu canonique (fins de ligne LF, comme sur Kaggle et GitHub)."""
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def manifest_paths(split):
    with MANIFEST.open(encoding="utf-8", newline="") as file:
        return [row["image_path"] for row in csv.DictReader(file) if row["split"] == split]


class TestPatchCoreTransistorExp001(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.report = read_json(EVIDENCE / "memory_bank_report.json")
        cls.restore = read_json(EVIDENCE / "restore_report.json")
        cls.registry = read_json(RUN)
        cls.config = read_json(CONFIG)

    def test_memory_bank_uses_only_normal_train_images(self):
        train = manifest_paths("train")
        used = self.report["train_images"]

        self.assertEqual(len(train), 170)
        self.assertEqual(len(used), self.report["train_image_count"])
        self.assertEqual(len(set(used)), len(used))
        self.assertEqual(set(used), set(train))
        self.assertTrue(set(used).isdisjoint(manifest_paths("val")))
        self.assertTrue(set(used).isdisjoint(manifest_paths("test")))

    def test_validation_image_is_outside_the_bank(self):
        image = self.report["validation_image"]

        self.assertIn(image, manifest_paths("val"))
        self.assertNotIn(image, self.report["train_images"])
        self.assertEqual(image, self.config["inference"]["image_path"])

    def test_shapes_follow_the_documented_configuration(self):
        n_images = self.config["memory_bank"]["sample_count"]
        ratio = self.config["memory_bank"]["sampling_ratio"]
        n_embeddings = n_images * 32 * 32

        self.assertEqual(self.report["embedding_shape"], [n_embeddings, 1536])
        self.assertEqual(self.report["memory_bank_shape"], [int(n_embeddings * ratio), 1536])
        self.assertEqual(self.report["backbone"], self.config["model"]["backbone"])
        self.assertEqual(self.report["layers"], self.config["model"]["layers"])
        self.assertEqual(self.report["coreset_ratio"], ratio)

    def test_reports_and_registry_agree(self):
        for document in (self.report, self.restore, self.registry["artifacts"]):
            self.assertEqual(document["checkpoint_sha256"], CHECKPOINT_SHA256)
        for document in (self.report, self.restore, self.registry["reproducibility"]):
            self.assertEqual(document["git_commit"], GIT_COMMIT)

        self.assertEqual(self.report["manifest_sha256"], sha256(MANIFEST))
        self.assertEqual(self.registry["reproducibility"]["manifest_sha256"], sha256(MANIFEST))
        self.assertEqual(self.registry["metrics"]["raw_score"], self.report["anomaly_score"])
        self.assertEqual(self.registry["metrics"]["memory_bank_shape"], self.report["memory_bank_shape"])
        self.assertEqual(self.restore["memory_bank_shape"], self.report["memory_bank_shape"])

    def test_reference_prediction_matches_report(self):
        with np.load(EVIDENCE / "reference_prediction.npz", allow_pickle=False) as reference:
            score = reference["pred_score"]
            anomaly_map = reference["anomaly_map"]

        self.assertEqual(score.shape, (1,))
        self.assertEqual(anomaly_map.shape, (1, 1, 256, 256))
        self.assertTrue(np.isfinite(anomaly_map).all())
        self.assertEqual(float(score[0]), self.report["anomaly_score"])
        self.assertEqual(float(score[0]), self.restore["gpu_reference_score"])

    def test_cpu_restoration_is_within_tolerances(self):
        tolerances = self.restore["tolerances"]

        self.assertEqual(self.restore["restoration_status"], "PASS")
        self.assertTrue(self.restore["strict_state_dict_loading"])
        self.assertFalse(self.restore["memory_bank_recomputed"])
        self.assertAlmostEqual(
            abs(self.restore["cpu_restored_score"] - self.restore["gpu_reference_score"]),
            self.restore["score_absolute_difference"],
            places=9,
        )
        self.assertLessEqual(self.restore["score_absolute_difference"], tolerances["score_absolute"])
        self.assertLessEqual(self.restore["map_max_absolute_difference"], tolerances["map_absolute"])
        self.assertLessEqual(
            self.restore["map_max_error_relative_to_range"], tolerances["map_relative_to_range"]
        )
        self.assertEqual(self.registry["restoration"]["restoration_status"], "PASS")


if __name__ == "__main__":
    unittest.main()
