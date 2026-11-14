"""Integrity checks for the archived Hazelnut EXP-001 evidence, not model validation."""
import csv
import hashlib
import json
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = ROOT / "docs/data/us-04-04"


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


class HazelnutArchiveTests(unittest.TestCase):
    def test_manifest_split_contract(self):
        provenance = read_json(EVIDENCE / "provenance.json")
        path = ROOT / "docs/data/us-02-03/hazelnut_manifest.csv"
        self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(),
                         provenance["manifest_sha256"])
        with path.open(encoding="utf-8", newline="") as stream:
            rows = list(csv.DictReader(stream))
        splits = {}
        for split, count in (("train", 313), ("val", 78), ("test", 110)):
            selected = [r for r in rows if r["split"] == split]
            splits[split] = {r["image_path"] for r in selected}
            self.assertEqual(len(selected), count)
            self.assertEqual(len(splits[split]), count)
            self.assertTrue(all(r["category"] == "hazelnut" for r in selected))
            if split != "test":
                self.assertTrue(all(r["label"] == "0" for r in selected))
        for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
            self.assertFalse(splits[a] & splits[b])
        reference = read_json(EVIDENCE / "inference_report.json")["reference_image"]
        self.assertIn(reference, splits["val"])

    def test_original_artifact_hashes_and_prediction(self):
        provenance = read_json(EVIDENCE / "provenance.json")
        for name, digest in provenance["evidence_sha256"].items():
            with self.subTest(artifact=name):
                self.assertEqual(hashlib.sha256((EVIDENCE / name).read_bytes()).hexdigest(),
                                 digest)
        report = read_json(EVIDENCE / "inference_report.json")
        path = EVIDENCE / "reference_prediction.npz"
        self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(),
                         report["reference_npz_sha256"])
        with np.load(path, allow_pickle=False) as prediction:
            score, anomaly_map = prediction["pred_score"], prediction["anomaly_map"]
            self.assertEqual(score.shape, (1,))
            self.assertEqual(anomaly_map.shape, (1, 1, 256, 256))
            self.assertEqual(score.dtype, np.dtype("float32"))
            self.assertEqual(anomaly_map.dtype, np.dtype("float32"))
            self.assertTrue(np.isfinite(score).all())
            self.assertTrue(np.isfinite(anomaly_map).all())
            self.assertEqual(float(score.item()), report["pred_score"])
            self.assertEqual(float(anomaly_map.min()), report["anomaly_map_min"])
            self.assertEqual(float(anomaly_map.max()), report["anomaly_map_max"])
            self.assertEqual(str(prediction["image_path"].item()), report["reference_image"])

    def test_reports_are_consistent_but_not_scientific_acceptance(self):
        coreset = read_json(EVIDENCE / "coreset_report.json")
        gpu = read_json(EVIDENCE / "inference_report.json")
        cpu = read_json(EVIDENCE / "restore_report.json")
        run = read_json(ROOT / "experiments/runs/patchcore-hazelnut/EXP-001.json")
        self.assertEqual(coreset["input_shape"], [313 * 1024, 1536])
        self.assertEqual(coreset["memory_bank_shape"],
                         [int(coreset["input_shape"][0] * coreset["sampling_ratio"]), 1536])
        self.assertEqual(coreset["memory_bank_sha256"], gpu["memory_bank_sha256"])
        self.assertEqual(cpu["memory_bank_sha256"], gpu["memory_bank_sha256"])
        self.assertEqual(cpu["checkpoint_sha256"], gpu["checkpoint_sha256"])
        self.assertEqual(cpu["reference_image"], gpu["reference_image"])
        self.assertEqual(cpu["gpu_score"], gpu["pred_score"])
        self.assertAlmostEqual(abs(cpu["cpu_score"] - cpu["gpu_score"]),
                               cpu["score_abs_diff"], places=12)
        span = gpu["anomaly_map_max"] - gpu["anomaly_map_min"]
        self.assertGreater(span, 0)
        self.assertAlmostEqual(cpu["map_max_abs_diff"] / span,
                               cpu["map_relative_max_diff"], places=12)
        for metric, tolerance in (("score_abs_diff", "score_absolute"),
                                  ("map_max_abs_diff", "map_max_absolute"),
                                  ("map_relative_max_diff", "map_relative_max")):
            self.assertGreaterEqual(cpu[metric], 0)
            self.assertLessEqual(cpu[metric], cpu["tolerances"][tolerance])
        self.assertEqual(cpu["status"], "PASS")
        self.assertTrue(cpu["strict_load"])
        self.assertFalse(cpu["coreset_recomputed"])
        self.assertEqual(run["scientific_acceptance"], "not_validated")
        self.assertEqual(run["status"], "archived_with_protocol_limitation")

    def test_notebook_scientific_code_preserved(self):
        provenance = read_json(EVIDENCE / "provenance.json")
        for kind, export in provenance["exports"].items():
            with self.subTest(notebook=kind):
                nb = read_json(ROOT / export["repository_notebook"])
                code_cells = [c for c in nb["cells"] if c["cell_type"] == "code"]
                sources = ["".join(c["source"]) for c in code_cells]
                digest = hashlib.sha256(json.dumps(
                    sources, ensure_ascii=False, separators=(",", ":")
                ).encode()).hexdigest()
                self.assertEqual(digest, export["code_cells_sha256"])
                self.assertEqual(len(sources), export["code_cell_count"])
                self.assertEqual(sum(len(c.get("outputs", [])) for c in code_cells),
                                 export["output_count"])
                self.assertIn("Protocole non validé", "".join(nb["cells"][0]["source"]))
                self.assertIsNone(export["kaggle_version_url"])
        # The CPU export does not preserve restoration execution outputs.
        cpu = read_json(ROOT / provenance["exports"]["cpu"]["repository_notebook"])
        cells = [c for c in cpu["cells"] if c["cell_type"] == "code"]
        self.assertTrue(all(not c["outputs"] for c in cells[:5]))


if __name__ == "__main__":
    unittest.main()
