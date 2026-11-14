"""EXP-002 evidence and exported extraction logic; no dataset, GPU or Anomalib install.

Synthetic checks execute only inspected extraction assignments / the parity cell.
They test pooling dataflow and parity-check sensitivity, not a pretrained backbone.
The actual Anomalib 2.7.0 parity result is preserved in the Kaggle export.
"""
import ast
import contextlib
import copy
import csv
import hashlib
import io
import json
import re
import unittest
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = ROOT / "docs/data/us-04-04/EXP-002"


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def digest(value):
    return hashlib.sha256(value).hexdigest()


def gpu_cells():
    provenance = read_json(EVIDENCE / "provenance.json")
    notebook = read_json(ROOT / provenance["exports"]["gpu"]["repository_notebook"])
    return [c for c in notebook["cells"] if c["cell_type"] == "code"]


class _Features(nn.Module):
    def forward(self, images):
        return {"layer2": images[:, :2], "layer3": F.avg_pool2d(images[:, 1:], 2)}


class _SyntheticPatchcore(nn.Module):
    """Small deterministic tensor double; not the Anomalib backbone."""
    def __init__(self):
        super().__init__()
        self.feature_extractor = _Features()
        self.feature_pooler = nn.AvgPool2d(3, 1, 1)
        self.embedding_store = []

    def generate_embedding(self, features):
        first = features["layer2"]
        return torch.cat((first, F.interpolate(
            features["layer3"], size=first.shape[-2:], mode="bilinear"
        )), dim=1)

    def reshape_embedding(self, embeddings):
        return embeddings.permute(0, 2, 3, 1).reshape(-1, embeddings.shape[1])

    def forward(self, images):
        features = self.feature_extractor(images)
        features = {name: self.feature_pooler(value) for name, value in features.items()}
        result = self.reshape_embedding(self.generate_embedding(features))
        if self.training:
            self.embedding_store.append(result)
        return result


def extraction_nodes(cell, names):
    tree = ast.parse("".join(cell["source"]))
    nodes = sorted([
        n for n in ast.walk(tree) if isinstance(n, ast.Assign)
        and len(n.targets) == 1 and isinstance(n.targets[0], ast.Name)
        and n.targets[0].id in names
    ], key=lambda n: n.lineno)
    if [n.targets[0].id for n in nodes] != names:
        raise AssertionError("Exported extraction assignments changed")
    return nodes


class HazelnutEXP002Tests(unittest.TestCase):
    def test_processed_paths_match_normal_train_and_splits_are_disjoint(self):
        manifest = ROOT / "docs/data/us-02-03/hazelnut_manifest.csv"
        provenance = read_json(EVIDENCE / "provenance.json")
        self.assertEqual(digest(manifest.read_bytes()), provenance["manifest_sha256"])
        with manifest.open(encoding="utf-8", newline="") as stream:
            rows = list(csv.DictReader(stream))
        sets = {}
        for split, count in (("train", 313), ("val", 78), ("test", 110)):
            selected = [r for r in rows if r["split"] == split]
            sets[split] = {r["image_path"] for r in selected}
            self.assertEqual(len(selected), count)
            self.assertEqual(len(sets[split]), count)
            self.assertTrue(all(r["category"] == "hazelnut" for r in selected))
            if split != "test":
                self.assertTrue(all(r["label"] == "0" for r in selected))
        for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
            self.assertFalse(sets[a] & sets[b])
        report = read_json(EVIDENCE / "coreset_report.json")
        self.assertEqual(len(report["train_image_paths"]), 313)
        self.assertEqual(set(report["train_image_paths"]), sets["train"])
        reference = read_json(EVIDENCE / "inference_report.json")["reference_image"]
        self.assertIn(reference, sets["val"])

    def test_original_evidence_and_executed_cells_are_preserved(self):
        provenance = read_json(EVIDENCE / "provenance.json")
        for name, expected in provenance["evidence_sha256"].items():
            self.assertEqual(digest((EVIDENCE / name).read_bytes()), expected)
        for export in provenance["exports"].values():
            notebook = read_json(ROOT / export["repository_notebook"])
            original_cells = notebook["cells"][1:]
            canonical = json.dumps(original_cells, ensure_ascii=False, separators=(",", ":"))
            self.assertEqual(digest(canonical.encode()), export["original_cells_sha256"])
            codes = [c for c in original_cells if c["cell_type"] == "code"]
            self.assertEqual(len(codes), export["code_cell_count"])
            self.assertTrue(all(c["execution_count"] is not None and c["outputs"] for c in codes))
            self.assertFalse(any(o["output_type"] == "error" for c in codes for o in c["outputs"]))
            for cell in codes:
                ast.parse("".join(cell["source"]))

    def test_prediction_hash_values_and_restoration_tolerances(self):
        co = read_json(EVIDENCE / "coreset_report.json")
        gpu = read_json(EVIDENCE / "inference_report.json")
        cpu = read_json(EVIDENCE / "restore_report.json")
        for report in (co, gpu, cpu):
            self.assertEqual(report["experiment_id"], "EXP-002")
            self.assertEqual(report["category"], "hazelnut")
            self.assertEqual(report["memory_bank_sha256"], co["memory_bank_sha256"])
        self.assertEqual(co["input_shape"], [313 * 1024, 1536])
        self.assertEqual(co["sampling_ratio"], 0.01)
        self.assertEqual(co["memory_bank_shape"], [int(313 * 1024 * 0.01), 1536])
        self.assertEqual(cpu["checkpoint_sha256"], gpu["checkpoint_sha256"])
        self.assertEqual(cpu["reference_image"], gpu["reference_image"])
        self.assertTrue(cpu["strict_load"] and cpu["weights_only"])
        self.assertFalse(cpu["coreset_recomputed"])
        path = EVIDENCE / "reference_prediction.npz"
        self.assertEqual(digest(path.read_bytes()), gpu["reference_npz_sha256"])
        with np.load(path, allow_pickle=False) as prediction:
            score, amap = prediction["pred_score"], prediction["anomaly_map"]
            self.assertEqual(score.shape, (1,))
            self.assertEqual(amap.shape, (1, 1, 256, 256))
            self.assertEqual(score.dtype, np.dtype("float32"))
            self.assertEqual(amap.dtype, np.dtype("float32"))
            self.assertTrue(np.isfinite(score).all() and np.isfinite(amap).all())
            self.assertEqual(float(score.item()), gpu["pred_score"])
            self.assertEqual(float(amap.min()), gpu["anomaly_map_min"])
            self.assertEqual(float(amap.max()), gpu["anomaly_map_max"])
            self.assertEqual(str(prediction["image_path"].item()), gpu["reference_image"])
            span = float(amap.max() - amap.min())
        self.assertEqual(cpu["gpu_score"], gpu["pred_score"])
        self.assertAlmostEqual(abs(cpu["cpu_score"] - cpu["gpu_score"]), cpu["score_abs_diff"], places=12)
        self.assertAlmostEqual(cpu["map_max_abs_diff"] / span, cpu["map_relative_max_diff"], places=12)
        for metric, tolerance, limit in (
            ("score_abs_diff", "score_absolute", 1e-3),
            ("map_max_abs_diff", "map_max_absolute", 1e-3),
            ("map_relative_max_diff", "map_relative_max", 1e-4),
        ):
            self.assertEqual(cpu["tolerances"][tolerance], limit)
            self.assertGreaterEqual(cpu[metric], 0)
            self.assertLessEqual(cpu[metric], limit)
        self.assertEqual(cpu["status"], "PASS")

    def test_exported_extraction_pools_features_before_embedding(self):
        images = torch.arange(2 * 3 * 8 * 8, dtype=torch.float32).reshape(2, 3, 8, 8) / 100
        raw = _Features()(images)
        # Independent tensor expectation, with border effects that expose missing pooling.
        expected_maps = [F.avg_pool2d(raw[k], 3, 1, 1) for k in ("layer2", "layer3")]
        expected_map = torch.cat((expected_maps[0], F.interpolate(
            expected_maps[1], size=(8, 8), mode="bilinear"
        )), dim=1)
        expected = expected_map.permute(0, 2, 3, 1).reshape(-1, 4)
        for index, names in (
            (6, ["features", "pooled_features", "embedding_map", "embeddings"]),
            (8, ["features", "pooled_features", "feature_map", "flat_embeddings"]),
        ):
            with self.subTest(original_cell=index + 1):
                nodes = extraction_nodes(gpu_cells()[index], names)
                ns = {"model": _SyntheticPatchcore().eval(), "images": images, "current_images": images}
                exec(compile(ast.Module(body=nodes, type_ignores=[]), "<exported extraction>", "exec"), ns)
                torch.testing.assert_close(ns[names[-1]], expected, rtol=0, atol=0)
                self.assertEqual(ns["model"].embedding_store, [])
                # Mutation check: routing unpooled features must change the result.
                broken = copy.deepcopy(nodes)
                broken[2].value.args[0] = ast.Name(id="features", ctx=ast.Load())
                module = ast.fix_missing_locations(ast.Module(body=broken, type_ignores=[]))
                exec(compile(module, "<missing pooling mutation>", "exec"), ns)
                self.assertFalse(torch.allclose(ns[names[-1]], expected))

    def test_exported_parity_control_detects_mismatch_and_clears_store(self):
        source = "".join(gpu_cells()[7]["source"])
        images = torch.arange(2 * 3 * 8 * 8, dtype=torch.float32).reshape(2, 3, 8, 8) / 100
        for mismatch in (False, True):
            with self.subTest(mismatch=mismatch):
                model = _SyntheticPatchcore().eval()
                expected = model(images)
                ns = {"torch": torch, "model": model, "images": images,
                      "embeddings": expected + (1 if mismatch else 0)}
                with contextlib.redirect_stdout(io.StringIO()):
                    if mismatch:
                        with self.assertRaises(AssertionError):
                            exec(compile(source, "<exported parity control>", "exec"), ns)
                    else:
                        exec(compile(source, "<exported parity control>", "exec"), ns)
                        self.assertEqual(ns["max_difference"], 0.0)
                self.assertFalse(model.training)
                self.assertEqual(model.embedding_store, [])

    def test_configuration_registry_and_document_links(self):
        config = read_json(ROOT / "experiments/configs/patchcore_hazelnut_exp002.json")
        run = read_json(ROOT / "experiments/runs/patchcore-hazelnut/EXP-002.json")
        co = read_json(EVIDENCE / "coreset_report.json")
        gpu = read_json(EVIDENCE / "inference_report.json")
        self.assertEqual(config["experiment_id"], run["experiment_id"])
        self.assertEqual(config["model"]["backbone"], co["backbone"])
        self.assertEqual(config["model"]["layers"], co["layers"])
        self.assertEqual(config["coreset"]["sampling_ratio"], co["sampling_ratio"])
        self.assertEqual(config["extraction"]["sample_count"], co["train_images"])
        self.assertEqual(run["measured"]["gpu_inference"], gpu)
        self.assertEqual(run["artifacts"]["checkpoint_sha256"], gpu["checkpoint_sha256"])
        for document in ("docs/epic-04/us-04-04.md", "docs/EPIC-04-patchcore.md",
                         "docs/epic-04/us-04-04-exp001.md"):
            path = ROOT / document
            for target in re.findall(r"\]\(([^)]+)\)", path.read_text(encoding="utf-8")):
                if not target.startswith(("https:", "http:", "#")):
                    self.assertTrue((path.parent / target.split("#")[0]).exists(), target)


if __name__ == "__main__":
    unittest.main()
