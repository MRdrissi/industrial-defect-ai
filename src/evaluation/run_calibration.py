"""Reproducible six-instance calibration entry point (CPU or Kaggle GPU)."""
from __future__ import annotations

import argparse
import csv
import tempfile
import hashlib
import importlib.metadata
import json
import platform
import subprocess
from pathlib import Path

from .calibration import (
    CalibrationSample, SCORE_DEFINITIONS, _json, _bytes, _verified_bytes,
    export_calibration, load_protocol, load_threshold, validation_paths,
)
from .exporters import file_sha256

ROOT = Path(__file__).resolve().parents[2]
CATALOG = ROOT / "experiments/configs/calibration_instances.json"


def canonical_json_sha256(path):
    """Semantic JSON fingerprint, independent of BOM and platform line endings."""
    document = _json(Path(path).read_bytes().removeprefix(b"\xef\xbb\xbf"))
    return hashlib.sha256(_bytes(document)).hexdigest()


def read_catalog(path=CATALOG):
    catalog = _json(Path(path).read_bytes())
    entries = catalog["instances"]
    expected = {f"{f}-{c}" for f in ("cae", "patchcore")
                for c in ("bottle", "hazelnut", "transistor")}
    if len(entries) != 6 or {e["id"] for e in entries} != expected:
        raise ValueError("Exactly six unique catalog instances are required.")
    load_protocol(ROOT/catalog["protocol"], catalog["protocol_sha256"])
    for item in entries:
        family, category = item["model_family"], item["category"]
        if item["id"] != f"{family}-{category}" or item["score_definition"] != SCORE_DEFINITIONS[family]:
            raise ValueError("Invalid instance identity/score definition.")
        validation_paths(ROOT/item["manifest"], category=category,
                         expected_sha256=item["manifest_sha256"],
                         expected_count=item["expected_validation_count"])
        if canonical_json_sha256(ROOT/item["model_config"]) != item["model_config_canonical_sha256"]:
            raise ValueError("Model configuration fingerprint mismatch.")
        if canonical_json_sha256(ROOT/item["source_registry"]) != item["source_registry_canonical_sha256"]:
            raise ValueError("Source experiment fingerprint mismatch.")
    return catalog


def _validation_dataset(item, dataset_root):
    from src.data.mvtec_dataset import MVTecDataset
    from src.data.transforms import MVTecTransform
    root = Path(dataset_root).resolve()
    expected = validation_paths(ROOT/item["manifest"], category=item["category"],
                                expected_sha256=item["manifest_sha256"],
                                expected_count=item["expected_validation_count"])
    # Preflight all and only Validation files before any inference.
    for relative in sorted(expected):
        path = (root/relative).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            raise FileNotFoundError(f"Missing/unsafe Validation image: {relative}")
    # The common dataset validates every row it receives. Give it only the
    # already-verified Validation projection so it never inspects Test labels.
    with tempfile.TemporaryDirectory(prefix="validation-manifest-") as temp:
        projection = Path(temp)/"validation.csv"
        with projection.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream, lineterminator="\n")
            writer.writerow(["category", "split", "image_path", "label", "defect_type", "mask_path"])
            writer.writerows((item["category"], "val", path, 0, "good", "")
                             for path in sorted(expected))
        dataset = MVTecDataset(root, projection, item["category"], "val",
                               MVTecTransform(size=(256, 256), profile=item["model_family"]))
    if {s["image_path"] for s in dataset.samples} != set(expected):
        raise ValueError("Dataset differs from logical Validation.")
    return dataset


def restore_instance(item, checkpoint, device="cpu"):
    """Restore only a fingerprinted existing model, never train or build a coreset."""
    import torch
    if file_sha256(checkpoint) != item["checkpoint_sha256"]:
        raise ValueError("Checkpoint SHA-256 mismatch.")
    if item["model_family"] == "cae":
        from src.inference.cae_inference import CAEInference, CAEInferenceConfig
        return CAEInference.from_checkpoint(
            item["category"], ROOT/item["model_config"], checkpoint, device,
            CAEInferenceConfig(smoothing_sigma=4.0, score_method="max"))
    if importlib.metadata.version("anomalib") != item["anomalib_version"]:
        raise ValueError("PatchCore requires the recorded Anomalib 2.7.0.")
    from src.inference.patchcore_inference import PatchCoreInference
    if item["checkpoint_format"] == "hazelnut_exp002":
        # Adapt the historical dictionary in memory; do not rewrite the checkpoint.
        from anomalib.models.image.patchcore.torch_model import PatchcoreModel
        saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
        config = saved["config"]
        required = {"category": "hazelnut", "experiment_id": "EXP-002",
                    "pooling_corrected": True, "backbone": "wide_resnet50_2",
                    "layers": ["layer2", "layer3"], "num_neighbors": 1,
                    "memory_bank_shape": [3205, 1536], "image_size": [256, 256],
                    "memory_bank_sha256": item["memory_bank_sha256"]}
        if any(config.get(k) != v for k, v in required.items()):
            raise ValueError("Hazelnut checkpoint is not the corrected EXP-002.")
        model = PatchcoreModel(backbone=config["backbone"], layers=config["layers"],
                               pre_trained=False, num_neighbors=config["num_neighbors"])
        model.load_state_dict(saved["state_dict"], strict=True)
        bank = model.memory_bank.detach().cpu().contiguous()
        if bank.dtype != torch.float32 or hashlib.sha256(bank.numpy().tobytes()).hexdigest() != item["memory_bank_sha256"]:
            raise ValueError("Corrected Memory Bank fingerprint mismatch.")
        model.feature_extractor.requires_grad_(False)
        engine = PatchCoreInference(model.to(device).eval(), item["category"])
    else:
        engine = PatchCoreInference.from_checkpoint(item["category"], checkpoint, device,
                                                    item["checkpoint_sha256"])
    bank = engine.model.memory_bank
    if list(bank.shape) != item["memory_bank_shape"] or not torch.isfinite(bank).all():
        raise ValueError("Unexpected Memory Bank.")
    return engine


def collect_validation_scores(dataset, engine):
    """Inference at batch size 1; never inspect or score the Test subset."""
    result = []
    for index in range(len(dataset)):
        sample = dataset[index]
        if sample["label"] != 0:
            raise ValueError("Non-normal calibration sample.")
        prediction = engine.predict(sample["image"])
        result.append(CalibrationSample(sample["image_path"], sample["label"], prediction.score))
    return result


def run_instance(instance_id, *, dataset_root, checkpoint, output_dir, device="cpu",
                 catalog_path=CATALOG, preflight=False):
    catalog = read_catalog(catalog_path)
    item = next((e for e in catalog["instances"] if e["id"] == instance_id), None)
    if item is None:
        raise ValueError("Unknown instance.")
    if Path(output_dir).exists():
        raise FileExistsError(output_dir)
    # Hash verification does not load an untrusted pickle.
    if file_sha256(checkpoint) != item["checkpoint_sha256"]:
        raise ValueError("Checkpoint SHA-256 mismatch.")
    dataset = _validation_dataset(item, dataset_root)
    if preflight:
        return {"instance": instance_id, "validation_images": len(dataset),
                "checkpoint_sha256": item["checkpoint_sha256"], "status": "inputs_verified"}
    import torch
    import numpy as np
    # Refuse uncommitted scientific code/configuration, including untracked files.
    status = subprocess.check_output(
        ["git", "status", "--porcelain", "--", "src", "experiments/configs"], cwd=ROOT, text=True)
    if status.strip():
        raise ValueError("Commit scientific code/configuration before real calibration.")
    git_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    torch.manual_seed(42)
    np.random.seed(42)
    if device.startswith("cuda"):
        if not torch.cuda.is_available():
            raise ValueError("CUDA is unavailable.")
        torch.cuda.manual_seed_all(42)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    engine = restore_instance(item, checkpoint, device)
    samples = collect_validation_scores(dataset, engine)
    provenance = {
        "run_id": Path(output_dir).name, "model_family": item["model_family"],
        "category": item["category"], "source_experiment": item["source_experiment"],
        "git_commit": git_commit, "checkpoint_sha256": item["checkpoint_sha256"],
        "model_config_sha256": file_sha256(ROOT/item["model_config"]),
        "model_config_canonical_sha256": item["model_config_canonical_sha256"],
        "catalog_sha256": file_sha256(catalog_path),
        "device": str(engine.device),
        "hardware": torch.cuda.get_device_name(engine.device) if engine.device.type == "cuda" else platform.processor() or platform.machine(),
        "python_version": platform.python_version(), "torch_version": str(torch.__version__),
        "numpy_version": np.__version__, "split": "val",
        "score_definition": item["score_definition"], "preprocessing": item["preprocessing"],
        "seed": 42, "batch_size": 1,
        "dependencies": {name: importlib.metadata.version(name) for name in
                         (["torchvision", "Pillow"] + (["anomalib", "timm", "lightning"] if item["model_family"] == "patchcore" else []))},
    }
    kwargs = {"manifest_path": ROOT/item["manifest"], "protocol_path": ROOT/catalog["protocol"],
              "expected_manifest_sha256": item["manifest_sha256"],
              "expected_protocol_sha256": catalog["protocol_sha256"],
              "expected_count": item["expected_validation_count"]}
    digest = export_calibration(output_dir, samples, provenance=provenance, **kwargs)
    frozen = load_threshold(output_dir, expected_integrity_sha256=digest,
                            model_family=item["model_family"], category=item["category"],
                            expected_checkpoint_sha256=item["checkpoint_sha256"], **kwargs)
    return {"instance": instance_id, "threshold": frozen.threshold,
            "threshold_hex": frozen.threshold.hex(), "integrity_sha256": digest,
            "validation_images": len(dataset), "status": "calibrated"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--instance", required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--catalog", type=Path, default=CATALOG)
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    print(json.dumps(run_instance(args.instance, dataset_root=args.dataset_root,
          checkpoint=args.checkpoint, output_dir=args.output, device=args.device,
          catalog_path=args.catalog, preflight=args.preflight), indent=2))


if __name__ == "__main__":
    main()
