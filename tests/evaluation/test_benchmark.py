
import csv
import json
from dataclasses import replace

import numpy as np
import pytest

from src.evaluation.benchmark import evaluate_predictions
from src.evaluation.exporters import (
    EvaluationRecord,
    file_sha256,
)


@pytest.fixture
def records():
    """Four deterministic synthetic predictions."""
    result = []

    samples = [
        ("good/001.png", 0, 0.10, 1.0),
        ("good/002.png", 0, 0.40, 2.0),
        ("crack/001.png", 1, 0.35, 3.0),
        ("crack/002.png", 1, 0.80, 4.0),
    ]

    for image_path, label, score, latency in samples:
        mask = np.zeros((2, 2), dtype=np.uint8)
        anomaly_map = np.full(
            (2, 2), 0.1, dtype=np.float32
        )

        if label == 1:
            mask[0, 1] = 1
            anomaly_map[0, 1] = 0.9

        result.append(
            EvaluationRecord(
                image_path=f"bottle/test/{image_path}",
                label=label,
                score=score,
                mask=mask,
                anomaly_map=anomaly_map,
                latency_ms=latency,
            )
        )

    return result


@pytest.fixture
def run_config():
    return {
        "run_id": "synthetic-patchcore-bottle",
        "model_family": "patchcore",
        "category": "bottle",
        "split": "test",
        "latency_scope": "forward_only",
        "device": "cpu",
        "git_commit": "a" * 40,
        "checkpoint_sha256": "b" * 64,
        "dataset_manifest_sha256": "c" * 64,
        "hardware": "synthetic-test-cpu",
        "python_version": "synthetic",
        "torch_version": "synthetic",
        "target_size_hw": [2, 2],
    }


def test_reference_image_and_pixel_metrics(records):
    report = evaluate_predictions(
        records, threshold=0.30
    )

    image = report["image"]

    assert report["n_images"] == 4
    assert image["image_auroc"] == pytest.approx(0.75)
    assert image["precision"] == pytest.approx(2 / 3)
    assert image["recall"] == pytest.approx(1.0)
    assert image["f1"] == pytest.approx(0.8)

    assert (image["tp"], image["tn"]) == (2, 1)
    assert (image["fp"], image["fn"]) == (1, 0)
    assert report["pixel"]["pixel_auroc"] == pytest.approx(1.0)


def test_latency_summary(records):
    report = evaluate_predictions(records)

    latency = report["latency"]

    assert latency["unit"] == "ms"
    assert latency["mean_ms"] == pytest.approx(2.5)
    assert latency["median_ms"] == pytest.approx(2.5)
    assert latency["min_ms"] == 1.0
    assert latency["max_ms"] == 4.0
    assert latency["p95_ms"] == pytest.approx(3.85)


def test_export_consistency(tmp_path, records, run_config):
    output = tmp_path / "benchmark-run"

    report = evaluate_predictions(
        records,
        threshold=0.30,
        output_dir=output,
        run_config=run_config,
    )

    assert (output / "metrics.json").exists()
    assert (output / "predictions.csv").exists()
    assert (output / "manifest.json").exists()

    document = json.loads(
        (output / "metrics.json").read_text(
            encoding="utf-8"
        )
    )

    assert document["metrics"] == report
    assert document["run"]["threshold"] == pytest.approx(0.30)
    assert document["n_images"] == 4

    with (output / "predictions.csv").open(
            newline="", encoding="utf-8"
    ) as stream:
        rows = list(csv.DictReader(stream))

    assert len(rows) == 4

    manifest = json.loads(
        (output / "manifest.json").read_text(
            encoding="utf-8"
        )
    )

    for artifact in manifest["artifacts"]:
        assert file_sha256(
            output / artifact["file"]
        ) == artifact["sha256"]


def test_single_class_auc_undefined(records):
    normal_only = records[:2]

    report = evaluate_predictions(normal_only)

    assert report["image"]["image_auroc"] is None
    assert report["pixel"]["pixel_auroc"] is None


def test_without_threshold_no_binary_metrics(records):
    report = evaluate_predictions(records)

    assert report["image"]["image_auroc"] == pytest.approx(0.75)
    assert report["image"]["precision"] is None
    assert report["image"]["recall"] is None
    assert report["image"]["f1"] is None


def test_metrics_independent_of_record_order(records):
    first = evaluate_predictions(records, threshold=0.30)
    reversed_order = evaluate_predictions(
        list(reversed(records)),
        threshold=0.30,
    )

    assert first == reversed_order


def test_duplicate_paths_rejected(records):
    with pytest.raises(ValueError, match="Duplicate"):
        evaluate_predictions(
            [records[0], records[0]]
        )


def test_mixed_spatial_resolutions_rejected(records):
    changed = replace(
        records[1],
        mask=np.zeros((4, 4), dtype=np.uint8),
        anomaly_map=np.zeros((4, 4), dtype=np.float32),
    )

    with pytest.raises(ValueError, match="spatial"):
        evaluate_predictions(
            [records[0], changed]
        )


def test_inconsistent_label_and_mask_rejected(records):
    bad = replace(
        records[0],
        mask=np.ones((2, 2), dtype=np.uint8),
    )

    with pytest.raises(ValueError):
        evaluate_predictions([bad])


def test_empty_dataset_rejected():
    with pytest.raises(ValueError):
        evaluate_predictions([])


def test_export_requires_configuration(tmp_path, records):
    output = tmp_path / "invalid"

    with pytest.raises(ValueError):
        evaluate_predictions(
            records,
            output_dir=output,
        )

    assert not output.exists()


def test_conflicting_threshold_rejected(
        tmp_path, records, run_config
):
    config = {
        **run_config,
        "threshold": 0.90,
    }

    output = tmp_path / "invalid"

    with pytest.raises(ValueError, match="Conflicting"):
        evaluate_predictions(
            records,
            threshold=0.30,
            output_dir=output,
            run_config=config,
        )

    assert not output.exists()


def test_existing_run_not_overwritten(
        tmp_path, records, run_config
):
    output = tmp_path / "run"

    evaluate_predictions(
        records,
        threshold=0.30,
        output_dir=output,
        run_config=run_config,
    )

    original_hash = file_sha256(
        output / "metrics.json"
    )

    with pytest.raises(FileExistsError):
        evaluate_predictions(
            records,
            threshold=0.30,
            output_dir=output,
            run_config=run_config,
        )

    assert file_sha256(
        output / "metrics.json"
    ) == original_hash


def test_nonfinite_image_score_rejected(records):
    bad = replace(records[0], score=float("nan"))

    with pytest.raises(ValueError):
        evaluate_predictions([bad])


def test_negative_latency_rejected(records):
    bad = replace(records[0], latency_ms=-1.0)

    with pytest.raises(ValueError):
        evaluate_predictions([bad])

def test_evaluation_protocol_valid_json():
    import json
    from pathlib import Path

    config_path = (
            Path(__file__).resolve().parents[2]
            / "experiments"
            / "configs"
            / "evaluation_protocol.json"
    )

    raw = config_path.read_bytes()

    assert not raw.startswith(b"\xef\xbb\xbf"), (
        "Evaluation protocol must be UTF-8 without BOM."
    )

    config = json.loads(raw.decode("utf-8"))

    assert config["protocol_version"] == "1.0.0"
    assert config["instances"]["expected_count"] == 6
    assert config["dataset"]["test_used_for_calibration"] is False


def test_protocol_essential_parameters_and_no_test_calibration():
    from pathlib import Path
    from src.evaluation.exporters import REQUIRED_PROVENANCE
    path = Path(__file__).resolve().parents[2]/"experiments/configs/evaluation_protocol.json"
    config = json.loads(path.read_text(encoding="utf-8"))
    dataset, calibration = config["dataset"], config["calibration"]
    assert dataset["training_split"] == "train"
    assert dataset["calibration_split"] == "val"
    assert dataset["evaluation_split"] == "test"
    assert dataset["validation_contains_only_normal_images"] is True
    assert dataset["test_used_for_calibration"] is False
    assert calibration["test_labels_used_for_calibration"] is False
    assert calibration["thresholds_frozen_before_test"] is True
    assert calibration["method"] == "normal_validation_quantile"
    assert calibration["quantile"] == 0.95
    assert calibration["quantile_method"] == "linear"
    assert calibration["number_of_thresholds"] == config["instances"]["expected_count"] == 6
    assert set(config["instances"]["model_families"]) == {"cae", "patchcore"}
    assert set(config["instances"]["categories"]) == {"bottle", "hazelnut", "transistor"}
    prediction = config["prediction_contract"]
    assert (prediction["normal_label"], prediction["anomaly_label"]) == (0, 1)
    assert prediction["decision_rule"] == "score > threshold"
    assert prediction["score_normalization_on_test"] is False
    spatial = config["spatial_evaluation"]
    assert spatial["target_size_hw"] == [256, 256]
    assert spatial["mask_interpolation"] == "nearest"
    assert spatial["anomaly_map_interpolation"] == "bilinear"
    assert spatial["align_corners"] is False and spatial["antialias"] is True
    assert spatial["positive_mask_disappearance"] == "error"
    assert spatial["evaluation_map_dtype"] == "float64"
    assert config["metrics"]["undefined_auroc"] is None
    assert config["metrics"]["zero_division"] == 0
    assert config["latency"]["scope"] == "model_inference_only"
    assert config["latency"]["batch_size"] == 1
    assert config["latency"]["warmup_runs_per_model"] == 3
    assert config["latency"]["synchronize_cuda"] is True
    assert config["exports"]["overwrite_existing_run"] is False
    assert set(config["required_provenance"]) == set(REQUIRED_PROVENANCE)
