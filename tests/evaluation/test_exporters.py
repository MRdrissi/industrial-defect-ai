
import csv
import json
import hashlib

import numpy as np
import pytest

from src.evaluation.exporters import (
    EvaluationRecord,
    export_evaluation,
    file_sha256,
)


@pytest.fixture
def config():
    return {
        "run_id": "synthetic-cae-bottle",
        "model_family": "cae",
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
        "threshold": None,
    }


@pytest.fixture
def records():
    normal_mask = np.zeros((2, 2), dtype=np.uint8)

    anomalous_mask = np.array(
        [[0, 1], [0, 0]], dtype=np.uint8
    )

    return [
        EvaluationRecord(
            image_path="bottle/test/good/001.png",
            label=0,
            score=0.1,
            mask=normal_mask,
            anomaly_map=np.full((2, 2), 0.1),
            latency_ms=2.0,
        ),
        EvaluationRecord(
            image_path="bottle/test/broken_large/001.png",
            label=1,
            score=0.9,
            mask=anomalous_mask,
            anomaly_map=np.array([
                [0.1, 0.9],
                [0.2, 0.1]
            ]),
            latency_ms=4.0,
        ),
    ]


def test_complete_export(tmp_path, records, config):
    output = tmp_path / "run"

    result = export_evaluation(
        output,
        records,
        metrics={"image_auroc": 1.0, "pixel_auroc": 1.0},
        run_config=config,
    )

    assert result == output
    assert (output / "predictions.csv").is_file()
    assert (output / "metrics.json").is_file()
    assert (output / "manifest.json").is_file()
    assert len(list((output / "artifacts").glob("*.npz"))) == 2


def test_csv_predictions(tmp_path, records, config):
    output = export_evaluation(
        tmp_path / "run", records, {}, config
    )

    with (output / "predictions.csv").open(
            newline="", encoding="utf-8"
    ) as stream:
        rows = list(csv.DictReader(stream))

    assert len(rows) == 2
    assert rows[0]["label"] == "0"
    assert rows[1]["label"] == "1"
    assert float(rows[1]["score"]) == pytest.approx(0.9)
    assert float(rows[0]["latency_ms"]) == pytest.approx(2.0)


def test_npz_masks_and_maps(tmp_path, records, config):
    output = export_evaluation(
        tmp_path / "run", records, {}, config
    )

    with np.load(
            output / "artifacts/00001.npz",
            allow_pickle=False
    ) as data:
        assert data["mask"].shape == (2, 2)
        assert data["anomaly_map"].shape == (2, 2)
        assert data["mask"].dtype == np.uint8
        assert data["anomaly_map"].dtype == np.float64
        assert data["mask"][0, 1] == 1


def test_metrics_and_latency(tmp_path, records, config):
    output = export_evaluation(
        tmp_path / "run",
        records,
        {"image_auroc": 1.0, "f1": None},
        config,
        )

    data = json.loads(
        (output / "metrics.json").read_text(encoding="utf-8")
    )

    assert data["metrics"]["image_auroc"] == 1.0
    assert data["metrics"]["f1"] is None
    assert data["n_images"] == 2
    assert data["latency_summary"]["mean_ms"] == 3.0
    assert data["run"]["model_family"] == "cae"


def test_artifact_hashes(tmp_path, records, config):
    output = export_evaluation(
        tmp_path / "run", records, {}, config
    )

    manifest = json.loads(
        (output / "manifest.json").read_text(encoding="utf-8")
    )

    for item in manifest["artifacts"]:
        assert file_sha256(output / item["file"]) == item["sha256"]

    for filename, digest in manifest["files"].items():
        assert file_sha256(output / filename) == digest


def test_existing_run_cannot_be_overwritten(
        tmp_path, records, config
):
    output = tmp_path / "run"

    export_evaluation(output, records, {}, config)

    original_hash = file_sha256(output / "metrics.json")

    with pytest.raises(FileExistsError):
        export_evaluation(output, records, {}, config)

    assert file_sha256(output / "metrics.json") == original_hash


def test_duplicate_paths_rejected(tmp_path, records, config):
    duplicated = [records[0], records[0]]

    with pytest.raises(ValueError, match="Duplicate"):
        export_evaluation(
            tmp_path / "run", duplicated, {}, config
        )


def test_normal_positive_mask_rejected(
        tmp_path, records, config
):
    bad = EvaluationRecord(
        image_path="bottle/test/good/002.png",
        label=0,
        score=0.1,
        mask=np.ones((2, 2)),
        anomaly_map=np.zeros((2, 2)),
        latency_ms=1.0,
    )

    with pytest.raises(ValueError):
        export_evaluation(
            tmp_path / "run", [bad], {}, config
        )

    assert not (tmp_path / "run").exists()


def test_mismatched_shapes_rejected(
        tmp_path, records, config
):
    bad = EvaluationRecord(
        image_path="bottle/test/good/002.png",
        label=0,
        score=0.1,
        mask=np.zeros((2, 2)),
        anomaly_map=np.zeros((4, 4)),
        latency_ms=1.0,
    )

    with pytest.raises(ValueError):
        export_evaluation(
            tmp_path / "run", [bad], {}, config
        )


def test_negative_latency_rejected(
        tmp_path, records, config
):
    bad = EvaluationRecord(
        image_path="bottle/test/good/002.png",
        label=0,
        score=0.1,
        mask=np.zeros((2, 2)),
        anomaly_map=np.zeros((2, 2)),
        latency_ms=-1,
    )

    with pytest.raises(ValueError):
        export_evaluation(
            tmp_path / "run", [bad], {}, config
        )


def test_nonfinite_metrics_rejected(
        tmp_path, records, config
):
    with pytest.raises(ValueError):
        export_evaluation(
            tmp_path / "run",
            records,
            {"image_auroc": float("nan")},
            config,
            )


def test_missing_provenance_rejected(
        tmp_path, records, config
):
    incomplete = {"model_family": "cae"}

    with pytest.raises(ValueError):
        export_evaluation(
            tmp_path / "run", records, {}, incomplete
        )


def test_hash_matches_hashlib(tmp_path):
    path = tmp_path / "example.txt"
    path.write_bytes(b"industrial-defect-ai")

    expected = hashlib.sha256(
        b"industrial-defect-ai"
    ).hexdigest()

    assert file_sha256(path) == expected


def test_float64_npz_round_trip_and_pixel_metric(tmp_path, config):
    from src.evaluation.benchmark import evaluate_predictions
    maps = np.array([[1., 1.+1e-9], [1., 1.+1e-9]], dtype=np.float64)
    item = EvaluationRecord("bottle/test/crack/a.png", 1, .5,
                            np.array([[0, 1], [0, 1]]), maps, .1)
    report = evaluate_predictions([item], output_dir=tmp_path/"run", run_config=config)
    assert report["pixel"]["pixel_auroc"] == 1.
    with np.load(tmp_path/"run/artifacts/00000.npz", allow_pickle=False) as data:
        np.testing.assert_array_equal(data["anomaly_map"], maps)


@pytest.mark.parametrize("key", [
    "git_commit", "checkpoint_sha256", "dataset_manifest_sha256",
    "hardware", "python_version", "torch_version", "threshold", "target_size_hw",
])
def test_required_provenance_enforced(tmp_path, records, config, key):
    del config[key]
    with pytest.raises(ValueError, match=key):
        export_evaluation(tmp_path/"run", records, {}, config)
    assert not (tmp_path/"run").exists()


@pytest.mark.parametrize("key,value", [
    ("git_commit", "unknown"), ("checkpoint_sha256", "z"*64),
    ("target_size_hw", [256, 256]), ("split", "train"),
])
def test_inconsistent_provenance_rejected(tmp_path, records, config, key, value):
    config[key] = value
    with pytest.raises(ValueError):
        export_evaluation(tmp_path/"run", records, {}, config)


@pytest.mark.parametrize("path", [
    "../image.png", "/image.png", "C:/image.png", "a//b.png", "a/./b.png", ".",
    "a\\b.png", "a\nb.png",
])
def test_noncanonical_paths_rejected(tmp_path, records, config, path):
    from dataclasses import replace
    with pytest.raises(ValueError):
        export_evaluation(tmp_path/"run", [replace(records[0], image_path=path)], {}, config)


def test_failed_export_leaves_no_run_or_lock(tmp_path, records, config, monkeypatch):
    def fail(*args, **kwargs):
        raise OSError("simulated storage failure")
    monkeypatch.setattr(np, "savez_compressed", fail)
    with pytest.raises(OSError, match="storage"):
        export_evaluation(tmp_path/"run", records, {}, config)
    assert list(tmp_path.iterdir()) == []


def test_concurrent_export_refused(tmp_path, records, config, monkeypatch):
    original = np.savez_compressed
    attempts = []
    def save(*args, **kwargs):
        with pytest.raises(FileExistsError):
            export_evaluation(tmp_path/"run", records, {}, config)
        attempts.append(True)
        original(*args, **kwargs)
    monkeypatch.setattr(np, "savez_compressed", save)
    export_evaluation(tmp_path/"run", records, {}, config)
    assert len(attempts) == len(records)
    assert not (tmp_path/".run.lock").exists()


def test_target_created_during_export_preserved(tmp_path, records, config, monkeypatch):
    original = np.savez_compressed
    def save(*args, **kwargs):
        (tmp_path/"run").mkdir(exist_ok=True)
        (tmp_path/"run/foreign.txt").write_text("preserve", encoding="utf-8")
        original(*args, **kwargs)
    monkeypatch.setattr(np, "savez_compressed", save)
    with pytest.raises(FileExistsError):
        export_evaluation(tmp_path/"run", records, {}, config)
    assert (tmp_path/"run/foreign.txt").read_text(encoding="utf-8") == "preserve"
    assert sorted(p.name for p in tmp_path.iterdir()) == ["run"]


def test_invalid_record_type_rejected(tmp_path, config):
    with pytest.raises(TypeError, match="EvaluationRecord"):
        export_evaluation(tmp_path/"run", [object()], {}, config)
