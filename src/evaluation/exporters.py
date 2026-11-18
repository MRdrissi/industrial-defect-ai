"""Reproducible exports for IndustrialDefect AI evaluation."""

import csv
import re
from contextlib import contextmanager
import hashlib
import json
import tempfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np


SCHEMA_VERSION = "1.0"
REQUIRED_PROVENANCE = (
    "run_id", "model_family", "category", "split", "git_commit",
    "checkpoint_sha256", "dataset_manifest_sha256", "device", "hardware",
    "python_version", "torch_version", "threshold", "latency_scope",
    "target_size_hw",
)


@contextmanager
def _export_lock(target: Path):
    """Serialize cooperating exporters; never remove another writer's lock."""
    lock = target.parent / f".{target.name}.lock"
    with lock.open("x", encoding="utf-8"):
        pass
    try:
        yield
    finally:
        lock.unlink()


def _validate_run_config(config: dict, shape: tuple[int, int]):
    for key in REQUIRED_PROVENANCE:
        if key not in config:
            raise ValueError(f"Missing run configuration: {key}")
        if key not in ("threshold", "target_size_hw"):
            if not isinstance(config[key], str) or not config[key].strip():
                raise ValueError(f"Invalid run configuration: {key}")
    if config["model_family"] not in ("cae", "patchcore"):
        raise ValueError("Unknown model_family.")
    if config["category"] not in ("bottle", "hazelnut", "transistor"):
        raise ValueError("Unknown category.")
    if config["split"] not in ("val", "test"):
        raise ValueError("Evaluation split must be val or test.")
    for key, length in (("git_commit", 40), ("checkpoint_sha256", 64),
                        ("dataset_manifest_sha256", 64)):
        if re.fullmatch(rf"[0-9a-fA-F]{{{length}}}", config[key]) is None:
            raise ValueError(f"Invalid fingerprint: {key}")
    if config["threshold"] is not None:
        _finite_number(config["threshold"], "threshold")
    size = config["target_size_hw"]
    if (not isinstance(size, (list, tuple)) or len(size) != 2
            or any(type(n) is not int or n <= 0 for n in size)
            or tuple(size) != shape):
        raise ValueError("target_size_hw must match the spatial evaluation resolution.")



@dataclass(frozen=True)
class EvaluationRecord:
    """Prediction and ground truth for one image."""

    image_path: str
    label: int
    score: float
    mask: np.ndarray
    anomaly_map: np.ndarray
    latency_ms: float


def file_sha256(path: Path) -> str:
    """Compute a file fingerprint without loading it entirely."""
    digest = hashlib.sha256()

    with Path(path).open("rb") as stream:
        for chunk in iter(
                lambda: stream.read(1024 * 1024), b""
        ):
            digest.update(chunk)

    return digest.hexdigest()


def _finite_number(value: Any, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or np.iscomplexobj(value):
        raise ValueError(f"{name} must be numeric.")

    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be numeric.") from exc

    if not np.isfinite(number):
        raise ValueError(f"{name} must be finite.")

    return number


def _json_safe(value: Any) -> Any:
    """Convert metadata to strict, JSON-compatible values."""
    if value is None or isinstance(value, (str, bool)):
        return value

    if isinstance(value, (int, np.integer)):
        return int(value)

    if isinstance(value, (float, np.floating)):
        return _finite_number(value, "JSON numeric value")

    if isinstance(value, dict):
        if not all(isinstance(k, str) for k in value):
            raise ValueError("JSON dictionary keys must be strings.")

        return {
            key: _json_safe(item)
            for key, item in value.items()
        }

    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]

    raise ValueError(
        f"Unsupported JSON metadata type: {type(value).__name__}"
    )


def _validate_record(record: EvaluationRecord):
    if not isinstance(record, EvaluationRecord):
        raise TypeError("Expected EvaluationRecord.")

    path = record.image_path

    if not isinstance(path, str) or not path.strip():
        raise ValueError("image_path must be non-empty.")

    parsed = PurePosixPath(path)

    if (
            parsed.is_absolute()
            or ".." in parsed.parts
            or "\\" in path
            or ":" in path
            or parsed.as_posix() != path
            or path == "."
            or any(ord(char) < 32 for char in path)
    ):
        raise ValueError("image_path must be a safe relative path.")

    if (
            isinstance(record.label, (bool, np.bool_))
            or not isinstance(record.label, (int, np.integer))
            or record.label not in (0, 1)
    ):
        raise ValueError("Image label must be 0 or 1.")

    score = _finite_number(record.score, "score")
    latency = _finite_number(
        record.latency_ms, "latency_ms"
    )

    if latency < 0:
        raise ValueError("latency_ms cannot be negative.")

    mask = np.asarray(record.mask)
    anomaly_map = np.asarray(record.anomaly_map)

    if mask.ndim != 2 or anomaly_map.ndim != 2:
        raise ValueError("Mask and map must both be 2D.")

    if (
            mask.size == 0
            or anomaly_map.size == 0
            or mask.shape != anomaly_map.shape
    ):
        raise ValueError(
            "Mask and anomaly map must have "
            "matching non-empty shapes."
        )

    if mask.dtype.kind not in "biuf" or not np.isin(mask, [0, 1]).all():
        raise ValueError("Ground-truth mask must be binary.")

    if record.label == 0 and np.any(mask):
        raise ValueError(
            "Normal image cannot have positive mask pixels."
        )

    if record.label == 1 and not np.any(mask):
        raise ValueError(
            "Anomalous image requires positive mask pixels."
        )

    if anomaly_map.dtype.kind not in "iuf":
        raise ValueError("Anomaly map must contain real numeric scores.")
    try:
        anomaly_map = anomaly_map.astype(np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(
            "Anomaly map must contain numeric scores."
        ) from exc

    if not np.isfinite(anomaly_map).all():
        raise ValueError("Anomaly map contains NaN or infinity.")

    return (
        score,
        latency,
        mask.astype(np.uint8),
        anomaly_map,
    )


def _write_json(path: Path, data: dict):
    path.write_text(
        json.dumps(
            _json_safe(data),
            indent=2,
            ensure_ascii=False,
            sort_keys=True,
            allow_nan=False,
        ) + "\n",
        encoding="utf-8",
        )


def export_evaluation(
        output_dir: str | Path,
        records: list[EvaluationRecord],
        metrics: dict,
        run_config: dict,
) -> Path:
    """Export one evaluation run atomically.

    Files:
      predictions.csv
      metrics.json
      manifest.json
      artifacts/00000.npz ...

    Refuses to overwrite an existing run.
    """
    target = Path(output_dir)

    if target.exists():
        raise FileExistsError(
            f"Evaluation output already exists: {target}"
        )

    records = list(records)

    if not records:
        raise ValueError("At least one record is required.")

    if not isinstance(metrics, dict):
        raise ValueError("metrics must be a dictionary.")

    if not isinstance(run_config, dict):
        raise ValueError("run_config must be a dictionary.")

    # Validate everything before writing anything
    validated = []
    seen_paths = set()

    for record in records:
        score, latency, mask, amap = _validate_record(record)
        if record.image_path in seen_paths:
            raise ValueError(
                f"Duplicate image_path: {record.image_path}"
            )

        seen_paths.add(record.image_path)

        validated.append((record, score, latency, mask, amap))

    shape = validated[0][3].shape
    if any(item[3].shape != shape for item in validated):
        raise ValueError("All records must use the same spatial resolution.")
    _validate_run_config(run_config, shape)

    safe_metrics = _json_safe(metrics)
    safe_config = _json_safe(run_config)

    latencies = np.array(
        [item[2] for item in validated],
        dtype=np.float64,
    )

    parent = target.parent
    parent.mkdir(parents=True, exist_ok=True)

    # Temporary staging avoids publishing incomplete runs
    with _export_lock(target), tempfile.TemporaryDirectory(
            dir=parent,
            prefix=".evaluation-staging-",
    ) as temporary_dir:

        staging = Path(temporary_dir)
        artifact_dir = staging / "artifacts"
        artifact_dir.mkdir()

        rows = []
        artifacts = []

        for index, (record, score, latency, mask, amap) in enumerate(
                validated
        ):
            relative_file = f"artifacts/{index:05d}.npz"
            artifact_path = staging / relative_file

            np.savez_compressed(
                artifact_path,
                mask=mask,
                anomaly_map=amap,
            )

            digest = file_sha256(artifact_path)

            rows.append({
                "index": index,
                "image_path": record.image_path,
                "label": int(record.label),
                "score": score,
                "latency_ms": latency,
                "artifact_file": relative_file,
                "artifact_sha256": digest,
            })

            artifacts.append({
                "image_path": record.image_path,
                "file": relative_file,
                "sha256": digest,
            })

        # Per-image predictions CSV
        csv_path = staging / "predictions.csv"

        with csv_path.open(
                "w", newline="", encoding="utf-8"
        ) as stream:
            writer = csv.DictWriter(
                stream,
                fieldnames=[
                    "index",
                    "image_path",
                    "label",
                    "score",
                    "latency_ms",
                    "artifact_file",
                    "artifact_sha256",
                ],
            )

            writer.writeheader()
            writer.writerows(rows)

        # Aggregate metrics and provenance
        metrics_data = {
            "schema_version": SCHEMA_VERSION,
            "run": safe_config,
            "n_images": len(records),
            "metrics": safe_metrics,
            "latency_summary": {
                "unit": "ms",
                "mean_ms": float(latencies.mean()),
                "median_ms": float(np.median(latencies)),
                "p95_ms": float(np.percentile(latencies, 95)),
                "min_ms": float(latencies.min()),
                "max_ms": float(latencies.max()),
            },
        }

        metrics_path = staging / "metrics.json"
        _write_json(metrics_path, metrics_data)

        manifest = {
            "schema_version": SCHEMA_VERSION,
            "run_id": safe_config["run_id"],
            "n_images": len(records),
            "artifacts": artifacts,
            "files": {
                "predictions.csv": file_sha256(csv_path),
                "metrics.json": file_sha256(metrics_path),
            },
        }

        _write_json(staging / "manifest.json", manifest)

        if target.exists():
            raise FileExistsError(
                f"Evaluation output already exists: {target}"
            )

        staging.rename(target)

    return target
