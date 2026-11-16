"""Model-independent scientific evaluation pipeline."""

from collections.abc import Iterable
from pathlib import Path

import numpy as np

from .exporters import (
    EvaluationRecord,
    _validate_record,
    export_evaluation,
)
from .metrics import (
    compute_image_metrics,
    compute_pixel_auroc,
)


def _check_threshold_consistency(config: dict, threshold):
    """Reject contradictions between evaluation and run metadata."""
    if "threshold" not in config:
        return

    stored = config["threshold"]

    if stored is None:
        if threshold is not None:
            raise ValueError("Conflicting threshold in run_config.")
        return

    if isinstance(stored, (bool, np.bool_)) or np.iscomplexobj(stored):
        raise ValueError("Invalid threshold in run_config.")

    try:
        stored = float(stored)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(
            "Invalid threshold in run_config."
        ) from exc

    if (
            not np.isfinite(stored)
            or threshold is None
            or stored != threshold
    ):
        raise ValueError("Conflicting threshold in run_config.")


def evaluate_predictions(
        records: Iterable[EvaluationRecord],
        *,
        threshold: float | None = None,
        output_dir: str | Path | None = None,
        run_config: dict | None = None,
) -> dict:
    """Evaluate one model/category on a fixed dataset.

    No threshold is learned here.
    A supplied threshold must come from a separate
    calibration procedure.

    All masks and maps must be spatially aligned
    to the same evaluation resolution.

    Latencies must have been measured by the caller.
    """
    records = list(records)

    if not records:
        raise ValueError("Evaluation dataset cannot be empty.")

    if (output_dir is None) != (run_config is None):
        raise ValueError(
            "output_dir and run_config must be supplied together."
        )

    # Validate records before computing metrics or exporting.
    labels = []
    scores = []
    masks = []
    maps = []
    latencies = []
    seen_paths = set()
    expected_shape = None

    for record in records:
        if not isinstance(record, EvaluationRecord):
            raise TypeError("Expected EvaluationRecord.")

        # Reuse the exporter's validation contract.
        score, latency, mask, anomaly_map = _validate_record(
            record
        )

        if record.image_path in seen_paths:
            raise ValueError(
                f"Duplicate image_path: {record.image_path}"
            )

        seen_paths.add(record.image_path)

        if expected_shape is None:
            expected_shape = mask.shape
        elif mask.shape != expected_shape:
            raise ValueError(
                "All masks and maps must use the same "
                "spatial evaluation resolution."
            )

        labels.append(int(record.label))
        scores.append(score)
        masks.append(mask)
        maps.append(anomaly_map)
        latencies.append(latency)

    # Image-level metrics, including confusion matrix.
    image_metrics = compute_image_metrics(
        labels,
        scores,
        threshold=threshold,
    )

    # Pooled pixel-level AUROC.
    # Include normal images with all-zero ground-truth masks.
    pixel_auroc = compute_pixel_auroc(
        np.stack(masks),
        np.stack(maps),
    )

    latency_values = np.asarray(
        latencies, dtype=np.float64
    )

    report = {
        "n_images": len(records),
        "image": image_metrics,
        "pixel": {
            "pixel_auroc": pixel_auroc,
        },
        "latency": {
            "unit": "ms",
            "mean_ms": float(latency_values.mean()),
            "median_ms": float(
                np.median(latency_values)
            ),
            "p95_ms": float(
                np.percentile(latency_values, 95)
            ),
            "min_ms": float(latency_values.min()),
            "max_ms": float(latency_values.max()),
        },
    }

    if output_dir is not None:
        if not isinstance(run_config, dict):
            raise ValueError("run_config must be a dictionary.")

        # Persist the exact threshold used for predictions.
        config = dict(run_config)

        _check_threshold_consistency(
            config,
            image_metrics["threshold"],
        )

        config["threshold"] = image_metrics["threshold"]

        export_evaluation(
            output_dir=output_dir,
            records=records,
            metrics=report,
            run_config=config,
        )

    return report
