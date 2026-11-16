"""Model-independent evaluation metrics for IndustrialDefect AI."""

import numpy as np
from sklearn.metrics import (
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
)


def _validate_labels(labels):
    """Validate binary image labels: 0=normal, 1=anomalous."""
    arr = np.asarray(labels)

    if arr.ndim != 1 or arr.size == 0:
        raise ValueError("Labels must be a non-empty 1D array.")

    if arr.dtype.kind not in "biuf" or not np.isin(arr, [0, 1]).all():
        raise ValueError("Labels must contain only 0 and 1.")

    return arr.astype(np.int64)


def _validate_scores(scores):
    """Validate finite, continuous anomaly scores."""
    arr = np.asarray(scores)
    if arr.dtype.kind not in "iuf":
        raise ValueError("Scores must be real numeric values.")
    arr = arr.astype(np.float64)

    if arr.ndim != 1 or arr.size == 0:
        raise ValueError("Scores must be a non-empty 1D array.")

    if not np.isfinite(arr).all():
        raise ValueError("Scores contain NaN or infinity.")

    return arr


def compute_image_metrics(labels, scores, threshold=None):
    """Compute image-level detection metrics.

    AUROC is threshold-independent.
    Precision, Recall and F1 require a fixed threshold.

    Decision rule: anomalous if score > threshold.

    Undefined metrics are returned as None.
    """
    y_true = _validate_labels(labels)
    y_score = _validate_scores(scores)

    if y_true.shape != y_score.shape:
        raise ValueError("Labels and scores must have equal length.")

    result = {
        "n_images": int(y_true.size),
        "n_normal": int(np.sum(y_true == 0)),
        "n_anomalous": int(np.sum(y_true == 1)),
        "image_auroc": None,
        "threshold": None,
        "precision": None,
        "recall": None,
        "f1": None,
        "tp": None,
        "tn": None,
        "fp": None,
        "fn": None,
    }

    # AUROC requires both classes
    if np.unique(y_true).size == 2:
        result["image_auroc"] = float(
            roc_auc_score(y_true, y_score)
        )

    # Classification metrics only with a fixed threshold
    if threshold is not None:
        if isinstance(threshold, (bool, np.bool_)) or np.iscomplexobj(threshold):
            raise ValueError("Threshold must be numeric.")

        try:
            threshold = float(threshold)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("Threshold must be numeric.") from exc

        if not np.isfinite(threshold):
            raise ValueError("Threshold must be finite.")

        y_pred = (y_score > threshold).astype(np.int64)

        result.update({
            "threshold": threshold,
            "precision": float(
                precision_score(
                    y_true, y_pred, zero_division=0
                )
            ),
            "recall": float(
                recall_score(
                    y_true, y_pred, zero_division=0
                )
            ),
            "f1": float(
                f1_score(
                    y_true, y_pred, zero_division=0
                )
            ),
            "tp": int(np.sum((y_true == 1) & (y_pred == 1))),
            "tn": int(np.sum((y_true == 0) & (y_pred == 0))),
            "fp": int(np.sum((y_true == 0) & (y_pred == 1))),
            "fn": int(np.sum((y_true == 1) & (y_pred == 0))),
        })

    return result


def compute_pixel_auroc(masks, anomaly_maps):
    """Compute pooled pixel-level AUROC.

    Expected inputs:
        (N, H, W) or (N, 1, H, W)

    Masks must be binary.
    Anomaly maps must contain finite continuous scores.

    All pixels are pooled across the dataset.
    Normal images contribute all-zero masks.
    """
    y_true = np.asarray(masks)
    y_score = np.asarray(anomaly_maps)
    if y_score.dtype.kind not in "iuf":
        raise ValueError("Anomaly maps must be real numeric values.")
    y_score = y_score.astype(np.float64)

    # Support channel-first single-channel tensors
    if y_true.ndim == 4 and y_true.shape[1] == 1:
        y_true = y_true[:, 0]

    if y_score.ndim == 4 and y_score.shape[1] == 1:
        y_score = y_score[:, 0]

    if y_true.ndim != 3 or y_score.ndim != 3:
        raise ValueError(
            "Masks and maps must have shape (N,H,W) "
            "or (N,1,H,W)."
        )

    if y_true.shape != y_score.shape:
        raise ValueError(
            "Masks and anomaly maps must have identical shapes."
        )

    if y_true.size == 0:
        raise ValueError("Empty pixel evaluation dataset.")

    if y_true.dtype.kind not in "biuf" or not np.isin(y_true, [0, 1]).all():
        raise ValueError("Ground-truth masks must be binary.")

    if not np.isfinite(y_score).all():
        raise ValueError("Anomaly maps contain NaN or infinity.")

    labels_flat = y_true.reshape(-1).astype(np.uint8)
    scores_flat = y_score.reshape(-1)

    # Undefined if no positive or no negative pixels
    if np.unique(labels_flat).size < 2:
        return None

    return float(
        roc_auc_score(labels_flat, scores_flat)
    )
