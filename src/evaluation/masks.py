"""Mask and anomaly-map preparation for scientific evaluation."""

import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F


def _as_batch(values, name):
    """Normalize HW, NHW or N1HW arrays to NHW."""
    arr = np.asarray(values)

    if arr.ndim == 2:
        arr = arr[np.newaxis, :, :]
    elif arr.ndim == 4:
        if arr.shape[1] != 1:
            raise ValueError(
                f"{name} must have exactly one channel."
            )
        arr = arr[:, 0]
    elif arr.ndim != 3:
        raise ValueError(
            f"{name} must have shape (H,W), "
            "(N,H,W) or (N,1,H,W)."
        )

    if any(dim == 0 for dim in arr.shape):
        raise ValueError(f"{name} cannot be empty.")

    return arr


def _validate_size(target_size):
    """Validate target spatial dimensions (height, width)."""
    if not isinstance(target_size, (tuple, list)):
        raise ValueError("target_size must be (height, width).")

    if len(target_size) != 2:
        raise ValueError("target_size must contain two values.")

    height, width = target_size

    if any(
            isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, np.integer))
            or value <= 0
            for value in (height, width)
    ):
        raise ValueError(
            "Target height and width must be positive integers."
        )

    return int(height), int(width)


def resize_binary_masks(masks, target_size):
    """Resize binary masks using nearest-neighbor interpolation.

    Inputs: HW, NHW or N1HW
    Output: (N,H,W), uint8 with values 0 or 1.
    """
    arr = _as_batch(masks, "Masks")
    height, width = _validate_size(target_size)

    if arr.dtype.kind not in "biuf" or not np.isin(arr, [0, 1]).all():
        raise ValueError("Masks must be binary (0 or 1).")

    arr = arr.astype(np.uint8, copy=False)

    if arr.shape[-2:] == (height, width):
        return arr.copy()

    output = [
        np.asarray(
            Image.fromarray(mask).resize(
                (width, height),
                resample=Image.Resampling.NEAREST
            ),
            dtype=np.uint8
        )
        for mask in arr
    ]

    result = np.stack(output)

    lost = np.any(arr, axis=(1, 2)) & ~np.any(result, axis=(1, 2))
    if lost.any():
        raise ValueError(
            f"Positive mask disappeared during resizing: indices {np.flatnonzero(lost).tolist()}"
        )
    assert np.isin(result, [0, 1]).all()
    return result


def resize_anomaly_maps(anomaly_maps, target_size):
    """Resize continuous anomaly maps using bilinear interpolation.

    No thresholding or score normalization is applied.
    Output: (N,H,W), float64; half-pixel centers, antialias=True.
    """
    arr = _as_batch(anomaly_maps, "Anomaly maps")
    height, width = _validate_size(target_size)

    if arr.dtype.kind not in "iuf":
        raise ValueError("Anomaly maps must contain real numeric scores.")
    # Never collapse distinct float64 scores to float32 ties.
    arr = arr.astype(np.float64)
    if not np.isfinite(arr).all():
        raise ValueError("Anomaly maps contain NaN or infinity.")
    if arr.shape[-2:] == (height, width):
        return arr.copy()

    result = F.interpolate(
        torch.from_numpy(arr[:, None]),
        size=(height, width),
        mode="bilinear",
        align_corners=False,
        antialias=True,
    )[:, 0].numpy()

    if not np.isfinite(result).all():
        raise ValueError("Resized anomaly maps are not finite.")

    return result


def prepare_ground_truth_mask(mask, label, target_size):
    """Prepare one ground-truth mask.

    A missing mask is permitted only for a normal image.
    An anomalous image requires a non-empty positive mask.
    """
    height, width = _validate_size(target_size)

    if (isinstance(label, (bool, np.bool_))
            or not isinstance(label, (int, np.integer))
            or label not in (0, 1)):
        raise ValueError("Image label must be 0 or 1.")

    if mask is None:
        if label == 1:
            raise ValueError(
                "An anomalous image requires a ground-truth mask."
            )
        return np.zeros((height, width), dtype=np.uint8)

    arr = _as_batch(mask, "Ground-truth mask")

    if arr.shape[0] != 1:
        raise ValueError("Expected exactly one mask.")

    if arr.dtype.kind not in "biuf" or not np.isin(arr, [0, 1]).all():
        raise ValueError("Ground-truth mask must be binary.")

    has_positive = bool(np.any(arr))

    if label == 0 and has_positive:
        raise ValueError(
            "Normal image has a positive ground-truth mask."
        )

    if label == 1 and not has_positive:
        raise ValueError(
            "Anomalous image has an empty ground-truth mask."
        )

    return resize_binary_masks(
        arr, (height, width)
    )[0]


def align_pixel_inputs(masks, anomaly_maps, target_size=None):
    """Align ground-truth masks and anomaly maps.

    By default, preserves the ground-truth mask resolution.
    Explicit target_size overrides that choice.
    """
    mask_batch = _as_batch(masks, "Masks")
    map_batch = _as_batch(anomaly_maps, "Anomaly maps")

    if mask_batch.shape[0] != map_batch.shape[0]:
        raise ValueError(
            "Masks and anomaly maps must have "
            "the same number of images."
        )

    if target_size is None:
        target_size = mask_batch.shape[-2:]

    aligned_masks = resize_binary_masks(
        mask_batch, target_size
    )
    aligned_maps = resize_anomaly_maps(
        map_batch, target_size
    )

    assert aligned_masks.shape == aligned_maps.shape

    return aligned_masks, aligned_maps
