
import numpy as np
import pytest

from src.evaluation.metrics import (
    compute_image_metrics,
    compute_pixel_auroc,
)


# ============================================
# IMAGE METRICS
# ============================================

def test_image_metrics_reference_example():
    labels = [0, 0, 1, 1]
    scores = [0.10, 0.40, 0.35, 0.80]

    result = compute_image_metrics(
        labels,
        scores,
        threshold=0.30
    )

    assert result["image_auroc"] == pytest.approx(0.75)
    assert result["precision"] == pytest.approx(2 / 3)
    assert result["recall"] == pytest.approx(1.0)
    assert result["f1"] == pytest.approx(0.8)

    assert result["tp"] == 2
    assert result["tn"] == 1
    assert result["fp"] == 1
    assert result["fn"] == 0


def test_perfect_image_auroc():
    result = compute_image_metrics(
        labels=[0, 0, 1, 1],
        scores=[0.1, 0.2, 0.8, 0.9]
    )

    assert result["image_auroc"] == pytest.approx(1.0)
    assert result["f1"] is None


def test_single_class_auroc_undefined():
    result = compute_image_metrics(
        labels=[0, 0, 0],
        scores=[0.1, 0.2, 0.3]
    )

    assert result["image_auroc"] is None


def test_no_positive_predictions():
    result = compute_image_metrics(
        labels=[0, 1],
        scores=[0.1, 0.2],
        threshold=0.5
    )

    assert result["precision"] == 0.0
    assert result["recall"] == 0.0
    assert result["f1"] == 0.0
    assert result["fn"] == 1


def test_threshold_is_strictly_greater():
    result = compute_image_metrics(
        labels=[0, 1],
        scores=[0.3, 0.3],
        threshold=0.3
    )

    assert result["tp"] == 0
    assert result["fp"] == 0


def test_invalid_scores_rejected():
    with pytest.raises(ValueError):
        compute_image_metrics(
            [0, 1],
            [0.1, np.nan]
        )


def test_invalid_labels_rejected():
    with pytest.raises(ValueError):
        compute_image_metrics(
            [0, 2],
            [0.1, 0.9]
        )


def test_length_mismatch_rejected():
    with pytest.raises(ValueError):
        compute_image_metrics(
            [0, 1],
            [0.1]
        )


# ============================================
# PIXEL METRICS
# ============================================

def test_perfect_pixel_auroc():
    masks = np.array([
        [[0, 0],
         [0, 0]],
        [[0, 1],
         [1, 0]],
    ])

    maps = np.array([
        [[0.1, 0.1],
         [0.2, 0.1]],
        [[0.2, 0.9],
         [0.8, 0.1]],
    ])

    result = compute_pixel_auroc(masks, maps)

    assert result == pytest.approx(1.0)


def test_normal_masks_without_positive_pixels():
    masks = np.zeros((3, 4, 4), dtype=np.uint8)
    maps = np.random.default_rng(42).random((3, 4, 4))

    result = compute_pixel_auroc(masks, maps)

    assert result is None


def test_single_channel_dimension_supported():
    masks = np.array([[[[0, 1], [0, 1]]]])
    maps = np.array([[[[0.1, 0.9], [0.2, 0.8]]]])

    assert compute_pixel_auroc(
        masks, maps
    ) == pytest.approx(1.0)


def test_mismatched_pixel_shapes_rejected():
    masks = np.zeros((2, 4, 4))
    maps = np.zeros((2, 8, 8))

    with pytest.raises(ValueError):
        compute_pixel_auroc(masks, maps)


def test_nonbinary_masks_rejected():
    masks = np.array([[[0, 2], [0, 1]]])
    maps = np.array([[[0.1, 0.2], [0.3, 0.4]]])

    with pytest.raises(ValueError):
        compute_pixel_auroc(masks, maps)


def test_nan_pixel_scores_rejected():
    masks = np.array([[[0, 1], [0, 1]]])
    maps = np.array([[[0.1, np.nan], [0.3, 0.4]]])

    with pytest.raises(ValueError):
        compute_pixel_auroc(masks, maps)


@pytest.mark.parametrize("bad", [np.inf, -np.inf, np.nan, 1+2j, True])
def test_invalid_continuous_score_values(bad):
    with pytest.raises(ValueError):
        compute_image_metrics([0, 1], np.array([bad, bad]))
    with pytest.raises(ValueError):
        compute_pixel_auroc([[[0, 1]]], np.array([[[bad, bad]]]))


@pytest.mark.parametrize("bad", [np.inf, np.nan, True, np.complex128(1+2j)])
def test_invalid_threshold(bad):
    with pytest.raises(ValueError):
        compute_image_metrics([0, 1], [0.1, 0.9], threshold=bad)


def test_ties_and_single_positive_class():
    assert compute_image_metrics([0, 1], [1., 1.])["image_auroc"] == 0.5
    result = compute_image_metrics([1, 1], [0.5, 0.7], threshold=0.6)
    assert result["image_auroc"] is None
    assert result["precision"] == 1.
    assert result["recall"] == 0.5
    assert result["f1"] == pytest.approx(2/3)
    assert compute_pixel_auroc([[[1, 1]]], [[[0.2, 0.3]]]) is None


def test_normal_only_classification_zero_division():
    result = compute_image_metrics([0, 0], [0.1, 0.2], threshold=0.5)
    assert result["image_auroc"] is None
    assert (result["precision"], result["recall"], result["f1"]) == (0., 0., 0.)
