import numpy as np
import pytest

from src.evaluation.masks import (
    resize_binary_masks,
    resize_anomaly_maps,
    prepare_ground_truth_mask,
    align_pixel_inputs,
)
from src.evaluation.metrics import compute_pixel_auroc


def test_binary_mask_nearest_neighbor():
    mask = np.array([
        [0, 1],
        [0, 0]
    ], dtype=np.uint8)

    result = resize_binary_masks(mask, (4, 4))

    expected = np.array([
        [0, 0, 1, 1],
        [0, 0, 1, 1],
        [0, 0, 0, 0],
        [0, 0, 0, 0]
    ], dtype=np.uint8)

    assert result.shape == (1, 4, 4)
    np.testing.assert_array_equal(result[0], expected)


def test_normal_image_without_mask():
    result = prepare_ground_truth_mask(
        mask=None,
        label=0,
        target_size=(256, 256)
    )

    assert result.shape == (256, 256)
    assert result.dtype == np.uint8
    assert np.count_nonzero(result) == 0


def test_anomalous_image_requires_mask():
    with pytest.raises(ValueError):
        prepare_ground_truth_mask(
            mask=None,
            label=1,
            target_size=(256, 256)
        )


def test_normal_image_rejects_positive_mask():
    mask = np.array([[0, 1], [0, 0]])

    with pytest.raises(ValueError):
        prepare_ground_truth_mask(mask, 0, (4, 4))


def test_anomalous_image_rejects_empty_mask():
    mask = np.zeros((2, 2), dtype=np.uint8)

    with pytest.raises(ValueError):
        prepare_ground_truth_mask(mask, 1, (4, 4))


def test_bilinear_map_preserves_continuous_scores():
    anomaly_map = np.array([
        [0.0, 1.0],
        [0.0, 0.0]
    ], dtype=np.float32)

    result = resize_anomaly_maps(
        anomaly_map, (4, 4)
    )

    assert result.shape == (1, 4, 4)
    assert result.dtype == np.float64

    # Interior pixel contains an interpolated value.
    assert result[0, 1, 2] == pytest.approx(0.5625)


def test_anomaly_map_nan_rejected():
    anomaly_map = np.array([
        [0.1, np.nan],
        [0.2, 0.3]
    ])

    with pytest.raises(ValueError):
        resize_anomaly_maps(anomaly_map, (4, 4))


def test_align_different_spatial_resolutions():
    masks = np.array([
        [[0, 1],
         [0, 0]]
    ])

    maps = np.ones((1, 4, 4), dtype=np.float32)

    aligned_masks, aligned_maps = align_pixel_inputs(
        masks, maps
    )

    # By default, align to ground-truth size.
    assert aligned_masks.shape == (1, 2, 2)
    assert aligned_maps.shape == (1, 2, 2)

    assert set(np.unique(aligned_masks)) <= {0, 1}


def test_explicit_target_resolution():
    masks = np.array([
        [[0, 1],
         [0, 0]]
    ])

    maps = np.array([
        [[0.1, 0.9],
         [0.2, 0.3]]
    ])

    aligned_masks, aligned_maps = align_pixel_inputs(
        masks, maps, target_size=(4, 4)
    )

    assert aligned_masks.shape == (1, 4, 4)
    assert aligned_maps.shape == (1, 4, 4)


def test_batch_count_mismatch_rejected():
    masks = np.zeros((2, 4, 4))
    maps = np.zeros((3, 4, 4))

    with pytest.raises(ValueError):
        align_pixel_inputs(masks, maps)


def test_invalid_target_size_rejected():
    masks = np.zeros((1, 2, 2))

    with pytest.raises(ValueError):
        resize_binary_masks(masks, (0, 256))


def test_alignment_compatible_with_pixel_auroc():
    masks = np.array([
        [[0, 0],
         [0, 0]],
        [[0, 1],
         [0, 0]],
    ], dtype=np.uint8)

    maps = np.array([
        [[0.1, 0.1],
         [0.1, 0.1]],
        [[0.1, 0.9],
         [0.1, 0.1]],
    ], dtype=np.float32)

    aligned_masks, aligned_maps = align_pixel_inputs(
        masks, maps
    )

    auroc = compute_pixel_auroc(
        aligned_masks, aligned_maps
    )

    assert auroc == pytest.approx(1.0)


def test_small_positive_mask_must_not_silently_disappear():
    mask = np.zeros((4, 4), dtype=np.uint8)
    mask[0, 0] = 1
    for operation in (
        lambda: resize_binary_masks(mask, (1, 1)),
        lambda: prepare_ground_truth_mask(mask, 1, (1, 1)),
        lambda: align_pixel_inputs(mask, np.zeros((4, 4)), (1, 1)),
    ):
        with pytest.raises(ValueError, match="disappeared"):
            operation()


def test_map_precision_preserved_without_resize():
    values = np.array([[1., 1. + 1e-9]], dtype=np.float64)
    actual = resize_anomaly_maps(values, (1, 2))
    np.testing.assert_array_equal(actual[0], values)
    assert compute_pixel_auroc([[[0, 1]]], actual) == 1.0


def test_bilinear_resize_preserves_small_differences():
    values = np.array([[1., 1. + 1e-9], [1., 1. + 1e-9]])
    actual = resize_anomaly_maps(values, (4, 4))
    expected = 1. + 1e-9 * np.array([0., .25, .75, 1.])
    np.testing.assert_allclose(actual[0], np.tile(expected, (4, 1)), rtol=0, atol=1e-15)


def test_bilinear_downsampling_antialias_reference():
    # A separable ramp: downsampling 4 -> 2 with antialias weights
    # [3, 3, 1]/7 at the boundary gives centers 5/7 and 16/7.
    values = np.tile(np.arange(4, dtype=np.float64), (4, 1))
    actual = resize_anomaly_maps(values, (2, 2))
    np.testing.assert_allclose(actual[0], [[5/7, 16/7], [5/7, 16/7]])


@pytest.mark.parametrize("bad", [np.inf, np.nan, 1+2j, True])
def test_invalid_map_values(bad):
    with pytest.raises(ValueError):
        resize_anomaly_maps(np.full((2, 2), bad), (4, 4))
