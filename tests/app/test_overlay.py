"""Tests de la superposition heatmap / image — US-06-04 (IDL-77).

Le point clé est l'alignement spatial : une zone chaude placée à un endroit
de la carte 256 x 256 doit retomber au même endroit relatif de l'image
d'origine, pour les tailles des trois catégories MVTec.
"""

import numpy as np
import pytest
from PIL import Image

from app.overlay import (
    blend,
    colorize,
    make_overlay,
    peak_position,
    resize_map,
    scale_for_display,
    scale_relative,
    scale_to_threshold,
)

# Taille d'origine des images MVTec AD (US-02-01) et taille des cartes des modèles
CATEGORY_SIZES = {"bottle": 900, "hazelnut": 1024, "transistor": 1024}
MAP_SIZE = 256


def hot_spot_map(row, col, sigma=4.0, size=MAP_SIZE):
    """Carte 256 x 256 avec une zone chaude gaussienne, maximale en (row, col)."""
    rows, cols = np.mgrid[0:size, 0:size]
    distance2 = (rows - row) ** 2 + (cols - col) ** 2
    return np.exp(-distance2 / (2 * sigma**2)).astype(np.float32)


@pytest.mark.parametrize("category, side", CATEGORY_SIZES.items())
def test_hot_zone_stays_aligned_after_resize(category, side):
    # Zone chaude hors du centre pour détecter une inversion ou un décalage d'axes
    row, col = 64, 192
    resized = resize_map(hot_spot_map(row, col), (side, side))
    peak_row, peak_col = peak_position(resized)

    scale = side / MAP_SIZE
    tolerance = 2 * scale  # deux pixels de la carte, convertis en pixels de l'image
    assert resized.shape == (side, side)
    assert abs(peak_row - row * scale) <= tolerance
    assert abs(peak_col - col * scale) <= tolerance


@pytest.mark.parametrize("category, side", CATEGORY_SIZES.items())
def test_overlay_has_image_size_for_each_category(category, side):
    image = Image.new("RGB", (side, side), (120, 120, 120))
    heatmap, overlay, scale = make_overlay(image, hot_spot_map(128, 128), threshold=0.5)

    assert scale == "threshold"
    assert heatmap.shape == (side, side, 3)
    assert overlay.shape == (side, side, 3)
    assert overlay.dtype == np.uint8


def test_non_square_image_keeps_width_and_height():
    image = Image.new("RGB", (640, 480))
    heatmap, overlay, _ = make_overlay(image, np.zeros((256, 256)), threshold=1.0)

    assert overlay.shape == (480, 640, 3)
    assert peak_position(hot_spot_map(30, 220), (640, 480)) == pytest.approx((30 * 480 / 256, 220 * 640 / 256), abs=5)


def test_map_already_at_image_size_is_unchanged():
    anomaly_map = np.random.default_rng(0).random((50, 70), dtype=np.float32)
    assert np.array_equal(resize_map(anomaly_map, (70, 50)), anomaly_map)


def test_model_output_shapes_are_accepted():
    # PatchCore renvoie [1, 1, 256, 256], le contrat [H, W] ; un tenseur CPU passe par np.asarray
    assert resize_map(np.zeros((1, 1, 256, 256)), (900, 900)).shape == (900, 900)


def test_threshold_is_the_middle_of_the_color_scale():
    scaled = scale_to_threshold(np.array([[0.0, 0.2, 0.4, 0.8]]), threshold=0.4)
    assert scaled.tolist() == [[0.0, 0.25, 0.5, 1.0]]


def test_missing_threshold_uses_relative_scale():
    # Modèle réel sans seuil calibré (US-06-03) : couleurs relatives à l'image
    scaled, scale = scale_for_display(np.array([[2.0, 4.0, 6.0]]), threshold=None)
    assert scale == "relative"
    assert scaled.tolist() == [[0.0, 0.5, 1.0]]

    image = Image.new("RGB", (900, 900))
    _, overlay, scale = make_overlay(image, hot_spot_map(64, 192), threshold=None)
    assert scale == "relative"
    assert overlay.shape == (900, 900, 3)


def test_flat_map_does_not_divide_by_zero():
    assert np.all(scale_relative(np.full((8, 8), 3.0)) == 0.0)


def test_colors_go_from_blue_to_red():
    colors = colorize(np.array([[0.0, 1.0]]))
    low, high = colors[0, 0], colors[0, 1]
    assert low[2] > low[0]    # bas de l'échelle : bleu
    assert high[0] > high[2]  # haut de l'échelle : rouge


def test_blend_respects_opacity():
    image = Image.new("RGB", (4, 4), (100, 100, 100))
    heat = np.full((4, 4, 3), 200, dtype=np.uint8)

    assert np.all(blend(image, heat, alpha=0.0) == 100)
    assert np.all(blend(image, heat, alpha=1.0) == 200)
    assert np.all(blend(image, heat, alpha=0.5) == 150)


@pytest.mark.parametrize(
    "call",
    [
        lambda: resize_map(np.zeros((3, 8, 8)), (8, 8)),
        lambda: resize_map(np.array([[np.nan]]), (4, 4)),
        lambda: resize_map(np.zeros((8, 8)), (0, 8)),
        lambda: scale_to_threshold(np.zeros((8, 8)), 0.0),
        lambda: blend(Image.new("RGB", (4, 4)), np.zeros((4, 4, 3)), alpha=1.5),
        lambda: blend(Image.new("RGB", (4, 4)), np.zeros((5, 5, 3))),
    ],
)
def test_invalid_inputs_are_rejected(call):
    with pytest.raises(ValueError):
        call()
