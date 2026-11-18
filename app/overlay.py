"""Superposition de la carte d'anomalie sur l'image — US-06-04 (IDL-77).

Fonctions pures (NumPy + Pillow), testables sans Streamlit ni modèle :

1. ``resize_map``    : remet la carte à la taille de l'image d'origine.
   Les modèles réels produisent une carte 256 x 256 alors que les images
   MVTec font 900 x 900 (bottle) ou 1024 x 1024 (hazelnut, transistor) ;
   sans ce redimensionnement, la zone chaude serait décalée.
2. ``scale_for_display`` : ramène la carte dans [0, 1], indépendamment du seuil,
   avec une échelle relative à chaque image
   (minimum → 0, maximum → 1, ``scale_relative``).
3. ``colorize``      : transforme la carte [0, 1] en couleurs (bleu → rouge).
4. ``blend``         : pose la heatmap colorée sur l'image avec une opacité.

La heatmap est une aide visuelle : elle ne change ni le score ni la décision.
"""

from __future__ import annotations

import numpy as np
from PIL import Image

DEFAULT_ALPHA = 0.45

# Palette continue bleu → cyan → vert → jaune → rouge (positions dans [0, 1]).
_PALETTE_POSITIONS = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
_PALETTE_COLORS = np.array(
    [
        [0, 0, 160],
        [0, 170, 255],
        [40, 200, 70],
        [255, 220, 0],
        [220, 20, 20],
    ],
    dtype=np.float32,
)


def _as_2d_float(anomaly_map) -> np.ndarray:
    """Accepte un tenseur PyTorch CPU ou un tableau NumPy [H, W] (ou [1, 1, H, W])."""
    array = np.asarray(anomaly_map, dtype=np.float32)
    while array.ndim > 2 and array.shape[0] == 1:  # [1, 1, H, W] -> [H, W]
        array = array[0]
    if array.ndim != 2 or 0 in array.shape:
        raise ValueError(f"Carte d'anomalie [H, W] attendue, reçu {tuple(np.shape(anomaly_map))}.")
    if not np.isfinite(array).all():
        raise ValueError("La carte d'anomalie contient NaN ou Inf.")
    return array


def resize_map(anomaly_map, size: tuple[int, int]) -> np.ndarray:
    """Redimensionne la carte à ``size = (largeur, hauteur)`` de l'image (bilinéaire)."""
    width, height = size
    if width <= 0 or height <= 0:
        raise ValueError("Taille d'image invalide.")
    array = _as_2d_float(anomaly_map)
    if array.shape == (height, width):
        return array.copy()
    resized = Image.fromarray(array).resize((width, height), Image.Resampling.BILINEAR)
    return np.asarray(resized, dtype=np.float32)


def scale_to_threshold(anomaly_map, threshold: float) -> np.ndarray:
    """Échelle commune : 0 → 0, seuil → 0,5, deux fois le seuil ou plus → 1."""
    if not np.isfinite(threshold) or threshold <= 0:
        raise ValueError("Le seuil doit être un nombre strictement positif.")
    array = _as_2d_float(anomaly_map)
    return np.clip(array / (2.0 * threshold), 0.0, 1.0)


def scale_relative(anomaly_map) -> np.ndarray:
    """Échelle propre à l'image : minimum → 0, maximum → 1 (carte plate → 0)."""
    array = _as_2d_float(anomaly_map)
    low, high = float(array.min()), float(array.max())
    if high <= low:
        return np.zeros_like(array)
    return (array - low) / (high - low)


def scale_for_display(
    anomaly_map,
    threshold: float | None = None,
) -> tuple[np.ndarray, str]:
    return scale_relative(anomaly_map), "relative"


def colorize(scaled_map: np.ndarray) -> np.ndarray:
    """Carte [H, W] dans [0, 1] → image RGB uint8 [H, W, 3]."""
    values = np.clip(np.asarray(scaled_map, dtype=np.float32), 0.0, 1.0)
    channels = [
        np.interp(values, _PALETTE_POSITIONS, _PALETTE_COLORS[:, c]) for c in range(3)
    ]
    return np.stack(channels, axis=-1).round().astype(np.uint8)


def blend(image: Image.Image, heatmap_rgb: np.ndarray, alpha: float = DEFAULT_ALPHA) -> np.ndarray:
    """Superpose la heatmap à l'image : (1 - alpha) * image + alpha * heatmap."""
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("L'opacité doit être dans [0, 1].")
    base = np.asarray(image.convert("RGB"), dtype=np.float32)
    heat = np.asarray(heatmap_rgb, dtype=np.float32)
    if base.shape != heat.shape:
        raise ValueError(f"Tailles différentes : image {base.shape}, heatmap {heat.shape}.")
    return ((1.0 - alpha) * base + alpha * heat).round().astype(np.uint8)


def make_overlay(
    image: Image.Image,
    anomaly_map,
    threshold: float | None,
    alpha: float = DEFAULT_ALPHA,
) -> tuple[np.ndarray, np.ndarray, str]:
    """Retourne (heatmap colorée, superposition, échelle utilisée), à la taille de l'image."""
    resized = resize_map(anomaly_map, image.size)
    scaled, scale = scale_for_display(resized, threshold)
    heatmap = colorize(scaled)
    return heatmap, blend(image, heatmap, alpha), scale


def peak_position(anomaly_map, size: tuple[int, int] | None = None) -> tuple[int, int]:
    """Position (ligne, colonne) du maximum, après redimensionnement éventuel."""
    array = resize_map(anomaly_map, size) if size is not None else _as_2d_float(anomaly_map)
    row, col = np.unravel_index(int(np.argmax(array)), array.shape)
    return int(row), int(col)
