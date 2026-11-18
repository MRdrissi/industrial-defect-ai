"""
Tests de l'inférence CAE (US-03-05, ST-03-05-02 / IDL-36).

Vérifie, sans dataset ni GPU :
  - les dimensions et le contrat des sorties pour les trois catégories
    (images d'origine 900×900 pour bottle, 1024×1024 pour hazelnut et transistor) ;
  - le rechargement d'un checkpoint et les erreurs de chargement ;
  - la cohérence de la carte d'anomalie et du score (localisation, lissage, max / mean) ;
  - le refus des entrées invalides.

Lancement (depuis la racine du dépôt) :
    python -m unittest discover -s tests/inference -p "test_*.py" -v
"""

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import nn

from src.data.transforms import MVTecTransform
from src.inference.cae_inference import (
    CAEInference,
    CAEInferenceConfig,
    extract_state_dict,
    load_cae,
)
from src.models.convolutional_autoencoder import CAEConfig, ConvolutionalAutoencoder

REPO_ROOT = Path(__file__).resolve().parents[2]
BASELINE_CONFIG = REPO_ROOT / "experiments" / "configs" / "cae_baseline.json"

# Taille d'origine des images MVTec AD de chaque catégorie (US-02-01)
CATEGORY_SIZES = {"bottle": 900, "hazelnut": 1024, "transistor": 1024}

CONTRACT_KEYS = {
    "category", "model_family", "model_name", "score", "anomaly_map_shape",
    "score_method", "smoothing_sigma", "mean_mse",
}


class _FixedModel(nn.Module):
    """Faux modèle déterministe : la « reconstruction » est donnée par une fonction."""

    def __init__(self, fn):
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(1))  # pour que le moteur trouve un device
        self.fn = fn

    def forward(self, x):
        return self.fn(x)


def _baseline_model() -> ConvolutionalAutoencoder:
    """CAE de référence (256×256) non entraîné, initialisé de façon reproductible."""
    torch.manual_seed(42)
    return ConvolutionalAutoencoder(CAEConfig())


def _mvtec_like_image(size: int, seed: int) -> torch.Tensor:
    """Image RGB aléatoire de la taille d'origine, passée par le prétraitement CAE."""
    rng = np.random.default_rng(seed)
    image = Image.fromarray(rng.integers(0, 256, (size, size, 3), dtype=np.uint8))
    mask = Image.new("L", (size, size), 0)
    tensor, _ = MVTecTransform(size=(256, 256), profile="cae")(image, mask)
    return tensor


class TestCAEInferenceOutputs(unittest.TestCase):
    """Dimensions et contrat des sorties sur les trois catégories."""

    @classmethod
    def setUpClass(cls):
        cls.model = _baseline_model()

    def test_outputs_for_each_category(self):
        for i, (category, size) in enumerate(CATEGORY_SIZES.items()):
            with self.subTest(category=category):
                engine = CAEInference(self.model, category, "cae_baseline_v1")
                result = engine.predict(_mvtec_like_image(size, seed=i))

                self.assertEqual(result.category, category)
                self.assertEqual(result.model_family, "cae")
                self.assertEqual(tuple(result.anomaly_map.shape), (256, 256))
                self.assertEqual(tuple(result.reconstruction.shape), (3, 256, 256))
                self.assertEqual(result.anomaly_map.dtype, torch.float32)
                self.assertEqual(result.anomaly_map.device.type, "cpu")
                self.assertGreaterEqual(result.anomaly_map.min().item(), 0.0)
                self.assertGreaterEqual(result.reconstruction.min().item(), 0.0)
                self.assertLessEqual(result.reconstruction.max().item(), 1.0)
                self.assertTrue(np.isfinite(result.score) and result.score >= 0.0)

    def test_result_contract_is_json_serializable(self):
        engine = CAEInference(self.model, "transistor", "cae_baseline_v1")
        report = engine.predict(_mvtec_like_image(1024, seed=7)).to_dict()

        self.assertEqual(set(report), CONTRACT_KEYS)
        self.assertEqual(report["anomaly_map_shape"], [256, 256])
        self.assertEqual(report["score_method"], "max")
        json.dumps(report)  # ne doit pas lever d'erreur

    def test_batch_matches_single_predictions(self):
        engine = CAEInference(self.model, "bottle")
        images = torch.stack([_mvtec_like_image(900, seed=s) for s in range(3)])

        batch_results = engine.predict_batch(images)
        self.assertEqual(len(batch_results), 3)
        for image, batch_result in zip(images, batch_results):
            single = engine.predict(image)
            self.assertAlmostEqual(single.score, batch_result.score, places=6)
            self.assertTrue(torch.allclose(single.anomaly_map, batch_result.anomaly_map, atol=1e-6))


class TestCAECheckpointLoading(unittest.TestCase):
    """Rechargement d'un checkpoint sans réentraînement."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.tmp_dir = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_reloaded_checkpoint_gives_identical_results(self):
        model = _baseline_model()
        checkpoint = self.tmp_dir / "best_model.pt"
        torch.save(model.state_dict(), checkpoint)

        reference = CAEInference(model, "hazelnut")
        reloaded = CAEInference.from_checkpoint("hazelnut", BASELINE_CONFIG, checkpoint)
        image = _mvtec_like_image(1024, seed=3)

        self.assertEqual(reloaded.model_name, "cae_baseline_v1")
        self.assertFalse(reloaded.model.training)
        self.assertEqual(reference.predict(image).score, reloaded.predict(image).score)
        self.assertTrue(torch.equal(
            reference.predict(image).reconstruction, reloaded.predict(image).reconstruction
        ))

    def test_full_checkpoint_format_is_supported(self):
        """Format bottle / hazelnut : poids sous 'model_state_dict' + métadonnées."""
        model = _baseline_model()
        checkpoint = self.tmp_dir / "bottle_best.pth"
        torch.save(
            {
                "epoch": 495,
                "best_val_mse": 0.000368,
                "git_commit": "abc1234",
                "model_config": {"latent_channels": 128, "image_size": [256, 256]},
                "model_state_dict": model.state_dict(),
            },
            checkpoint,
        )

        reloaded = CAEInference.from_checkpoint("bottle", BASELINE_CONFIG, checkpoint)
        image = _mvtec_like_image(900, seed=5)
        self.assertEqual(CAEInference(model, "bottle").predict(image).score,
                         reloaded.predict(image).score)

    def test_unknown_checkpoint_format_is_rejected(self):
        with self.assertRaises(KeyError):
            extract_state_dict({"epoch": 10, "optimizer": {}})
        with self.assertRaises(TypeError):
            extract_state_dict([1, 2, 3])

    def test_missing_files_raise_errors(self):
        checkpoint = self.tmp_dir / "best_model.pt"
        torch.save(_baseline_model().state_dict(), checkpoint)

        with self.assertRaises(FileNotFoundError):
            load_cae(self.tmp_dir / "absent.json", checkpoint)
        with self.assertRaises(FileNotFoundError):
            load_cae(BASELINE_CONFIG, self.tmp_dir / "absent.pt")

    def test_checkpoint_with_other_architecture_is_rejected(self):
        other = ConvolutionalAutoencoder(CAEConfig(latent_channels=64))
        checkpoint = self.tmp_dir / "other.pt"
        torch.save(other.state_dict(), checkpoint)

        with self.assertRaises(RuntimeError):  # chargement strict : tailles incompatibles
            load_cae(BASELINE_CONFIG, checkpoint)


class TestAnomalyMapAndScore(unittest.TestCase):
    """Cohérence de la carte d'anomalie et du score, avec des reconstructions contrôlées."""

    SIZE = 64

    def _engine(self, fn, **config):
        model = _FixedModel(fn)
        return CAEInference(model, "transistor", "fixed", CAEInferenceConfig(**config))

    @staticmethod
    def _erase_block(x, rows, cols):
        """Reconstruction qui « oublie » un bloc : erreur maximale dans ce bloc uniquement."""
        out = x.clone()
        out[:, :, rows, cols] = 0.0
        return out

    def test_perfect_reconstruction_gives_zero_map_and_score(self):
        engine = self._engine(lambda x: x)
        result = engine.predict(torch.rand(3, self.SIZE, self.SIZE))

        self.assertEqual(result.anomaly_map.abs().max().item(), 0.0)
        self.assertEqual(result.score, 0.0)
        self.assertEqual(result.extras["mean_mse"], 0.0)

    def test_score_methods_without_smoothing(self):
        image = torch.ones(3, self.SIZE, self.SIZE)
        block = (slice(20, 28), slice(36, 44))  # 8×8 = 64 pixels en erreur
        fn = lambda x: self._erase_block(x, *block)

        max_result = self._engine(fn, smoothing_sigma=0, score_method="max").predict(image)
        mean_result = self._engine(fn, smoothing_sigma=0, score_method="mean").predict(image)

        self.assertAlmostEqual(max_result.score, 1.0, places=6)
        self.assertAlmostEqual(mean_result.score, 64 / self.SIZE**2, places=6)
        self.assertAlmostEqual(max_result.extras["mean_mse"], mean_result.score, places=6)

    def test_anomaly_map_localizes_the_defect(self):
        image = torch.ones(3, self.SIZE, self.SIZE)
        rows, cols = slice(20, 28), slice(36, 44)
        result = self._engine(lambda x: self._erase_block(x, rows, cols)).predict(image)

        peak_row, peak_col = divmod(int(result.anomaly_map.argmax()), self.SIZE)
        self.assertTrue(rows.start <= peak_row < rows.stop)
        self.assertTrue(cols.start <= peak_col < cols.stop)

    def test_smoothing_attenuates_isolated_pixels_more_than_regions(self):
        image = torch.ones(3, self.SIZE, self.SIZE)
        pixel = self._engine(lambda x: self._erase_block(x, slice(32, 33), slice(32, 33)))
        region = self._engine(lambda x: self._erase_block(x, slice(28, 36), slice(28, 36)))

        pixel_score = pixel.predict(image).score
        region_score = region.predict(image).score
        self.assertGreater(pixel_score, 0.0)
        self.assertLess(pixel_score, 0.1 * region_score)


class TestInvalidInputs(unittest.TestCase):
    """Entrées et réglages refusés."""

    def setUp(self):
        self.engine = CAEInference(_FixedModel(lambda x: x), "bottle")

    def test_wrong_number_of_dimensions(self):
        with self.assertRaises(ValueError):
            self.engine.predict(torch.rand(1, 3, 64, 64))
        with self.assertRaises(ValueError):
            self.engine.predict_batch(torch.rand(3, 64, 64))

    def test_values_outside_unit_range(self):
        with self.assertRaises(ValueError):
            self.engine.predict(torch.full((3, 64, 64), 2.0))
        with self.assertRaises(ValueError):
            self.engine.predict(torch.full((3, 64, 64), -0.5))

    def test_wrong_spatial_size_for_real_cae(self):
        engine = CAEInference(_baseline_model(), "bottle")
        with self.assertRaises(ValueError):
            engine.predict(torch.rand(3, 128, 128))

    def test_invalid_inference_config(self):
        with self.assertRaises(ValueError):
            CAEInferenceConfig(smoothing_sigma=-1.0)
        with self.assertRaises(ValueError):
            CAEInferenceConfig(score_method="median")


if __name__ == "__main__":
    unittest.main()
