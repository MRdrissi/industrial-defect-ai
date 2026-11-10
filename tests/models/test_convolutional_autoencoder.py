import unittest

import torch
from torch import nn

from src.models.convolutional_autoencoder import (
    CAEConfig,
    ConvolutionalAutoencoder,
)


class TestConvolutionalAutoencoder(unittest.TestCase):

    def setUp(self):
        self.config = CAEConfig(
            image_size=(64, 64),
            seed=42,
        )

        torch.manual_seed(self.config.seed)

        self.model = ConvolutionalAutoencoder(
            self.config
        ).cpu()

    def test_forward_dimensions(self):
        self.model.eval()

        images = torch.rand(2, 3, 64, 64)

        with torch.no_grad():
            reconstruction = self.model(images)

        self.assertEqual(
            tuple(reconstruction.shape),
            tuple(images.shape),
        )

        self.assertTrue(
            torch.isfinite(reconstruction).all().item()
        )

        self.assertGreaterEqual(
            reconstruction.min().item(), 0.0
        )

        self.assertLessEqual(
            reconstruction.max().item(), 1.0
        )

    def test_encoder_and_bottleneck_dimensions(self):
        images = torch.rand(2, 3, 64, 64)

        with torch.no_grad():
            features = self.model.encoder(images)
            latent = self.model.bottleneck(features)

        self.assertEqual(
            tuple(features.shape),
            (2, 128, 8, 8),
        )

        self.assertEqual(
            tuple(latent.shape),
            (2, 128, 4, 4),
        )

    def test_configurable_architecture(self):
        config = CAEConfig(
            image_size=(64, 64),
            encoder_channels=(16, 32, 64),
            latent_channels=32,
        )

        model = ConvolutionalAutoencoder(config)

        with torch.no_grad():
            output = model(torch.rand(1, 3, 64, 64))

        self.assertEqual(
            tuple(output.shape),
            (1, 3, 64, 64),
        )

    def test_invalid_configuration_and_input(self):
        with self.assertRaises(ValueError):
            CAEConfig(input_channels=3.0)

        with self.assertRaises(ValueError):
            CAEConfig(image_size=(65, 64))

        with self.assertRaises(ValueError):
            CAEConfig(latent_channels=0)

        with self.assertRaises(ValueError):
            self.model(torch.rand(2, 1, 64, 64))

        with self.assertRaises(ValueError):
            self.model(torch.rand(2, 3, 128, 128))

    def test_mse_backward_and_optimizer(self):
        self.model.train()

        images = torch.rand(2, 3, 64, 64)

        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=1e-3,
        )

        criterion = nn.MSELoss()

        before = [
            p.detach().clone()
            for p in self.model.parameters()
        ]

        optimizer.zero_grad(set_to_none=True)

        reconstruction = self.model(images)
        loss = criterion(reconstruction, images)

        self.assertTrue(torch.isfinite(loss).item())

        loss.backward()

        gradients = [
            p.grad for p in self.model.parameters()
        ]

        self.assertTrue(
            all(grad is not None for grad in gradients)
        )

        self.assertTrue(
            all(
                torch.isfinite(grad).all().item()
                for grad in gradients
            )
        )

        self.assertTrue(
            any(
                torch.any(grad != 0).item()
                for grad in gradients
            )
        )

        optimizer.step()

        self.assertTrue(
            any(
                not torch.equal(old, new)
                for old, new in zip(
                    before,
                    self.model.parameters(),
                )
            )
        )

    def test_reproducible_initialization(self):
        torch.manual_seed(42)
        first = ConvolutionalAutoencoder(self.config)

        torch.manual_seed(42)
        second = ConvolutionalAutoencoder(self.config)

        for p1, p2 in zip(
                first.parameters(),
                second.parameters(),
        ):
            self.assertTrue(torch.equal(p1, p2))


    def test_reference_configuration(self):
        import json
        from pathlib import Path

        repo_root = Path(__file__).resolve().parents[2]

        config_path = (
                repo_root
                / "experiments"
                / "configs"
                / "cae_baseline.json"
        )

        with config_path.open(
                "r", encoding="utf-8"
        ) as file:
            payload = json.load(file)

        params = payload["config"]

        config = CAEConfig(
            input_channels=params["input_channels"],
            image_size=tuple(params["image_size"]),
            encoder_channels=tuple(params["encoder_channels"]),
            latent_channels=params["latent_channels"],
            seed=params["seed"],
        )

        torch.manual_seed(config.seed)
        model_a = ConvolutionalAutoencoder(config)

        torch.manual_seed(config.seed)
        model_b = ConvolutionalAutoencoder(config)

        self.assertEqual(config.image_size, (256, 256))

        self.assertEqual(
            sum(p.numel() for p in model_a.parameters()),
            855619,
        )

        for first, second in zip(
                model_a.parameters(),
                model_b.parameters(),
        ):
            self.assertTrue(torch.equal(first, second))


if __name__ == "__main__":
    unittest.main()
