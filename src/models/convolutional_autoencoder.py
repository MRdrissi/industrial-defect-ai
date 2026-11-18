from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class CAEConfig:
    """Configuration de l'architecture CAE V1."""

    input_channels: int = 3
    image_size: tuple[int, int] = (256, 256)

    encoder_channels: tuple[int, int, int] = (
        32, 64, 128
    )
    latent_channels: int = 128

    seed: int = 42

    def __post_init__(self):
        if type(self.input_channels) is not int or self.input_channels != 3:
            raise ValueError("Le CAE attend des images RGB.")

        if (
                len(self.image_size) != 2
                or any(
            type(v) is not int or v <= 0 or v % 16 != 0
            for v in self.image_size
        )
        ):
            raise ValueError(
                "Les dimensions doivent être positives "
                "et multiples de 16."
            )

        if (
                len(self.encoder_channels) != 3
                or any(
            type(c) is not int or c <= 0
            for c in self.encoder_channels
        )
        ):
            raise ValueError(
                "encoder_channels doit contenir "
                "3 entiers positifs."
            )

        if type(self.latent_channels) is not int or self.latent_channels <= 0:
            raise ValueError("latent_channels invalide.")

        if type(self.seed) is not int:
            raise ValueError("La seed doit être un entier.")


class ConvolutionalAutoencoder(nn.Module):
    """Autoencodeur convolutionnel symétrique sans skip connections."""

    def __init__(self, config: CAEConfig):
        super().__init__()

        self.config = config

        c1, c2, c3 = config.encoder_channels
        latent = config.latent_channels

        # 256 -> 128 -> 64 -> 32
        self.encoder = nn.Sequential(
            nn.Conv2d(3, c1, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(c1, c2, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(c2, c3, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
        )

        # 32 -> 16
        self.bottleneck = nn.Sequential(
            nn.Conv2d(c3, latent, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
        )

        # 16 -> 32 -> 64 -> 128 -> 256
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(
                latent, c3, kernel_size=4, stride=2, padding=1
            ),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(
                c3, c2, kernel_size=4, stride=2, padding=1
            ),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(
                c2, c1, kernel_size=4, stride=2, padding=1
            ),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(
                c1, config.input_channels,
                kernel_size=4, stride=2, padding=1
            ),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError("Entrée attendue : [B, 3, H, W]")

        if x.shape[1] != self.config.input_channels:
            raise ValueError("Nombre de canaux incorrect")

        if tuple(x.shape[2:]) != self.config.image_size:
            raise ValueError("Dimensions spatiales incorrectes")

        features = self.encoder(x)
        latent = self.bottleneck(features)
        reconstruction = self.decoder(latent)

        return reconstruction
