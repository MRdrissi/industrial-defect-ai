import torch
from torch.utils.data import DataLoader

from src.data.mvtec_dataset import MVTecDataset


def create_dataloader(
        dataset: MVTecDataset,
        batch_size: int = 16,
        num_workers: int = 0,
        seed: int = 42,
) -> DataLoader:
    """
    Construit un DataLoader reproductible pour MVTec AD.

    - train : mélange déterministe pour une seed donnée
    - val/test : ordre du manifeste conservé
    - aucun échantillon supprimé
    """

    if dataset.split not in ("train", "val", "test"):
        raise ValueError("Split non reconnu")

    if type(batch_size) is not int or batch_size <= 0:
        raise ValueError("batch_size doit être un entier positif")

    if type(num_workers) is not int or num_workers < 0:
        raise ValueError("num_workers doit être un entier >= 0")

    if type(seed) is not int:
        raise ValueError("seed doit être un entier")

    generator = torch.Generator()
    generator.manual_seed(seed)

    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=(dataset.split == "train"),
        num_workers=num_workers,
        drop_last=False,
        pin_memory=False,
        generator=generator,
    )
