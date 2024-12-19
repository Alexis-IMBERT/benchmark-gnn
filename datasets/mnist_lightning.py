"""mnist dataset for PyTorch Lightning"""

import os

import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor


class MNISTDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_dir: str = None,
        batch_size: int = 32,
        num_workers: int = 4,
    ):
        super().__init__()
        self.data_dir = data_dir if data_dir is not None else os.getcwd()
        self.batch_size = batch_size
        self.transform = ToTensor()
        self.num_workers = num_workers

    def prepare_data(self):
        # download
        MNIST(self.data_dir, train=True, download=True)
        MNIST(self.data_dir, train=False, download=True)

    def setup(self, stage=None):
        # Assign train/val datasets for use in dataloaders
        if stage == "fit" or stage is None:
            mnist_full = MNIST(self.data_dir, train=True, transform=self.transform)
            train_set_size = int(len(mnist_full) * 0.8)
            valid_set_size = len(mnist_full) - train_set_size
            self.mnist_train, self.mnist_val = random_split(
                mnist_full, [train_set_size, valid_set_size]
            )

        # Assign test dataset for use in dataloader(s)
        if stage == "test" or stage is None:
            self.mnist_test = MNIST(
                self.data_dir, train=False, transform=self.transform
            )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.mnist_train,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.mnist_val,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.mnist_test,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )
