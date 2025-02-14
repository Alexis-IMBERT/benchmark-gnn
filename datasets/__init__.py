"""Define custom PyTorch Lightning Dataset."""

from torch_geometric.data.lightning import LightningDataset
from torch_geometric.data import Dataset


class MyLyghtningDataset(LightningDataset):
    """
    A custom PyTorch Lightning Dataset.
    Allow to define Dataloader variable in the LightningModule directly in the config file.
    """

    def __init__(
        self,
        train_dataset,
        val_dataset: Dataset | None = None,
        test_dataset: Dataset | None = None,
        batch_size: int = 32,
        num_workers: int = 0,
        pin_memory: bool = False,
    ):
        super().__init__(
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            test_dataset=test_dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
