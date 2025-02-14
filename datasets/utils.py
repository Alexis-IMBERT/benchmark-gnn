import torch
from torch_geometric.data import Data, Dataset
import torch_geometric.data.lightning


class MyLightningDataset(torch_geometric.data.lightning.LightningDataset):
    def __init__(
        self,
        train_dataset: Dataset,
        val_dataset: Dataset,
        test_dataset: Dataset,
        num_workers: int,
        batch_size: int,
        pin_memory: bool,
    ):
        super().__init__(
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            test_dataset=test_dataset,
            num_workers=num_workers,
            batch_size=batch_size,
            pin_memory=pin_memory,
        )


def y_to_torch_float(data: Data) -> Data:
    data.y = data.y.to(torch.float)
    return data
