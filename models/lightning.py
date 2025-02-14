# define the LightningModule
import pytorch_lightning as pl
from torch import optim, ones_like
from torch.nn import Module
from torcheval.metrics import BinaryF1Score
import wandb


class GraphLitModel(pl.LightningModule):
    def __init__(
        self,
        name: str,
        module: Module,
        loss_fn: Module,
        learning_rate: float = 1e-3,
    ) -> None:
        super().__init__()
        self.name = name
        self.model = module
        self.loss_fn = loss_fn
        self.learning_rate = learning_rate
        self.save_hyperparameters()

    def training_step(self, batch):
        pred = self.model(batch.x, batch.edge_index, batch.batch)
        loss = self.loss_fn(pred, batch.y)
        self.log("train/loss", loss, on_epoch=True)
        return loss

    def test_step(self, batch):
        # this is the test loop
        pred = self.model(batch.x, batch.edge_index, batch.batch)
        loss = self.loss_fn(pred, batch.y)
        self.log("test/loss", loss, on_epoch=True)

        # Compute F1 score
        f1_metric = BinaryF1Score()
        pred_labels = pred.argmax(dim=1)
        f1_metric.update(pred_labels, batch.y)
        f1_score = f1_metric.compute()
        self.log("test/f1_score", f1_score, on_epoch=True)

        # Compute accuracy, recall, and precision
        accuracy = (pred_labels == batch.y).float().mean()
        self.log("test/accuracy", accuracy, on_epoch=True)

        true_positive = ((pred_labels == 1) & (batch.y == 1)).sum().float()
        false_positive = ((pred_labels == 1) & (batch.y == 0)).sum().float()
        false_negative = ((pred_labels == 0) & (batch.y == 1)).sum().float()

        precision = true_positive / (true_positive + false_positive)
        recall = true_positive / (true_positive + false_negative)

        self.log("test/precision", precision, on_epoch=True)
        self.log("test/recall", recall, on_epoch=True)

    def validation_step(self, batch):
        if self.trainer.global_step == 0:
            wandb.define_metric("val/f1_score", summary="max")
            wandb.define_metric("val/loss", summary="min")
        pred = self.model(batch.x, batch.edge_index, batch.batch)
        loss = self.loss_fn(pred, batch.y)
        self.log("val/loss", loss, on_epoch=True)
        f1_metric = BinaryF1Score()
        pred_labels = pred.argmax(dim=1)
        f1_metric.update(pred_labels, batch.y)
        f1_score = f1_metric.compute()
        self.log("val/f1_score", f1_score, on_epoch=True)

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=(self.learning_rate))


class XasOne(GraphLitModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def training_step(self, batch):
        batch.x = ones_like(batch.x)
        return super().training_step(batch)
