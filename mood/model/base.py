import abc
from functools import partial

import torch
from scipy.stats import entropy
from torch import nn
from itertools import tee
from typing import Optional, Union
from pytorch_lightning import LightningModule


class BaseModel(LightningModule, abc.ABC):
    def __init__(
        self,
        base_network: nn.Module,
        prediction_head: nn.Module,
        loss_fn: nn.Module,
        lr: float,
        weight_decay: Union[float, str],
        batch_size: int,
    ):
        super().__init__()
        self.base_network = base_network
        self.prediction_head = prediction_head
        self.loss_fn = partial(self.loss_function_wrapper, loss_fn=loss_fn)
        self.l2 = weight_decay
        self.lr = lr
        self.batch_size = batch_size

    def forward(self, x, domains: Optional = None, return_embedding: bool = False):
        embedding = self.base_network(x)
        label = self.prediction_head(embedding)
        out = (label, embedding) if return_embedding else label
        return out

    def training_step(
        self, batch, batch_idx, optimizer_idx: Optional[int] = None, dataset_idx: Optional[int] = None
    ):
        return self._step(batch, batch_idx, optimizer_idx)

    def validation_step(self, batch, batch_idx, dataset_idx: Optional[int] = None):
        return self._step(batch, batch_idx, optimizer_idx=None)

    def predict(self, dataloader):
        self.training = False
        with torch.inference_mode():
            res = torch.cat([self.forward(*X) for X, y in dataloader], dim=0)
        return res

    @staticmethod
    def loss_function_wrapper(preds, targets, loss_fn):
        if preds.ndim > 1:
            preds = preds.squeeze(dim=-1)
        targets = targets.float()
        return loss_fn(preds, targets)

    @abc.abstractmethod
    def _step(self, batch, batch_idx, optimizer_idx: Optional[int] = None):
        raise NotImplementedError

    def configure_optimizers(self):
        if self.l2 == "auto":
            parameters, parameters_copy = tee(self.parameters())
            self.l2 = 1.0 / sum(p.numel() for p in parameters_copy if p.requires_grad)

        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.l2)
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": lr_scheduler,
                "interval": "epoch",
                "frequency": 1,
                "monitor": "val_loss",
                "strict": True,
            },
        }

    def log(self, name: str, *args, **kwargs):
        prefix = "train_" if self.training else "val_"
        super(BaseModel, self).log(prefix + name, *args, batch_size=self.batch_size, **kwargs)

    @staticmethod
    def suggest_params(trial):
        width = trial.suggest_categorical("mlp_width", [64, 128, 256, 512])
        depth = trial.suggest_int("mlp_depth", 1, 5)
        lr = trial.suggest_float("lr", 1e-8, 1.0, log=True)
        weight_decay = trial.suggest_categorical(
            "weight_decay", ["auto", 0.0, 0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1, 1.0]
        )
        return {"mlp_width": width, "mlp_depth": depth, "lr": lr, "weight_decay": weight_decay}


class Ensemble(LightningModule):
    def __init__(self, models, is_regression):
        super().__init__()
        self.models = models
        self.is_regression = is_regression

    def predict(self, dataloader):
        return torch.stack([model.predict(dataloader) for model in self.models]).mean(dim=0)

    def predict_uncertainty(self, dataloader):

        if self.is_regression:
            return torch.stack([model.predict(dataloader) for model in self.models]).var(dim=0)

        else:
            proba = self.predict(dataloader)[:, 0]
            x_0 = torch.clip(proba, 1e-10, 1.0 - 1e-10).detach().cpu().numpy()
            x_1 = 1.0 - x_0
            return entropy([x_0, x_1], base=2)
