import abc
import torch

from itertools import tee
from typing import Optional, Union
from pytorch_lightning import Trainer, LightningModule
from sklearn.utils.multiclass import type_of_target


class BaseModel(LightningModule, abc.ABC):

    def __init__(self, lr: float, weight_decay: Union[float, str]):
        super().__init__()
        self.l2 = weight_decay
        self.lr = lr

    def training_step(self, batch, batch_idx, optimizer_idx: Optional[int] = None, dataset_idx: Optional[int] = None):
        return self._step(batch, batch_idx, optimizer_idx)

    def validation_step(self, batch, batch_idx, dataset_idx: Optional[int] = None):
        return self._step(batch, batch_idx, optimizer_idx=None)

    def loss_function_wrapper(self, preds, targets, loss_fn):
        target_type = type_of_target(targets.data.cpu().numpy())
        if target_type in ["binary", "continuous"]:
            if preds.ndim > 1:
                preds = preds.squeeze(dim=-1)
            targets = targets.float()
            return loss_fn(preds, targets)
        else:
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
            }
        }

    def log(self, name: str, *args, **kwargs):
        prefix = "train_" if self.training else "val_"
        super(BaseModel, self).log(prefix + name, *args, **kwargs)
