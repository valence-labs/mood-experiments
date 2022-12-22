from functools import partial
from typing import Optional, Union

import torch

from mood.model.base import BaseModel


class ERM(BaseModel):
    """Empirical Risk Minimization

    The "vanilla" neural network. Updates the weight to minimize the loss of the batch.

    References:
        Vapnik, V. N. (1998). Statistical Learning Theory. Wiley-Interscience.
        https://www.wiley.com/en-fr/Statistical+Learning+Theory-p-9780471030034
    """

    def __init__(
        self,
        base_network: torch.nn.Module,
        loss_fn: torch.nn.Module,
        optimizer: Union["str", torch.optim.Optimizer] = "Adam",
        lr: float = 1e-3,
        lr_scheduler: str = "ReduceLROnPlateau",
        weight_decay="auto",
    ):
        """
        Args:
            base_network: The neural network architecture
            loss_fn: The loss function to use when optimizing
        """
        super(ERM, self).__init__(
            optimizer,
            lr,
            lr_scheduler,
            weight_decay,
        )
        self.base_network = base_network
        self.loss_fn = partial(self.loss_function_wrapper, loss_fn=loss_fn)

    def forward(self, x, domains: Optional = None):
        preds = self.base_network(x)
        return preds

    def _step(self, batch, batch_idx, optimizer_idx: Optional[int] = None):
        (x, domain), y_true = batch
        y_pred = self.forward(x)
        loss = self._loss(y_pred, y_true)
        self.log("loss", loss, prog_bar=True)
        return loss

    def _loss(self, y_pred, y_true):
        return self.loss_fn(y_pred, y_true)
