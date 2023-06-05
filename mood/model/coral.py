from functools import partial
from typing import Optional, List

from torch import nn
from mood.model.base import BaseModel
from mood.model.utils import linear_interpolation


class CORAL(BaseModel):
    """CORAL is a **domain adaptation** method.

    In addition to the traditional loss, adds a penalty based on the difference of the first and second moment
    of the source and target features.

    References:
        Sun, B., & Saenko, K. (2016, October). Deep coral: Correlation alignment for deep domain adaptation.
        In European conference on computer vision (pp. 443-450). Springer, Cham.
        https://arxiv.org/abs/1607.01719
    """

    def __init__(
        self,
        base_network: nn.Module,
        prediction_head: nn.Module,
        loss_fn: nn.Module,
        batch_size: int,
        penalty_weight: float,
        penalty_weight_schedule: List[int],
        lr: float = 1e-4,
        weight_decay: float = 0,
    ):
        """
        Args:
            base_network: The neural network architecture endoing the features
            prediction_head: The neural network architecture that takes the concatenated
                representation of the domain and features and returns a task-specific prediction.
            loss_fn: The loss function to optimize for.
            penalty_weight: The weight to multiply the penalty by.
            penalty_weight_schedule: List of two integers as a very rudimental way of scheduling the
                penalty weight. The first integer is the last epoch at which the penalty weight is 0.
                The second integer is the first epoch at which the penalty weight is its max value.
                Linearly interpolates between the two.
        """

        super().__init__(
            lr=lr,
            weight_decay=weight_decay,
            base_network=base_network,
            prediction_head=prediction_head,
            loss_fn=loss_fn,
            batch_size=batch_size,
        )

        self.penalty_weight = penalty_weight
        if len(penalty_weight_schedule) != 2:
            raise ValueError("The penalty weight schedule needs to define two values; The start and end step")
        self.start = penalty_weight_schedule[0]
        self.duration = penalty_weight_schedule[1] - penalty_weight_schedule[0]

    def _step(self, batch, batch_idx=0, optimizer_idx=None):
        if self.training:
            batch_src, batch_tgt = batch["source"], batch["target"]
            (x_src, domains_src), y_true = batch_src
            x_tgt, domains_tgt = batch_tgt
            y_pred, phis_src = self.forward(x_src, return_embedding=True)
            _, phis_tgt = self.forward(x_tgt, return_embedding=True)
            loss = self._loss(y_pred, y_true, phis_src, phis_tgt)

        else:
            (x, domains), y_true = batch
            y_pred = self.forward(x)
            loss = self._loss(y_pred, y_true)

        self.log("loss", loss)
        return loss

    def _loss(self, y_pred, y_true, phis_src: Optional = None, phis_tgt: Optional = None):
        erm_loss = self.loss_fn(y_pred, y_true)

        if not self.training:
            return erm_loss

        penalty = self._coral_penalty(phis_src, phis_tgt)

        penalty_weight = linear_interpolation(
            self.current_epoch, self.duration, self.penalty_weight, start=self.start
        )

        loss = erm_loss + penalty_weight * penalty
        return loss

    @staticmethod
    def _coral_penalty(x, y):
        """The CORAL penalty aligns the Covariance matrix of the features across domains"""
        mean_x = x.mean(0, keepdim=True)
        mean_y = y.mean(0, keepdim=True)
        cent_x = x - mean_x
        cent_y = y - mean_y
        cova_x = (cent_x.T @ cent_x) / max(1, len(x) - 1)
        cova_y = (cent_y.T @ cent_y) / max(1, len(y) - 1)

        mean_diff = (mean_x - mean_y).pow(2).mean()
        cova_diff = (cova_x - cova_y).pow(2).mean()
        return mean_diff + cova_diff

    @staticmethod
    def suggest_params(trial):
        params = BaseModel.suggest_params(trial)
        params["penalty_weight"] = trial.suggest_float("penalty_weight", 0.0001, 100, log=True)
        params["penalty_weight_schedule"] = trial.suggest_categorical(
            "penalty_weight_schedule", [[0, 25], [0, 50], [0, 0], [25, 50]]
        )
        return params
