from functools import partial
from typing import Optional, List

import torch
from torch import nn

from mood.model.base import BaseModel
from mood.model.utils import linear_interpolation


class VREx(BaseModel):
    """Variance Risk Extrapolation (VREx) is a **domain generalization** method.

    Similar to MTL, computes the loss per domain and computes the mean of these domain-specific losses.
    Additionally, following equation 8 of the paper, returns an additional penalty based on the variance
    of the domain specific losses.

    References:
        Krueger, D., Caballero, E., Jacobsen, J. H., Zhang, A., Binas, J., Zhang, D., ... & Courville, A. (2021, July).
        Out-of-distribution generalization via risk extrapolation (rex).
        In International Conference on Machine Learning (pp. 5815-5826). PMLR.
        https://arxiv.org/abs/2003.00688
    """

    def __init__(
        self,
        base_network: nn.Module,
        prediction_head: nn.Module,
        loss_fn: nn.Module,
        batch_size: int,
        penalty_weight: float,
        penalty_weight_schedule: List[int],
        lr=1e-3,
        weight_decay=0,
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

        erm_losses = []
        for mini_batch in batch:
            (xs, _), y_true = mini_batch
            y_pred = self.forward(xs, return_embedding=False)
            erm_losses.append(self.loss_fn(y_pred, y_true))

        penalty_weight = linear_interpolation(
            step=self.current_epoch,
            duration=self.duration,
            max_value=self.penalty_weight,
            start=self.start,
        )

        erm_losses = torch.stack(erm_losses)
        erm_loss = erm_losses.mean()
        rex_penalty = erm_losses.var()
        loss = erm_loss + penalty_weight * rex_penalty

        self.log("loss", loss)
        return loss

    @staticmethod
    def suggest_params(trial):
        params = BaseModel.suggest_params(trial)
        params["penalty_weight"] = trial.suggest_float("penalty_weight", 0.0001, 100, log=True)
        params["penalty_weight_schedule"] = trial.suggest_categorical("penalty_weight_schedule", [[0, 25], [0, 50], [0, 0], [25, 50]])
        return params