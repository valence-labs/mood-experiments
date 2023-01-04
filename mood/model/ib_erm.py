from typing import List

import torch
from torch import nn

from mood.model.base import BaseModel
from mood.model.utils import linear_interpolation


class InformationBottleneckERM(BaseModel):
    """Information Bottleneck ERM (IB-ERM) is a **domain generalization** method.

    Similar to MTL, computes the loss per domain and computes the mean of these domain-specific losses.
    Additionally, adds a penalty based on the variance of the feature dimensions.

    References:
       Ahuja, K., Caballero, E., Zhang, D., Gagnon-Audet, J. C., Bengio, Y., Mitliagkas, I., & Rish, I. (2021).
       Invariance principle meets information bottleneck for out-of-distribution generalization.
       Advances in Neural Information Processing Systems, 34, 3438-3450.
       https://arxiv.org/abs/2106.06607
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

        phis = []
        erm_loss = 0

        for mini_batch in batch:
            (xs, _), y_true = mini_batch
            y_pred, phi = self.forward(xs, return_embedding=True)
            erm_loss += self.loss_fn(y_pred, y_true)
            phis.append(phi)

        erm_loss /= len(batch)
        phis = torch.cat(phis, dim=0)
        loss = self._loss(erm_loss, phis)
        self.log("loss", loss)
        return loss

    def _loss(self, erm_loss, phis):

        if not self.training:
            return erm_loss

        penalty_weight = linear_interpolation(
            step=self.current_epoch,
            duration=self.duration,
            max_value=self.penalty_weight,
            start=self.start,
        )

        penalty = 0
        for i in range(len(phis)):
            # Add the Information Bottleneck penalty
            penalty += self.ib_penalty(phis[i])
        penalty /= len(phis)

        loss = erm_loss + penalty_weight * penalty
        return loss

    @staticmethod
    def ib_penalty(features):
        if len(features) == 1:
            return 0.0
        return features.var(dim=0).mean()

    @staticmethod
    def suggest_params(trial):
        params = BaseModel.suggest_params(trial)
        params["penalty_weight"] = trial.suggest_float("penalty_weight", 0.0001, 100, log=True)
        params["penalty_weight_schedule"] = trial.suggest_categorical(
            "penalty_weight_schedule", [[0, 25], [0, 50], [0, 0], [25, 50]]
        )
        return params
