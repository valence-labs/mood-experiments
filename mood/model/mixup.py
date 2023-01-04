from typing import List

import torch
from torch import nn

from mood.model.base import BaseModel
from mood.model.utils import linear_interpolation


class Mixup(BaseModel):
    """Mixup is a **domain adaptation** method.

    Mixup interpolates both the features and (pseudo-)targets inter- and intra-domain and trains on
    these interpolates samples instead.

    References:
        Yan, S., Song, H., Li, N., Zou, L., & Ren, L. (2020). Improve unsupervised domain adaptation with mixup training.
        https://arxiv.org/abs/2001.00677
    """

    def __init__(
        self,
        base_network: nn.Module,
        prediction_head: nn.Module,
        loss_fn: nn.Module,
        batch_size: int,
        penalty_weight: float,
        penalty_weight_schedule: List[int],
        lr: float = 1e-3,
        weight_decay: float = "auto",
        augmentation_std: float = 0.1,
        no_augmentations: int = 10,
        alpha: float = 0.1,
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
            augmentation_std: The standard deviation of the noise to multiply each sample by
                as augmentation.
            no_augmentations: The number of augmentations to do to compute pseudo labels for the unlabeled samples
                of the target domain.
            alpha: The parameter of the Beta distribution used to compute the interpolation factor
        """
        super().__init__(
            lr=lr,
            weight_decay=weight_decay,
            base_network=base_network,
            prediction_head=prediction_head,
            loss_fn=loss_fn,
            batch_size=batch_size,
        )

        self._classification_loss = torch.nn.BCELoss()
        self._target_domain_loss = torch.nn.MSELoss()

        self._augmentation_std = augmentation_std
        self._no_augmentations = no_augmentations

        self.distribution_lambda = torch.distributions.Beta(alpha, alpha)

        self.penalty_weight = penalty_weight
        if len(penalty_weight_schedule) != 2:
            raise ValueError("The penalty weight schedule needs to define two values; The start and end step")
        self.start = penalty_weight_schedule[0]
        self.duration = penalty_weight_schedule[1] - penalty_weight_schedule[0]

    def _step(self, batch, batch_idx=0, optimizer_idx=None):

        if not self.training:
            (x, domains), y_true = batch
            y_pred = self.forward(x)
            loss = self.loss_fn(y_pred, y_true)
            self.log("loss", loss)
            return loss

        batch_src, batch_tgt = batch["source"], batch["target"]
        (x_src, _), y_src = batch_src
        x_tgt, _ = batch_tgt
        y_tgt = self._get_pseudo_labels(x_tgt)
        penalty_weight = linear_interpolation(
            step=self.current_epoch,
            duration=self.duration,
            max_value=self.penalty_weight,
            start=self.start,
        )

        loss = 0

        # Inter-domain, from source to target
        loss += self._loss(x_src, x_tgt, y_src, y_tgt, True)
        # Intra-domain, from source to source
        loss += self._loss(x_src, x_src, y_src, y_src, False)
        # Intra-domain, from target to target
        # Quote from the paper: "We set a linear schedule for w_t in training,
        # from 0 to a predefined maximum value. From initial experiments, we observe that
        # the algorithm is robust to other weighting parameters. Therefore we only
        # search w_t while simply fixing all other weightings to 1."
        loss += penalty_weight * self._loss(x_tgt, x_tgt, y_tgt, y_tgt, False)

        self.log("loss", loss)
        return loss

    def _get_pseudo_labels(self, xs):
        augmented_labels = []
        for i in range(self._no_augmentations):
            sample = xs * torch.normal(1, self._augmentation_std, xs.size(), device=xs.device)
            augmented_labels.append(self.forward(sample))
        pseudo_labels = torch.stack(augmented_labels).mean(0).squeeze()
        return pseudo_labels

    def _loss(self, x_src, x_tgt, y_src, y_tgt, inter_domain: bool):

        lam = self.distribution_lambda.sample()
        lam_prime = torch.max(lam, 1.0 - lam)

        x_src, x_tgt, y_src, y_tgt = self._get_random_pairs(x_src, x_tgt, y_src, y_tgt, round(len(x_src) / 3))
        x_st, y_st = self._mixup(x_src, x_tgt, y_src, y_tgt, lam_prime)
        y_pred_st, phi_st = self.forward(x_st, return_embedding=True)

        if inter_domain:

            # Predictive loss
            loss_q = self.loss_fn(y_pred_st, y_st)

            # Consistency regularizer
            y_pred_s, phi_s = self.forward(x_src, return_embedding=True)
            y_pred_t, phi_t = self.forward(x_tgt, return_embedding=True)
            zi_st = lam_prime * phi_s + (1.0 - lam_prime) * phi_t
            loss_z = torch.norm(zi_st - phi_st, dim=0).mean()
            loss = loss_q + loss_z

        # Intra target domain
        else:
            loss = self.loss_fn(y_pred_st, y_st)

        return loss

    def _get_random_pairs(self, x_src, x_tgt, y_src, y_tgt, size):
        assert len(x_src) == len(y_src)
        assert len(x_tgt) == len(y_tgt)

        size = max(torch.tensor(1), size)
        indices = torch.multinomial(torch.ones(len(x_src), device=x_src.device), size, replacement=True)
        x_src = x_src[indices]
        y_src = y_src[indices]

        indices = torch.multinomial(torch.ones(len(y_tgt), device=y_tgt.device), size, replacement=True)
        x_tgt = x_tgt[indices]
        y_tgt = y_tgt[indices]

        return x_src, x_tgt, y_src, y_tgt

    def _mixup(self, x_s, x_t, y_s, y_t, lam_prime):
        xi_st = lam_prime * x_s + (1.0 - lam_prime) * x_t
        yi_st = lam_prime * y_s + (1.0 - lam_prime) * y_t
        return xi_st, yi_st

    @staticmethod
    def suggest_params(trial):
        params = BaseModel.suggest_params(trial)
        params["penalty_weight"] = trial.suggest_float("penalty_weight", 0.0001, 100, log=True)
        params["penalty_weight_schedule"] = trial.suggest_categorical(
            "penalty_weight_schedule", [[0, 25], [0, 50], [0, 0], [25, 50]]
        )
        params["augmentation_std"] = trial.suggest_categorical("augmentation_std", 0.001, 0.15)
        params["no_augmentations"] = trial.suggest_categorical("no_augmentations", [3, 5, 10])
        params["alpha"] = trial.suggest_float("alpha", 0.0, 1.0)
        return params
