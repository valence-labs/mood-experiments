from itertools import chain, tee

import torch
import torch.nn.functional as F
from typing import Optional, List
from torch import nn

from mood.model.base import BaseModel
from mood.model.nn import FCLayer
from mood.model.utils import linear_interpolation


class DANN(BaseModel):
    """Domain Adversarial Neural Network (DANN) is a **domain adaptation** method.

    Adversarial framework that includes a prediction and discriminator network. The goal of the discriminator
    is to classify the domain (source or target) from the hidden embedding. The goal of the predictor is to achieve
    a good task-specific performance. By optimizing these in an adversarial fashion, the goal is to have
    domain-invariant features.

    References:
        Ganin, Y., Ustinova, E., Ajakan, H., Germain, P., Larochelle, H., Laviolette, F., ... & Lempitsky, V. (2016).
         Domain-adversarial training of neural networks. The journal of machine learning research, 17(1), 2096-2030.
         https://arxiv.org/abs/1505.07818
    """

    def __init__(
        self,
        base_network: nn.Module,
        prediction_head: FCLayer,
        loss_fn: nn.Module,
        batch_size: int,
        penalty_weight: float,
        penalty_weight_schedule: List[int],
        discriminator_network: Optional[nn.Module] = None,
        lr: float = 1e-3,
        discr_lr: float = 1e-3,
        weight_decay: float = "auto",
        discr_weight_decay: float = "auto",
        lambda_reg: float = 0.1,
        n_discr_steps_per_predictor_step=3,
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
            discriminator_network: The discriminator network that predicts the domain from the hidden embedding.
            lambda_reg: An additional weighing factor for the penalty. Following the implementation of DomainBed.
        """
        super().__init__(
            lr=lr,
            weight_decay=weight_decay,
            base_network=base_network,
            prediction_head=prediction_head,
            loss_fn=loss_fn,
            batch_size=batch_size,
        )

        self._discriminator_network = discriminator_network
        if self._discriminator_network is None:
            self._discriminator_network = FCLayer(prediction_head.input_dim, 2, activation=None)

        self.penalty_weight = penalty_weight
        if len(penalty_weight_schedule) != 2:
            raise ValueError("The penalty weight schedule needs to define two values; The start and end step")
        self.start = penalty_weight_schedule[0]
        self.duration = penalty_weight_schedule[1] - penalty_weight_schedule[0]

        self._discriminator_loss = nn.CrossEntropyLoss()
        self.discr_lr = discr_lr
        self.discr_l2 = discr_weight_decay
        self.lambda_reg = lambda_reg
        self._n_discr_steps_per_predictor_step = n_discr_steps_per_predictor_step

    @staticmethod
    def get_optimizer(parameters, lr, weight_decay, monitor: str = "val_loss"):

        if weight_decay == "auto":
            parameters, parameters_copy = tee(parameters)
            weight_decay = 1.0 / sum(p.numel() for p in parameters_copy if p.requires_grad)

        optimizer = torch.optim.Adam(parameters, lr=lr, weight_decay=weight_decay)
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": lr_scheduler,
                "interval": "epoch",
                "frequency": 1,
                "monitor": monitor,
                "strict": True,
            },
        }

    def configure_optimizers(self):

        optimizer_predictor = self.get_optimizer(
            chain(self.base_network.parameters(), self.prediction_head.parameters()),
            self.lr,
            self.l2,
        )
        optimizer_predictor["frequency"] = self._n_discr_steps_per_predictor_step

        optimizer_discriminator = self.get_optimizer(
            chain(self.base_network.parameters(), self._discriminator_network.parameters()),
            self.discr_lr,
            self.discr_l2,
        )
        optimizer_discriminator["frequency"] = 1

        return optimizer_predictor, optimizer_discriminator

    def forward(
        self, x, domains: Optional = None, return_embedding: bool = False, return_discriminator: bool = False
    ):
        input_embeddings = self.base_network(x)
        label = self.prediction_head(input_embeddings)

        ret = (label,)
        if return_embedding:
            ret += (input_embeddings,)

        if return_discriminator:
            discr_out = self._discriminator_network(input_embeddings)
            ret += (discr_out,)

        if len(ret) == 1:
            return ret[0]
        return ret

    def _step(self, batch, batch_idx=0, optimizer_idx=None):

        if not self.training:
            (x, domains), y_true = batch
            y_pred = self.forward(x)
            loss = self.loss_fn(y_pred, y_true)
            self.log("loss", loss)
            return loss

        batch_src, batch_tgt = batch["source"], batch["target"]

        (x_src, domains_src), y_true = batch_src
        x_tgt, domains_tgt = batch_tgt

        y_pred, input_embeddings_src, discr_out_src = self.forward(
            x_src, return_embedding=True, return_discriminator=True
        )
        _, input_embeddings_tgt, discr_out_tgt = self.forward(
            x_tgt, return_embedding=True, return_discriminator=True
        )

        erm_loss = self.loss_fn(y_pred, y_true)

        # For losses w.r.t. the discriminator
        domain_pred = torch.cat([discr_out_src, discr_out_tgt], dim=0)
        domain_true = torch.cat([torch.zeros_like(discr_out_src), torch.ones_like(discr_out_tgt)], dim=0)
        loss = self._discriminator_loss(domain_pred, domain_true)

        domain_pred_softmax = F.softmax(domain_pred, dim=1)
        penalty_loss = domain_pred_softmax[:, domain_true.long()].sum()

        # When optimizing the discriminator, PTL automatically set requires_grad to False
        # for all parameters not optimized by the parameter. However, we still need True to
        # be able to compute the gradient penalty
        if optimizer_idx == 1:
            for param in self.base_network.parameters():
                param.requires_grad = True

        penalty = self.gradient_reversal(penalty_loss, [input_embeddings_src, input_embeddings_tgt])

        if optimizer_idx == 1:
            for param in self.base_network.parameters():
                param.requires_grad = False

        penalty_weight = linear_interpolation(
            self.current_epoch, self.duration, self.penalty_weight, start=self.start
        )
        loss += penalty * penalty_weight

        if optimizer_idx == 1:
            # Train the discriminator
            self.log("discriminator_loss", loss)
            return loss
        else:
            # Train the predictor and add a penalty for making features domain-invariant
            loss = erm_loss + (self.lambda_reg * -loss)
            self.log("predictive_loss", loss)
            return loss

    @staticmethod
    def gradient_reversal(loss, inputs):
        grad = torch.cat(torch.autograd.grad(loss, inputs, create_graph=True))
        return (grad**2).sum(dim=1).mean(dim=0)

    @staticmethod
    def suggest_params(trial):
        params = BaseModel.suggest_params(trial)
        params["penalty_weight"] = trial.suggest_float("penalty_weight", 1e-10, 1.0, log=True)
        params["penalty_weight_schedule"] = trial.suggest_categorical(
            "penalty_weight_schedule", [[0, 25], [0, 50], [0, 0], [25, 50]]
        )
        params["discr_lr"] = trial.suggest_float("discr_lr", 1e-8, 1.0, log=True)
        params["discr_weight_decay"] = trial.suggest_categorical(
            "discr_weight_decay", ["auto", 0.0, 0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1, 1.0]
        )
        params["lambda_reg"] = trial.suggest_float("lambda_reg", 0.001, 10, log=True)
        params["n_discr_steps_per_predictor_step"] = trial.suggest_int(
            "n_discr_steps_per_predictor_step", 1, 5
        )
        return params
