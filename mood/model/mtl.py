from copy import deepcopy
from typing import Optional

import torch
from torch import nn
from mood.model.base import BaseModel


class MTL(BaseModel):
    """Marginal Transfer Learning (MTL) is a **domain generalization** method.

    MTL uses a representation of the domain.
    Additionally, rather than computing the loss for the entire batch at once, it computes the loss
    for each domain individually and then returns the mean of this.

    Originally proposed as a kernel method, it is here implemented as a Neural Network.

    References:
        Blanchard, G., Deshmukh, A. A., Dogan, U., Lee, G., & Scott, C. (2017).
          Domain generalization by marginal transfer learning.
          https://arxiv.org/abs/1711.07910
    """

    def __init__(
        self,
        base_network: nn.Module,
        prediction_head: nn.Module,
        loss_fn: nn.Module,
        batch_size: int,
        lr: float = 1e-3,
        weight_decay: float = 0,
    ):
        """
        Args:
            base_network: The neural network architecture endoing the features
            domain_network: The neural network architecture encoding the domain
            prediction_head: The neural network architecture that takes the concatenated
                representation of the domain and features and returns a task-specific prediction.
            loss_fn: The loss function to optimize for.
        """
        super().__init__(
            lr=lr,
            weight_decay=weight_decay,
            base_network=base_network,
            loss_fn=loss_fn,
            prediction_head=prediction_head,
            batch_size=batch_size,
        )
        self.domain_network = deepcopy(base_network)

    def forward(self, x, domains: Optional = None, return_embedding: bool = False):
        input_embeddings = self.base_network(x)
        domain_embeddings = self.domain_network(domains)
        # add noise to domains to avoid overfitting
        domain_embeddings = domain_embeddings + torch.randn_like(domain_embeddings)
        embeddings = torch.cat((domain_embeddings, input_embeddings), 1)
        label = self.prediction_head(embeddings)
        out = (label, embeddings) if return_embedding else label
        return out

    def _step(self, batch, batch_idx=0, optimizer_idx=None):
        xs = torch.cat([xs for (xs, _), _ in batch], dim=0)
        domains = torch.cat([ds for (_, ds), _ in batch], dim=0)
        y_true = torch.cat([ys for _, ys in batch], dim=0)
        ns = [len(ys) for _, ys in batch]

        y_pred, phis = self.forward(xs, domains, return_embedding=True)

        loss = self._loss(y_pred, y_true, ns)
        self.log("loss", loss)
        return loss

    def _loss(self, y_pred, y_true, ns):
        loss, i = 0, 0
        for n in ns:
            loss += self.loss_fn(y_pred[i : i + n], y_true[i : i + n])
            i += n
        loss = loss / len(ns)
        return loss
