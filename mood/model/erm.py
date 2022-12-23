from typing import Optional
from mood.model.base import BaseModel


class ERM(BaseModel):
    """Empirical Risk Minimization

    The "vanilla" neural network. Updates the weight to minimize the loss of the batch.

    References:
        Vapnik, V. N. (1998). Statistical Learning Theory. Wiley-Interscience.
        https://www.wiley.com/en-fr/Statistical+Learning+Theory-p-9780471030034
    """

    def _step(self, batch, batch_idx, optimizer_idx: Optional[int] = None):
        (x, domain), y_true = batch
        y_pred = self.forward(x)
        loss = self._loss(y_pred, y_true)
        self.log("loss", loss, prog_bar=True)
        return loss

    def _loss(self, y_pred, y_true):
        return self.loss_fn(y_pred, y_true)
