import enum
import torch
import datamol as dm
import numpy as np
from typing import Callable, Optional

from sklearn.metrics import roc_auc_score
from torchmetrics.functional import mean_absolute_error, mean_squared_error
from torchmetrics.functional.classification import binary_auroc
from torchmetrics.functional.regression.pearson import pearson_corrcoef
from torchmetrics.wrappers.bootstrapping import _bootstrap_sampler

from mood.dataset import MOOD_REGR_DATASETS


def weighted_pearson(preds, target, sample_weights=None):
    """
    The weighted Pearson correlation efficient. Based on:
     https://stats.stackexchange.com/a/222107
    """
    if sample_weights is None:
        # If not sample weights are provided, just rely on TorchMetrics' implementation
        return pearson_corrcoef(preds=preds, target=target)

    def _weighted_mean(x, w):
        return torch.sum(w * x) / torch.sum(w)

    # Copied over the implementation from TorchMetric and made three changes:
    # 1) Instead of computing the mean, compute the weighted mean
    # 2) Computing the weighted covariance
    # 3) Computing the weighted correlation

    preds_diff = preds - _weighted_mean(preds, sample_weights)
    target_diff = target - _weighted_mean(target, sample_weights)

    cov = _weighted_mean(preds_diff * target_diff, sample_weights)

    preds_std = torch.sqrt(_weighted_mean(preds_diff * preds_diff, sample_weights))
    target_std = torch.sqrt(_weighted_mean(target_diff * target_diff, sample_weights))

    corrcoef = cov / (preds_std * target_std + 1e-6)
    return torch.clamp(corrcoef, -1.0, 1.0)


def weighted_pearson_calibration(preds, target, uncertainty, sample_weights=None):
    error = torch.abs(preds - target)
    return weighted_pearson(error, uncertainty, sample_weights)


def weighted_mae(preds, target, sample_weights=None):
    if sample_weights is None:
        return mean_absolute_error(preds=preds, target=target)
    summed_mae = torch.abs(sample_weights * (preds - target)).sum()
    return summed_mae / torch.sum(sample_weights)


def weighted_brier_score(target, uncertainty, sample_weights=None):
    confidence = torch.stack([unc[tar] for unc, tar in zip(uncertainty, target)])
    if sample_weights is None:
        return mean_squared_error(preds=confidence, target=target)
    summed_mse = torch.square(sample_weights * (confidence - target)).sum()
    brier_score = summed_mse / torch.sum(sample_weights)
    return brier_score


def weighted_auroc(preds, target, sample_weights=None):
    if sample_weights is None:
        return binary_auroc(preds=preds, target=target)

    # TorchMetrics does not actually support sample weights, so we rely on the sklearn implementation
    preds = preds.cpu().numpy()
    target = target.cpu().numpy()
    sample_weights = sample_weights.cpu().numpy()
    return roc_auc_score(y_true=target, y_score=preds, sample_weight=sample_weights)


class TargetType(enum.Enum):
    REGRESSION = "regression"
    BINARY_CLASSIFICATION = "binary_classification"

    def is_regression(self):
        return self == TargetType.REGRESSION


class Metric:
    def __init__(
        self,
        name: str,
        fn: Callable,
        mode: str,
        target_type: TargetType,
        needs_predictions: bool = True,
        needs_uncertainty: bool = False,
        range_min: Optional[float] = None,
        range_max: Optional[float] = None,
    ):
        self.fn_ = fn
        self.name = name
        self.mode = mode
        self.target_type = target_type
        self.needs_predictions = needs_predictions
        self.needs_uncertainty = needs_uncertainty
        self.range_min = range_min
        self.range_max = range_max

    @property
    def is_calibration(self):
        return self.needs_uncertainty

    @classmethod
    def get_default_calibration_metric(cls, dataset):
        is_regression = dataset in MOOD_REGR_DATASETS
        if is_regression:
            metric = cls.by_name("Pearson")
        else:
            metric = cls.by_name("Brier score")
        return metric

    @classmethod
    def get_default_performance_metric(cls, dataset):
        is_regression = dataset in MOOD_REGR_DATASETS
        if is_regression:
            metric = cls.by_name("MAE")
        else:
            metric = cls.by_name("AUROC")
        return metric

    @classmethod
    def by_name(cls, name):
        if name == "MAE":
            return cls(
                name="MAE",
                fn=weighted_mae,
                mode="min",
                target_type=TargetType.REGRESSION,
                needs_predictions=True,
                needs_uncertainty=False,
                range_min=0,
                range_max=None,
            )
        elif name == "Pearson":
            return cls(
                name="Pearson",
                fn=weighted_pearson_calibration,
                mode="max",
                target_type=TargetType.REGRESSION,
                needs_predictions=True,
                needs_uncertainty=True,
                range_min=-1,
                range_max=1,
            )
        elif name == "AUROC":
            return cls(
                name="AUROC",
                fn=weighted_auroc,
                mode="max",
                target_type=TargetType.BINARY_CLASSIFICATION,
                needs_predictions=True,
                needs_uncertainty=False,
                range_min=0,
                range_max=1,
            )
        elif name == "Brier score":
            return cls(
                name="Brier score",
                fn=weighted_brier_score,
                mode="min",
                target_type=TargetType.BINARY_CLASSIFICATION,
                needs_predictions=False,
                needs_uncertainty=True,
                range_min=0,
                range_max=1,
            )

    def __call__(
        self, y_true, y_pred: Optional = None, uncertainty: Optional = None, sample_weights: Optional = None
    ):
        if self.needs_uncertainty and uncertainty is None:
            raise ValueError("Uncertainty estimates needed, but not provided.")
        if self.needs_predictions and y_pred is None:
            raise ValueError("Predictions needed, but not provided.")
        kwargs = self.to_kwargs(y_true, y_pred, uncertainty, sample_weights)
        return self.fn_(**kwargs).item()

    @staticmethod
    def preprocess_targets(y_true, is_regression: bool):
        if not isinstance(y_true, torch.Tensor):
            y_true = torch.tensor(y_true)
        if is_regression:
            y_true = y_true.float().squeeze()
            if y_true.ndim == 0:
                y_true = y_true.unsqueeze(0)
        else:
            y_true = y_true.int()

        return y_true

    @staticmethod
    def preprocess_predictions(y_pred, device):
        if not isinstance(y_pred, torch.Tensor):
            y_pred = torch.tensor(y_pred, device=device)
        y_pred = y_pred.float().squeeze()
        if y_pred.ndim == 0:
            y_pred = y_pred.unsqueeze(0)
        return y_pred

    @staticmethod
    def preprocess_uncertainties(uncertainty, device):
        return Metric.preprocess_predictions(uncertainty, device)

    @staticmethod
    def preprocess_sample_weights(sample_weights, device):
        if sample_weights is not None and not isinstance(sample_weights, torch.Tensor):
            sample_weights = torch.tensor(sample_weights, device=device)
        return sample_weights

    @property
    def range(self):
        return self.range_min, self.range_max

    def to_kwargs(self, y_true, y_pred, uncertainty, sample_weights):
        kwargs = {"target": self.preprocess_targets(y_true, self.target_type.is_regression())}
        kwargs["sample_weights"] = self.preprocess_sample_weights(sample_weights, kwargs["target"].device)
        if self.needs_predictions:
            kwargs["preds"] = self.preprocess_predictions(y_pred, kwargs["target"].device)
        if self.needs_uncertainty:
            kwargs["uncertainty"] = self.preprocess_uncertainties(uncertainty, kwargs["target"].device)
        return kwargs


def compute_bootstrapped_metric(
    targets,
    metric: Metric,
    predictions: Optional = None,
    uncertainties: Optional = None,
    sample_weights: Optional = None,
    sampling_strategy: str = "poisson",
    n_bootstraps: int = 1000,
    n_jobs: Optional[int] = None,
):
    """
    Bootstrapping to compute confidence intervals for a metric.
    Inspired by https://stackoverflow.com/a/19132400 and
    https://github.com/PyTorchLightning/metrics/blob/master/torchmetrics/wrappers/bootstrapping.py
    """

    def fn(it):
        indices = _bootstrap_sampler(len(predictions), sampling_strategy=sampling_strategy)
        _sample_weights = None if sample_weights is None else sample_weights[indices]
        _uncertainties = None if uncertainties is None else uncertainties[indices]
        _predictions = None if predictions is None else predictions[indices]
        score = metric(targets[indices], _predictions, _uncertainties, _sample_weights)
        return score

    bootstrapped_scores = dm.utils.parallelized(fn, range(n_bootstraps), n_jobs=n_jobs)
    bootstrapped_scores = [score for score in bootstrapped_scores if score is not None]
    return np.mean(bootstrapped_scores), np.std(bootstrapped_scores)
