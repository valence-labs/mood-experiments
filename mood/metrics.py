import torch

import datamol as dm
import numpy as np

from typing import Optional

from sklearn.metrics import average_precision_score
from torch.distributions.utils import clamp_probs
from torchmetrics.functional.classification import (
    binary_auroc,
    binary_average_precision,
    binary_calibration_error,
)
from torchmetrics.functional.regression import mean_absolute_error, spearman_corrcoef
from torchmetrics.functional.regression.spearman import _rank_data, _spearman_corrcoef_update  # noqa
from torchmetrics.wrappers.bootstrapping import _bootstrap_sampler  # noqa


def weighted_spearman(preds, target, sample_weights=None):
    """
    The weighted Spearman correlation efficient. Based on:
     https://en.m.wikipedia.org/wiki/Pearson_correlation_coefficient#Weighted_correlation_coefficient
    """
    if sample_weights is None:
        # If not sample weights are provided, just rely on TorchMetrics' implementation
        return spearman_corrcoef(preds, target)

    def _weighted_mean(x, w):
        return torch.sum(w * x) / torch.sum(w)

    # Copied over the implementation from TorchMetric and made three changes:
    # 1) Instead of computing the mean, compute the weighted mean
    # 2) Computing the weighted covariance
    # 3) Computing the weighted correlation
    preds, target = _spearman_corrcoef_update(preds, target)

    preds = _rank_data(preds)
    target = _rank_data(target)

    preds_diff = preds - _weighted_mean(preds, sample_weights)
    target_diff = target - _weighted_mean(target, sample_weights)

    cov = _weighted_mean(preds_diff * target_diff, sample_weights)

    preds_std = torch.sqrt(_weighted_mean(preds_diff * preds_diff, sample_weights))
    target_std = torch.sqrt(_weighted_mean(target_diff * target_diff, sample_weights))

    corrcoef = cov / (preds_std * target_std + 1e-6)
    return torch.clamp(corrcoef, -1.0, 1.0)


def weighted_mae(preds, target, sample_weights=None):
    if sample_weights is None:
        return mean_absolute_error(preds=preds, target=target)
    summed_mae = torch.abs(sample_weights * (preds - target)).sum()
    return summed_mae / torch.sum(sample_weights)


def weighted_auprc(preds, target, sample_weights=None):
    if sample_weights is None:
        return binary_average_precision(preds=preds, target=target)

    # TorchMetrics does not actually support sample weights, so we rely on the sklearn implementation
    # https://github.com/Lightning-AI/metrics/issues/1098
    preds = preds.cpu().numpy()
    target = target.cpu().numpy()
    sample_weights = sample_weights.cpu().numpy()

    return average_precision_score(y_true=target, y_score=preds, sample_weight=sample_weights)


METRIC_SYNONYMS = {
    ("auroc",): binary_auroc,
    ("auprc", "average_precision"): weighted_auprc,
    ("mean_absolute_error", "mae"): weighted_mae,
    ("spearman", "spearman_corrcoef"): weighted_spearman, 
}

METRIC_TYPES = {
    ("auroc",): "classification",
    ("auprc", "average_precision"): "classification",
    ("mean_absolute_error", "mae"): "regression",
    ("spearman", "spearman_corrcoef"): "regression",
    ("ece, expected_calibration_error"): "classification",
}

METRIC_DIRECTIONS = {
    ("auroc",): "maximize",
    ("auprc", "average_precision"): "maximize",
    ("mean_absolute_error", "mae"): "minimize",
    ("spearman", "spearman_corrcoef"): "maximize",
    ("ece, expected_calibration_error"): "minimize",
}


def get_metric_direction(metric: str):
    metric_clean = metric.lower().strip().replace(" ", "_")
    for group, direction in METRIC_DIRECTIONS.items():
        if metric_clean in group:
            return direction
    raise NotImplementedError(f"{metric} is currently not a supported metric")


def get_metric_type(metric: str):
    metric_clean = metric.lower().strip().replace(" ", "_")
    for group, metric_type in METRIC_TYPES.items():
        if metric_clean in group:
            return metric_type
    raise NotImplementedError(f"{metric} is currently not a supported metric")


def get_metric(metric: str):
    metric_clean = metric.lower().strip().replace(" ", "_")
    for group, function in METRIC_SYNONYMS.items():
        if metric_clean in group:
            return function
    raise NotImplementedError(f"{metric} is currently not a supported metric")


def get_calibration_metric(is_regression: bool):
    return "Spearman" if is_regression else "ECE"
    
    
def is_better(metric: str, before: float, after: float):
    if get_metric_direction(metric) == "maximize":
        return before > after
    else:
        return after > before


def compute_bootstrapped_metric(
    predictions,
    targets,
    metric,
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
        return compute_metric(predictions[indices], targets[indices], metric, _sample_weights)

    bootstrapped_scores = dm.utils.parallelized(fn, range(n_bootstraps), n_jobs=n_jobs)
    bootstrapped_scores = [score for score in bootstrapped_scores if score is not None]
    return np.mean(bootstrapped_scores), np.std(bootstrapped_scores)


def compute_metric(predictions, targets, metric, sample_weights: Optional = None):
    if not isinstance(predictions, torch.Tensor):
        predictions = torch.tensor(predictions)
    if not isinstance(targets, torch.Tensor):
        targets = torch.tensor(targets, device=predictions.device)
    if sample_weights is not None and not isinstance(sample_weights, torch.Tensor):
        sample_weights = torch.tensor(sample_weights, device=predictions.device)

    f = get_metric(metric)

    predictions = predictions.float().squeeze()
    if predictions.ndim == 0:
        predictions = predictions.unsqueeze(0)

    if get_metric_type(metric) == "classification":
        targets = targets.int()
    else:
        targets = targets.float()

    # These metrics do not make sense when all targets are either 0 or 1
    if metric.lower() in ["auprc", "auroc"] and (all(targets == 0) or all(targets == 1)):
        return None

    if sample_weights is not None:
        return f(preds=predictions, target=targets, sample_weights=sample_weights).item()
    else:
        return f(preds=predictions, target=targets).item()


def compute_uncertainty_calibration(uncertainties, predictions, targets, is_regression, n_bins: int = 20):
    """
    Computes a metric that indicates how well the uncertainty is calibrated.
    A good calibration means that the uncertainty is likely high if the error is.
    And the other way around.
    
    For regression tasks, we compute the Spearman correlation between absolute error
    and uncertainty. For classification tasks, we compute the binned difference between
    the confidence and accuracy within that bin.
    """
    if not isinstance(targets, torch.Tensor):
        targets = torch.tensor(targets)
    if not isinstance(predictions, torch.Tensor):
        predictions = torch.tensor(predictions)
    if not isinstance(uncertainties, torch.Tensor):
        uncertainties = torch.tensor(uncertainties)

    uncertainties = uncertainties.float().squeeze()
    if uncertainties.ndim == 0:
        uncertainties = uncertainties.unsqueeze(0)
    predictions = predictions.float().squeeze()
    if predictions.ndim == 0:
        predictions = predictions.unsqueeze(0)

    if is_regression:
        targets = targets.float()
        errors = torch.abs(predictions - targets)
        spearman = weighted_spearman(errors, uncertainties, sample_weights=None).item()
        return (spearman + 1.0) / 2.0  # From [-1, 1] to [0, 1]
    
    else:
        if not ((0 <= uncertainties) * (uncertainties <= 1)).all():
            raise ValueError("All uncertainties should be between 0 and 1")

        confidences = clamp_probs(1.0 - uncertainties).squeeze()
        if confidences.ndim == 0:
            confidences = confidences.unsqueeze(0)
        error = binary_calibration_error(confidences, targets, norm="l1", n_bins=n_bins).item()
        return 1.0 - error
