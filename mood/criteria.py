import abc
from typing import Sequence

import numpy as np

from mood.dataset import SimpleMolecularDataset
from mood.distance import compute_knn_distance, get_distance_metric
from mood.metrics import Metric


MOOD_CRITERIA = [
    "Performance",
    "Domain Weighted Performance",
    "Distance Weighted Performance",
    "Calibration",
    "Calibration x Performance"
]


def get_mood_criteria(performance_metric, calibration_metric):
    """Endpoint for easily creating a criterion by name"""

    return {
        "Performance": PerformanceCriterion(performance_metric),
        "Calibration": CalibrationCriterion(calibration_metric),
        "Domain Weighted Performance": DomainWeightedPerformanceCriterion(performance_metric),
        "Distance Weighted Performance": DistanceWeightedPerformanceCriterion(performance_metric),
        "Calibration x Performance": CombinedCriterion(performance_metric, calibration_metric),
    }


class ModelSelectionCriterion(abc.ABC):
    """
    In MOOD, we argue that one of the tools to improve _model selection_, is the criterion we use
    to select. Besides selecting for raw, validation performance we suspect there could be better alternatives.
    This class defines the interface for a criterion to implement.

    We distinguish multiple iterations within a hyper-parameter search trial.
    For example, you might train and evaluate a model on N different splits before scoring the hyper-parameters.
    """

    def __init__(self, mode, needs_uncertainty: bool):
        self.mode = mode
        self.needs_uncertainty = needs_uncertainty
        self.scores = []

    @abc.abstractmethod
    def score(self, predictions, uncertainties, train: SimpleMolecularDataset, val: SimpleMolecularDataset):
        pass

    def compute_weights(self, train: SimpleMolecularDataset, val: SimpleMolecularDataset):
        return None

    def __call__(self, *args, **kwargs):
        return self.score(*args, **kwargs)

    def update(self, predictions, uncertainties, train: SimpleMolecularDataset, val: SimpleMolecularDataset):
        """Scores a single iteration with the hyper-parameter search."""
        self.scores.append(self.score(predictions, uncertainties, train, val))

    def critique(self):
        """Aggregates the scores of individual iterations."""
        if len(self.scores) == 0:
            raise RuntimeError("Cannot critique when no scores have been computed yet")

        if isinstance(self.scores[0], Sequence):
            lengths = set(len(s) for s in self.scores)
            if len(lengths) != 1:
                raise RuntimeError("All scores need to have the same number of dimensions")
            n = lengths.pop()
            return list(np.mean([s[i] for s in self.scores]) for i in range(n))
        else:
            score = np.mean(self.scores)

        self.reset()
        return score

    def reset(self):
        self.scores = []


class PerformanceCriterion(ModelSelectionCriterion):
    """Select models based on the mean validation performance."""

    def __init__(self, metric: Metric):
        super().__init__(mode=metric.mode, needs_uncertainty=False)
        if metric.is_calibration:
            raise ValueError(f"{metric.name} cannot be used with {type(self).__name__}")
        self.metric = metric

    def score(self, predictions, uncertainties, train: SimpleMolecularDataset, val: SimpleMolecularDataset):
        sample_weights = self.compute_weights(train, val)  # noqa
        return self.metric(y_true=val.y, y_pred=predictions, sample_weights=sample_weights)


class DomainWeightedPerformanceCriterion(PerformanceCriterion):
    """Select models based on the mean weighted validation performance, where the weight of each sample is
    1 over the domain frequency of the domain it is part of."""

    def compute_weights(self, train: SimpleMolecularDataset, val: SimpleMolecularDataset):
        _, inverse, counts = np.unique(val.domains, return_counts=True, return_inverse=True)
        counts = [n / sum(counts) for n in counts]
        weights = [counts[idx] for idx in inverse]
        return weights


class DistanceWeightedPerformanceCriterion(PerformanceCriterion):
    """Select models based on the mean weighted validation performance,
    where the weight of each sample is its distance to the train set."""

    def compute_weights(self, train: SimpleMolecularDataset, val: SimpleMolecularDataset):
        distance_metric = get_distance_metric(val.X)
        return compute_knn_distance(train.X, val.X, distance_metric, k=5, n_jobs=-1)


class CalibrationCriterion(ModelSelectionCriterion):
    """Select a model based on the mean validation calibration"""

    def __init__(self, metric: Metric):
        super().__init__(mode=metric.mode, needs_uncertainty=True)
        if not metric.is_calibration:
            raise ValueError(f"{metric.name} cannot be used with {type(self).__name__}")
        self.metric = metric

    def score(self, predictions, uncertainties, train: SimpleMolecularDataset, val: SimpleMolecularDataset):
        return self.metric(y_true=val.y, y_pred=predictions, uncertainty=uncertainties)


class CombinedCriterion(ModelSelectionCriterion):
    """Selects a model based on a combined score of the validation calibration
    and the validation performance. Since calibration score is between [0, 1], does so by
    either multiplying (when maximizing) or dividing (when minimizing)
    the performance score by the calibration score"""

    def __init__(self, performance_metric: Metric, calibration_metric: Metric):
        super().__init__(mode=[performance_metric.mode, calibration_metric.mode], needs_uncertainty=True)
        self.performance_criterion = PerformanceCriterion(performance_metric)
        self.calibration_criterion = CalibrationCriterion(calibration_metric)

    def score(self, predictions, uncertainties, train: SimpleMolecularDataset, val: SimpleMolecularDataset):
        prf_score = self.performance_criterion.score(predictions, uncertainties, train, val)
        cal_score = self.calibration_criterion.score(predictions, uncertainties, train, val)
        return prf_score, cal_score
