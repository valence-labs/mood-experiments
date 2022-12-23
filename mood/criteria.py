import abc

import numpy as np

from mood.dataset import SimpleMolecularDataset
from mood.distance import compute_knn_distance, get_distance_metric
from mood.metrics import Metric


def get_mood_criteria(performance_metric, calibration_metric):
    return {
        "Performance": PerformanceCriterion(performance_metric),
        "Calibration": CalibrationCriterion(calibration_metric),
        "Domain Weighted Performance": DomainWeightedPerformanceCriterion(performance_metric),
        "Distance Weighted Performance": DistanceWeightedPerformanceCriterion(performance_metric),
        "Calibration x Performance": CombinedCriterion(performance_metric, calibration_metric),
    }


class ModelSelectionCriterion(abc.ABC):

    def __init__(self, mode):
        self.mode = mode
        self.scores = []

    @abc.abstractmethod
    def score(self, predictions, uncertainties, train: SimpleMolecularDataset, val: SimpleMolecularDataset):
        pass

    def compute_weights(self, train: SimpleMolecularDataset, val: SimpleMolecularDataset):
        return None

    def __call__(self, *args, **kwargs):
        return self.score(*args, **kwargs)

    def update(self, predictions, uncertainties, train: SimpleMolecularDataset, val: SimpleMolecularDataset):
        self.scores.append(self.score(predictions, uncertainties, train, val))

    def critique(self):
        score = np.mean(self.scores)
        self.reset()
        return score

    def reset(self):
        self.scores = []


class PerformanceCriterion(ModelSelectionCriterion):

    def __init__(self, metric: Metric):
        super().__init__(mode=metric.mode)
        if metric.needs_uncertainty or not metric.needs_predictions:
            raise ValueError(f"{metric.name} cannot be used with {type(self).__name__}")
        self.metric = metric

    def score(self, predictions, uncertainties, train: SimpleMolecularDataset, val: SimpleMolecularDataset):
        sample_weights = self.compute_weights(train, val)  # noqa
        return self.metric(y_true=val.y, y_pred=predictions, sample_weights=sample_weights)


class DomainWeightedPerformanceCriterion(PerformanceCriterion):

    def compute_weights(self, train: SimpleMolecularDataset, val: SimpleMolecularDataset):
        _, inverse, counts = np.unique(val.domains, return_counts=True, return_inverse=True)
        counts = [n / sum(counts) for n in counts]
        weights = [counts[idx] for idx in inverse]
        return weights


class DistanceWeightedPerformanceCriterion(PerformanceCriterion):

    def compute_weights(self, train: SimpleMolecularDataset, val: SimpleMolecularDataset):
        distance_metric = get_distance_metric(val.X)
        return compute_knn_distance(train.X, val.X, distance_metric, k=5, n_jobs=-1)


class CalibrationCriterion(ModelSelectionCriterion):

    def __init__(self, metric: Metric):
        super().__init__(mode=metric.mode)
        if not metric.needs_uncertainty:
            raise ValueError(f"{metric.name} cannot be used with {type(self).__name__}")
        self.metric = metric

    def score(self, predictions, uncertainties, train: SimpleMolecularDataset, val: SimpleMolecularDataset):
        return self.metric(y_true=val.y, y_pred=predictions, uncertainty=uncertainties)


class CombinedCriterion(ModelSelectionCriterion):

    def __init__(self, performance_metric: Metric, calibration_metric: Metric):
        super().__init__(mode=performance_metric.mode)

        vmin, vmax = calibration_metric.range
        if vmin is None or vmax is None:
            raise ValueError(f"{calibration_metric.name} cannot be used with {type(self).__name__}")

        self.performance_criterion = PerformanceCriterion(performance_metric)
        self.calibration_criterion = CalibrationCriterion(calibration_metric)

    def score(self, predictions, uncertainties, train: SimpleMolecularDataset, val: SimpleMolecularDataset):
        prf_score = self.performance_criterion.score(predictions, uncertainties, train, val)
        cal_score = self.calibration_criterion.score(predictions, uncertainties, train, val)

        # Normalize to [0, 1]
        vmin, vmax = self.calibration_criterion.metric.range
        cal_score = (cal_score + abs(vmin)) / (vmax - vmin)

        # Since the calibration score is between 0 and 1,
        if self.mode == "min":
            score = prf_score / cal_score
        else:
            score = prf_score * cal_score

        return score
