import itertools
import zlib

import numpy as np

from mood.model import MOOD_DA_DG_ALGORITHMS
from mood.representations import MOOD_REPRESENTATIONS
from mood.splitter import MOOD_SPLITTERS
from mood.baselines import MOOD_BASELINES
from mood.criteria import get_mood_criteria
from mood.metrics import Metric


RCT_SEED = 1234
NUM_SEEDS = 10


def get_experimental_configurations(dataset):
    """
    To randomly sample different configurations of the RCT experiment, we use a deterministic approach.
    This facilitates reproducibility, but also makes it easy to run the experiment on preemptible instances.
    Otherwise, it could happen that models that take longer to train have a higher chance of failing,
    biasing the experiment.
    """

    # NOTE: Despite adding more convenient ways of getting all algorithms and criteria,
    #  using this would change the ordering of the RCT, which is why we kept the more clunky code.

    prf_metric = Metric.get_default_performance_metric(dataset)
    cal_metric = Metric.get_default_calibration_metric(dataset)

    mood_criteria = get_mood_criteria(prf_metric, cal_metric).keys()

    mood_baselines = MOOD_BASELINES.copy()
    mood_baselines.pop(mood_baselines.index("MLP"))
    mood_algorithms = mood_baselines + list(MOOD_DA_DG_ALGORITHMS.keys())

    all_options = list(
        itertools.product(
            mood_algorithms,
            MOOD_REPRESENTATIONS,
            MOOD_SPLITTERS,
            mood_criteria,
            list(range(NUM_SEEDS)),
        )
    )
    # NOTE: We add the hash of the dataset to make the sampled configurations dataset-dependent
    rng = np.random.default_rng(RCT_SEED + zlib.adler32(dataset.encode("utf-8")))
    rng.shuffle(all_options)
    return all_options
