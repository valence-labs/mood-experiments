import itertools

from mood.dataset import MOOD_DATASETS
from mood.representations import MOOD_REPRESENTATIONS
from mood.splitter import MOOD_SPLITTERS
from mood.baselines import SUPPORTED_BASELINES


# TODO: Update based on actual implementation
MOOD_ALGORITHMS = SUPPORTED_BASELINES + ["CORAL", "DANN", "Mixup", "IB-ERM", "MTL", "VREx"]


# TODO: Update based on actual implementation
MOOD_CRITERIA = ["performance", "calibration", "domain_weighted", "distance_weighted", "performance x calibration"]

NUM_SEEDS = 128


def get_experimental_configurations():
    all_options = itertools.product(
        MOOD_SPLITTERS,
        MOOD_REPRESENTATIONS,
        MOOD_ALGORITHMS,
        MOOD_CRITERIA,
        list(range(10))
    )
    print(len(list(all_options)))



if __name__ == "__main__":
    get_experimental_configurations()