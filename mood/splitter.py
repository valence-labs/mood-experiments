import tqdm 

import numpy as np
import datamol as dm
import seaborn as sns
import matplotlib.pyplot as plt

from matplotlib.cm import ScalarMappable
from dataclasses import dataclass
from loguru import logger
from collections import defaultdict
from typing import Union, List, Optional, Callable, Dict

from scipy.stats import wasserstein_distance
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import BaseShuffleSplit, GroupShuffleSplit
from sklearn.cluster import KMeans

    
@dataclass
class SplitCharacterization:
    """
    Within the context of MOOD, a split is characterized by 
    a distribution of distances and an associated representativeness score
    """
    distances: np.ndarray
    representativeness: float
    label: str
    
    @classmethod
    def concat(cls, splits):
        
        names = set([obj.label for obj in splits])
        if len(names) != 1:
            raise RuntimeError("Can only concatenate equally labeled split characterizations")
        
        dist = np.concatenate([obj.distances for obj in splits])
        score = np.mean([obj.representativeness for obj in splits])
        return cls(dist, score, names.pop())
    
    @staticmethod
    def best(splits):
        return max(splits, key=lambda spl: spl.representativeness)
    
    def __repr__(self):
        return self.__str__()
    
    def __str__(self):
        return f"{self.__class__.__name__}[{self.label}]"


class MOODSplitter(BaseShuffleSplit):
    """
    The MOOD splitter takes in multiple splitters and a set of 
    downstream molecules and prescribes one splitting method
    that creates the test set that is most representative of
    downstream applications.
    """
    
    def __init__(
        self, 
        splitters: Dict[str, BaseShuffleSplit], 
        downstream_distances: Optional[np.ndarray] = None,
        metric: Union[str, Callable] = "minkowski", 
        p: int = 2,
        k: int = 5,
    ):
        """
        Args:
            splitters: A list of splitter methods you are considering
            downstream_distances: A list of precomputed distances for the downstream application
            metric: The distance metric to use
            p: If the metric is the minkowski distance, this is the p in that distance.
            k: The number of nearest neighbors to use to compute the distance.
        """
        if not all(isinstance(obj, BaseShuffleSplit) for obj in splitters.values()):
            raise TypeError("All splitters should be BaseShuffleSplit objects")
        
        n_splits_per_splitter = [obj.get_n_splits() for obj in splitters.values()]
        if not len(set(n_splits_per_splitter)) == 1:
            raise TypeError("n_splits is inconsistent across the different splitters")
        self._n_splits = n_splits_per_splitter[0]
        
        self._p = p
        self._k = k
        self._metric = metric
        self._splitters = splitters
        self._downstream_distances = downstream_distances
        
        self._prescribed_splitter = None

    @staticmethod
    def visualize(downstream_distances: np.ndarray, splits: List[SplitCharacterization]):
        fig, ax = plt.subplots(figsize=(12, 6))
        
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_visible(False)
        ax.spines["top"].set_visible(False)
        ax.yaxis.set_ticklabels([])
        ax.yaxis.set_ticks([])
        
        cmap = sns.color_palette("rocket", len(splits) + 1)
        splits = sorted(splits, key=lambda spl: spl.representativeness)
        
        # Visualize the distribution of the downstream application
        sns.kdeplot(
            downstream_distances, 
            color=cmap[0], 
            alpha=0.2, 
            linestyle="--", 
            label="Downstream application",
        )
        
        # Visualize all splitting methods
        for rank, spl in enumerate(splits): 
            sns.kdeplot(spl.distances, color=cmap[rank + 1], ax=ax, label=spl.label)
        ax.set_xlabel(f"Distance")
        ax.legend()
        
        # Add a colorbar
        cbar = fig.colorbar(ScalarMappable(cmap="rocket"))
        cbar.set_label("Representativeness to downstream application (Rank)", rotation=270, labelpad=15)
        
        return ax
    
    @property
    def get_prescribed_splitter():
        if not self.fitted:
            raise RuntimeError("The splitter has not be fitted yet")
        return self._prescribed_splitter
    
    @property
    def fitted(self):
        return self._prescribed_splitter is not None
   
    def _compute_distance(self, X_from, X_to):
        """
        Computes the k-NN distance from one set to another
        
        Args:
            X_from: The set to compute the distance for
            X_to: The set to compute the distance to (i.e. the neighbor candidates)
        """            
        knn = NearestNeighbors(n_neighbors=self._k, metric=self._metric, p=self._p).fit(X_to)
        distances, ind = knn.kneighbors(X_from)
        distances = np.mean(distances, axis=1)
        return distances
    
    def score_representativeness(self, distances):
        """Scores a representativeness score between two distributions
        A higher score should be interpreted as _more_ representative"""
        if self._downstream_distances is None: 
            raise RuntimeError("No downstream distances provided or computed yet.")
        return -wasserstein_distance(distances, self._downstream_distances)
    
    def fit(self, X, y=None, groups=None, X_downstream=None, plot: bool = False, progress: bool = False):
        """Follows the MOOD specification to prescribe a train-test split
        that is most representative of downstream applications.
        
        In MOOD, the k-NN distance in the representation space functions 
        as a proxy of difficulty. The further a datapoint is from the training
        set, in general the lower a model's performance. Using that observation, 
        we prescribe the train-test split that best replicates the distance
        distribution (i.e. "the difficulty") of a downstream application.
        
        TODO: A limitation is that you cannot have different groups for different splits
            You would therefore need a separate splitter class for each of the groupings
            one is interested in.
        """
        
        if self._downstream_distances is None:
            self._downstream_distances = self._compute_distance(X_downstream, X)
        
        # Precompute all splits. Since splitters are implemented as generators,
        # we store the resulting splits so we can replicate them later on.
        split_chars = list()
        
        it = self._splitters.items()
        if progress:
            it = tqdm.tqdm(it, desc="Splitter")
            
        for name, splitter in it:
               
            # We possibly repeat the split multiple times to 
            # get a more reliable  estimate
            chars = []

            it_ = splitter.split(X, y, groups)
            if progress:
                it_ = tqdm.tqdm(it_, leave=False, desc="Split", total=self._n_splits)

            for split in it_:
                train, test = split
                distances = self._compute_distance(X[test], X[train])
                score = self.score_representativeness(distances)
                chars.append(SplitCharacterization(distances, score, name))
            
            split_chars.append(SplitCharacterization.concat(chars))
        
        # Rank different splitting methods by their ability to 
        # replicate the downstream distance distribution.
        chosen = SplitCharacterization.best(split_chars)
        self._prescribed_splitter = self._splitters[chosen.label]
        
        logger.info(f"Selected {chosen.label} as the most representative splitting method")
        
        if plot:
            # Visualize the results
            return self.visualize(
                self._downstream_distances, 
                split_chars
            )
    
    def _iter_indices(self, X=None, y=None, groups=None):
        """Generate (train, test) indices"""
        if not self.fitted:
            raise RuntimeError("The splitter has not be fitted yet")
        yield from self._prescribed_splitter._iter_indices(X, y, groups)


class PredefinedGroupShuffleSplit(GroupShuffleSplit):
    """Simple class that tackles the limitation of the MOODSplitter
    that all splitters need to use the same grouping."""
    
    def __init__(self, groups, n_splits=5, *, test_size=None, train_size=None, random_state=None):
        super().__init__(
            n_splits=n_splits,
            test_size=test_size,
            train_size=train_size,
            random_state=random_state,
        )
        self._groups = groups
        
    def _iter_indices(self, X=None, y=None, groups=None):
        """Generate (train, test) indices"""
        if groups is not None: 
            logger.warning("Ignoring the groups parameter in favor of the predefined groups")
        yield from super()._iter_indices(X, y, self._groups)


class KMeanSplit(GroupShuffleSplit):
    """Split based on the k-Mean clustering in input space"""
    
    def __init__(
        self, 
        metric: str = "euclidean", 
        n_clusters: int = 10, 
        n_splits=5, 
        *, 
        test_size=None, 
        train_size=None, 
        random_state=None
    ):
        
        super().__init__(
            n_splits=n_splits,
            test_size=test_size,
            train_size=train_size,
            random_state=random_state,
        )
        self._metric = metric
        self._n_clusters = n_clusters
        
        
    def _iter_indices(self, X=None, y=None, groups=None):
        """Generate (train, test) indices"""
        if groups is not None: 
            logger.warning("Ignoring the groups parameter in favor of the predefined groups")
        
        model = KMeans(self._n_clusters)
        model.fit(X)
        groups = model.labels_
        
        yield from super()._iter_indices(X, y, groups)
