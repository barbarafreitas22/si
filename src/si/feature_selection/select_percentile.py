import numpy as np
from si.base.transformer import Transformer
from si.data.dataset import Dataset
from si.statistics.f_classification import f_classification


class SelectPercentile(Transformer):
    """
    Select features according to a percentile of the highest scores.
    Feature selection is performed using a scoring function
    that computes a score (and p-value) for each feature.
    """

    def __init__(self, percentile: float, score_func: callable =f_classification, **kwargs):
        """
        Selects features from the given percentile of a score function 
        and returns a new Dataset object with the selected features.

        Parameters
        ----------
        score_func: callable
            Function to compute the score and p-values.
            Default: f_classification
        percentile: float
            Percentile used for select feautures.
        """

        super().__init__(**kwargs)

        if not (0 <= percentile <= 100):
            raise ValueError("Percentile must be a float between 0 and 100")
        
        self.score_func = score_func
        self.percentile = percentile
        self.F = None
        self.p = None

    def _fit(self, dataset: Dataset) -> 'SelectPercentile':
        """
        Computes the F-scores and p-values for each of the dataset features using the score function.

        Parameters
        ----------
        dataset: Dataset
            The dataset where feautures are selected.

        Returns
        -------
        self: SelectPercentile
            The instance with the F and P values calculated.
        """
        self.F, self.p = self.score_func(dataset)
        return self

    def _transform(self, dataset: Dataset) -> Dataset:
        """
        Selects the features with the highest scores according to the specified percentile.
        Handles ties by selecting the first features encountered (lowest index).

        Parameters
        ----------
        dataset: Dataset
            The dataset to transform.

        Returns
        -------
        dataset: Dataset
            The transformed dataset with the selected features.
        """
        len_features = len(dataset.features)
        k = int(len_features * (self.percentile / 100))

        #'Stable sort' preserves original feature index order for tie-breaking.
        sorted_indices = np.argsort(-self.F, kind='stable')
        best_indices = sorted_indices[:k]
        
        # Sort the best indices to maintain original feature order
        best_indices.sort()
        X_selected = dataset.X[:, best_indices]
        features_selected = [dataset.features[i] for i in best_indices]
        
        return Dataset(X=X_selected, y=dataset.y, features=features_selected, label=dataset.label)