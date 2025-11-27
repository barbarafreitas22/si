import numpy as np
from si.base.transformer import Transformer
from si.data.dataset import Dataset
from si.statistics import f_classification


class SelectPercentile(Transformer):
    
    def __init__(self, percentile: float, score_func: callable = f_classification, **kwargs):
        """
        Selects the features from the given percentile of a score function 
        and returns a new Dataset with the selected features.

        Parameters
        ----------
        percentile : float
            Percentile (0-100) for selecting features.
        score_func : callable, optional
            Variance analysis function. 
            Default: f_classification.
        """
        super().__init__(**kwargs)
        
        if not (0 <= percentile <= 100):
            raise ValueError("Percentile must be a float between 0 and 100")
        self.percentile = percentile
        self.score_func = score_func
        self.F = None
        self.p = None

    def _fit(self, dataset: Dataset) -> 'SelectPercentile':
        """
        Estimate the F and P values for each feature.

        Parameters
        ----------
        dataset : Dataset
            Dataset object where features are selected.
        
        Returns
        -------
        self : SelectPercentile
            The instance with the F and P values calculated.
        """
        self.F, self.p = self.score_func(dataset)
        return self

    def _transform(self, dataset: Dataset) -> Dataset:
        """
        Selects features with the highest F value up to the specific percentile.

        Parameters
        ----------
        dataset : Dataset
            Dataset object to select features.
        
        Returns
        -------
        dataset : Dataset
            A new Dataset object with the selected features.
        """
        threshold = np.percentile(self.F, 100 - self.percentile)
        
        mask = self.F >= threshold

        if mask.sum() > int(len(self.F) * self.percentile / 100):
            # kind='stable' to maintain original order for equal F values
            sorted_indices = np.argsort(-self.F, kind='stable') 
            num_features = int(len(self.F) * self.percentile / 100)
            selected_indices = sorted_indices[:num_features]
            mask = np.zeros_like(self.F, dtype=bool)
            mask[selected_indices] = True

        selected_features = np.array(dataset.features)[mask]
        return Dataset(X=dataset.X[:, mask], y=dataset.y, features=list(selected_features), label=dataset.label)