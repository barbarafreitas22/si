import numpy as np
from typing import List, Tuple
from si.base.model import Model
from si.data.dataset import Dataset
from si.metrics.accuracy import accuracy


class StackingClassifier(Model):
    """
    Stacking is an ensemble method that combines predictions from several base models 
    and uses another model (meta-model) to generate the final predictions.

    Parameters
    ----------
    models: List[Model]
        The initial set of base models.
    final_model: Model
        The model to make the final predictions (meta-model).
    """

    def __init__(self, models: List[Model], final_model: Model, **kwargs):
        """
        Initializes the Stacking Classifier.
        """
        super().__init__(**kwargs)
        self.models = models
        self.final_model = final_model

    def _get_base_predictions(self, dataset: Dataset) -> np.ndarray:
        """
        Gets predictions from all initial base models for a given dataset, 
        creating the meta-feature matrix.
        
        Returns
        -------
        np.ndarray
            A matrix where each column is the prediction vector from one base model.
        """
        # Get predictions from all initial base models
        predictions = [model.predict(dataset) for model in self.models]
        
        # Transpose to get (n_samples, n_models)
        return np.array(predictions).T


    def _fit(self, dataset: Dataset) -> 'StackingClassifier':
        """
        Trains the ensemble models by fitting the base models and then training 
        the meta-model on their predictions.

        Parameters
        ----------
        dataset: Dataset
            The training dataset.

        Returns
        -------
        self: StackingClassifier
            The fitted model.
        """
        # Train each of the initial base models 
        for model in self.models:
            model.fit(dataset)
            
        # Get predictions from the initial set of models (Meta-Features)
        X_meta = self._get_base_predictions(dataset)
        
        # Create temporary meta-dataset for training the final model
        meta_dataset = Dataset(X=X_meta, y=dataset.y, label=dataset.label)
        self.final_model.fit(meta_dataset)

        self.is_fitted = True
        return self

    def _predict(self, dataset: Dataset) -> np.ndarray:
        """
        Predicts the labels using the ensemble models.
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction.")
            
        
        X_meta = self._get_base_predictions(dataset)
        meta_dataset = Dataset(X=X_meta, y=None, label=None)
        
        # Get the final predictions using the final model
        final_predictions = self.final_model.predict(meta_dataset)
        
        return final_predictions

    def _score(self, dataset: Dataset, predictions: np.ndarray) -> float:
        """
        Computes the accuracy between predicted and real labels.
        
        Parameters
        ----------
        dataset: Dataset
            Dataset with true labels.
        predictions: np.ndarray
            Predicted labels, supplied by the base Model.score() method.

        Returns
        -------
        float
            The accuracy score.
        """
        return accuracy(dataset.y, predictions)