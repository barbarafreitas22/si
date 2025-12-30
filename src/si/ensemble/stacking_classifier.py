import numpy as np
from si.metrics.accuracy import accuracy
from si.data.dataset import Dataset
from si.base.model import Model

class StackingClassifier(Model):
    """
    StackingClassifier implements a stacking ensemble. It uses a set of initial models 
    (base models) to generate predictions, which are used as new features 
    to train a final model (meta-model) for the classification task.
    """

    def __init__(self, models: list, final_model: Model, **kwargs):
        """
        Initialize the StackingClassifier with the base models and a final model.

        Parameters
        ----------
        models : list
            List of base models used to make initial predictions.
        final_model : Model
            The meta-model that combines predictions from the base models.
        """
        super().__init__(**kwargs)
        self.models = models
        self.final_model = final_model
        self.new_dataset = None

    def _fit(self, dataset: Dataset) -> "StackingClassifier":
        """
        Train the StackingClassifier by fitting the base models and the meta-model.

        Parameters
        ----------
        dataset : Dataset
            The training dataset.

        Returns
        -------
        StackingClassifier
            The trained StackingClassifier instance.
        """
        for model in self.models:
            model.fit(dataset)

        base_predictions = np.array([model.predict(dataset) for model in self.models]).T

        self.new_dataset = Dataset(
            X=base_predictions,
            y=dataset.y,
            features=[f"model_{i}" for i in range(len(self.models))],
            label=dataset.label
        )

        self.final_model.fit(self.new_dataset)

        return self

    def _predict(self, dataset: Dataset) -> np.ndarray:
        """
        Predict the class labels using the trained StackingClassifier.

        Parameters
        ----------
        dataset : Dataset
            The dataset for which predictions are made.

        Returns
        -------
        np.ndarray
            Predicted class labels for the input dataset.
        """
        base_predictions = np.array([model.predict(dataset) for model in self.models]).T

        new_dataset = Dataset(
            X=base_predictions,
            y=None,
            features=[f"model_{i}" for i in range(len(self.models))],
            label=None
        )

        return self.final_model.predict(new_dataset)

    def _score(self, dataset: Dataset, predictions: np.ndarray) -> float:
        """
        Calculate the accuracy of the StackingClassifier.

        Parameters
        ----------
        dataset : Dataset
            Dataset containing true labels.
        predictions : np.ndarray
            Predictions made by the model.

        Returns
        -------
        float
            The accuracy of the model on the given dataset.
        """
        return accuracy(dataset.y, predictions)