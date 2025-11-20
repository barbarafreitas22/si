from typing import Callable, Union, Optional, List, Tuple
import numpy as np
from scipy.stats import mode 
from si.metrics.accuracy import accuracy
from si.models.decision_tree_classifier import DecisionTreeClassifier
from si.data.dataset import Dataset
from si.base.model import Model


class RandomForestClassifier(Model):
    """
    Random Forest is an ensemble machine learning technique that constructs multiple 
    Decision Trees and combines their individual predictions to improve predictive 
    accuracy and mitigate the risk of overfitting.
    """

    def __init__(self, n_estimators: int = 100, max_features: Optional[int] = None, 
                 min_sample_split: int = 5, max_depth: int = 10, 
                 mode: str = "gini", seed: int = 123, **kwargs):
        """
        Initialize the Random Forest Classifier with specified hyperparameters.

        Parameters
        ----------
        n_estimators : int, default=100
            Number of decision trees in the forest.
        max_features : Optional[int], default=None
            Maximum number of features to consider for each split (random feature subsetting).
        min_sample_split : int, default=5
            Minimum number of samples required to split a node (passed to base estimator).
        max_depth : int, default=10
            Maximum depth of the trees (passed to base estimator).
        mode : str, default="gini"
            Impurity calculation mode ("gini" or "entropy") (passed to base estimator).
        seed : int, default=123
            Random seed for reproducibility.
        """
        super().__init__(**kwargs)
        self.n_estimators = n_estimators
        self.max_features = max_features
        self.min_sample_split = min_sample_split
        self.max_depth = max_depth
        self.mode = mode if mode in {"gini", "entropy"} else "gini"
        self.seed = seed
        self.trees: List[Tuple[List[str], DecisionTreeClassifier]] = [] # Stores (features_names, tree_instance)
        
        # Attributes for Label Encoding/Decoding
        self.label_to_int = None
        self.int_to_label = None


    def _fit(self, dataset: Dataset) -> "RandomForestClassifier":
        """
        Train the Random Forest by building decision trees on bootstrap samples.
        """
        np.random.seed(self.seed)

        # Label Encoding: Convert string labels to integers for np.bincount
        unique_labels = np.unique(dataset.y)
        self.label_to_int = {label: i for i, label in enumerate(unique_labels)}
        self.int_to_label = {i: label for i, label in enumerate(unique_labels)}
        
        # Use integer labels for training
        y_int = np.array([self.label_to_int[label] for label in dataset.y], dtype=int)
        
        # Determine the number of features to randomly select
        if self.max_features is None:
            self.max_features = int(np.sqrt(dataset.shape()[1]))

        # Clear the existing trees
        self.trees = [] 
        n_samples = dataset.shape()[0]
        n_total_features = dataset.shape()[1]

        for _ in range(self.n_estimators):
            # Bootstrap sampling 
            sample_indices = np.random.choice(n_samples, size=n_samples, replace=True)
            
            # Feature subsampling (Random Forest: sample features without replacement)
            feature_indices = np.random.choice(n_total_features, size=self.max_features, replace=False)
            bootstrap_feature_names = [dataset.features[i] for i in feature_indices]

            # Create bootstrap dataset and train the decision tree on it (using y_int)
            bootstrap_dataset = Dataset(
                X=dataset.X[sample_indices][:, feature_indices],
                y=y_int[sample_indices],
                features=bootstrap_feature_names,
                label=dataset.label
            )
            
            tree = DecisionTreeClassifier(
                min_sample_split=self.min_sample_split,
                max_depth=self.max_depth,
                mode=self.mode
            )
            tree.fit(bootstrap_dataset)

            # Store the tree instance and the names of the features it was trained on
            self.trees.append((bootstrap_feature_names, tree))

        return self

    def _predict(self, dataset: Dataset) -> np.ndarray:
        """
        Predict target values for a dataset using majority voting from the ensemble.
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction.")

        all_predictions_int = []
        test_feature_names = list(dataset.features)
        
        for feature_names_used, tree in self.trees:
            
            # Ensure the test data is presented to the tree with only the features the tree was trained on.
            
            # Get the original indices of the test features that match the tree's feature subset
            indices_to_extract = [test_feature_names.index(name) for name in feature_names_used]
            
            subset_X = dataset.X[:, indices_to_extract]
            
            # Create a minimal dataset subset for prediction
            subset = Dataset(
                X=subset_X,
                y=dataset.y,
                features=feature_names_used,
                label=dataset.label
            )
            
            # Predictions are integers (0, 1, 2...)
            all_predictions_int.append(tree.predict(subset))

        # Aggregation: Transpose and prepare matrix for voting
        predictions_matrix = np.array(all_predictions_int).T
        
        # Majority Voting 
        final_predictions_int = np.apply_along_axis(lambda row: np.bincount(row).argmax(), axis=1, arr=predictions_matrix)
        
        # Label Decoding: Convert integer predictions back to original string labels
        return np.array([self.int_to_label[i] for i in final_predictions_int], dtype=object)

    def _score(self, dataset: Dataset, predictions: np.ndarray) -> float:
        """
        Calculates the accuracy between predicted and true labels.

        Parameters
        ----------
        dataset : Dataset
            Dataset with true labels.
        predictions : np.ndarray
            Predicted labels (y_pred), supplied by the base Model.score() method.

        Returns
        -------
        float
            Accuracy score.
        """
        return accuracy(dataset.y, predictions)