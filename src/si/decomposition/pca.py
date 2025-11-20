import numpy as np
from si.base.transformer import Transformer
from si.data.dataset import Dataset


class PCA(Transformer):
    """
    Principal Component Analysis (PCA) is a linear algebra technique
    to reduce the dimensionality of a data set using eigenvalue decomposition.

    Parameters
    ----------
    n_components : int
        The number of principal components to keep.

    Attributes
    ----------
    mean: np.ndarray
        The mean of each feature used for centering the data.
    components: np.ndarray
        The selected principal components (a matrix where each row is an eigenvector).
    explained_variance: np.ndarray
        The amount of variance explained by each principal component (vector of eigenvalues).
    """

    def __init__(self, n_components: int, **kwargs):
        """
        Initializes the PCA with the desired number of principal components.
        """
        super().__init__(**kwargs)
        self.n_components = n_components
        self.mean = None
        self.components = None
        self.explained_variance = None
        self.is_fitted = False

    def _fit(self, dataset: Dataset) -> "PCA":
        """
        Estimates the mean, principal components, and explained variance.

        Parameters
        ----------
        dataset : Dataset
            Dataset object containing the input data.

        Return
        -------
        self : PCA
            The fitted PCA object.
        """
        if not (1 <= self.n_components <= dataset.shape()[1]):
            raise ValueError(
                "The number of components must be between 1 and the number of variables in the data set."
            )
        self.mean = dataset.get_mean()
        X_centered = dataset.X - self.mean
        covariance_matrix = np.cov(X_centered, rowvar=False)
        eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)
        sorted_indices = np.argsort(eigenvalues)[::-1][:self.n_components]
        self.components = eigenvectors[:, sorted_indices].T   #transpose so each row is a principal component (V^T)
        self.explained_variance = eigenvalues[sorted_indices] / np.sum(eigenvalues)
        self.is_fitted = True
        return self

    def _transform(self, dataset: Dataset) -> Dataset:
        """
        Calculates the reduced dataset using the principal components.

        Parameters
        ----------
        dataset : Dataset
            Data set to be transformed.

        Return
        -------
        Dataset
            Reduced data set with the new principal components.
        """
        if not self.is_fitted:
            raise ValueError("The PCA needs to be adjusted before it can be used to transform the data.")

        # 1. Center the data using the mean inferred in _fit
        X_centered = dataset.X - self.mean

        # 2. Project the centered data into the principal components
        # self.components is stored as V^T, we use the transpose here to get V
        X_reduced = np.dot(X_centered, self.components.T)

        return Dataset(
            X=X_reduced,
            y=dataset.y,
            features=[f"PC{i + 1}" for i in range(self.n_components)],
            label=dataset.label
        )