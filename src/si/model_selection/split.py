from typing import Tuple

import numpy as np

from si.data.dataset import Dataset


def train_test_split(dataset: Dataset, test_size: float = 0.2, random_state: int = 42) -> Tuple[Dataset, Dataset]:
    """
    Split the dataset into training and testing sets

    Parameters
    ----------
    dataset: Dataset
        The dataset to split
    test_size: float
        The proportion of the dataset to include in the test split
    random_state: int
        The seed of the random number generator

    Returns
    -------
    train: Dataset
        The training dataset
    test: Dataset
        The testing dataset
    """
    # set random state
    np.random.seed(random_state)
    # get dataset size
    n_samples = dataset.shape()[0]
    # get number of samples in the test set
    n_test = int(n_samples * test_size)
    # get the dataset permutations
    permutations = np.random.permutation(n_samples)
    # get samples in the test set
    test_idxs = permutations[:n_test]
    # get samples in the training set
    train_idxs = permutations[n_test:]
    # get the training and testing datasets
    train = Dataset(dataset.X[train_idxs], dataset.y[train_idxs], features=dataset.features, label=dataset.label)
    test = Dataset(dataset.X[test_idxs], dataset.y[test_idxs], features=dataset.features, label=dataset.label)
    return train, test

def stratified_train_test_split(dataset: Dataset, test_size: float = 0.3, random_state: int = None) -> Tuple[Dataset, Dataset]:
    """
    Splits the dataset into training and testing sets, preserving the proportion 
    of each class (stratification).

    Parameters
    ----------
    dataset: Dataset
        The Dataset object to split into training and testing data.
    test_size: float, default=0.3
        The size of the testing Dataset (e.g., 0.2 for 20%).
    random_state: int or None, default=None
        Seed for generating permutations, ensuring reproducibility.

    Returns
    -------
    Tuple[Dataset, Dataset]
        A tuple containing the stratified train and test Dataset objects.
    """
    

    if not dataset.has_label():
        raise ValueError("Dataset must have labels for stratified splitting.")
    if not (0.0 < test_size < 1.0):
        raise ValueError("test_size must be a float between 0.0 and 1.0.")
        
    # Set random state for reproducibility
    rng = np.random.RandomState(random_state)

    classes = dataset.get_classes()
    train_indices = []
    test_indices = []
    
    # Loop through unique labels
    for class_ in classes:
        class_indices = np.where(dataset.y == class_)[0]
        n_class_samples = len(class_indices)
        
        # Calculate the number of test samples for the current class
        n_test_samples = int(np.ceil(test_size * n_class_samples))
        
        # Shuffle and select indices for the current class
        rng.shuffle(class_indices)
        test_indices.extend(class_indices[:n_test_samples])
        train_indices.extend(class_indices[n_test_samples:])
        
    # 7. Create train and test datasets
    
    train_indices = np.array(train_indices)
    test_indices = np.array(test_indices)
    X_train = dataset.X[train_indices]
    y_train = dataset.y[train_indices]
    X_test = dataset.X[test_indices]
    y_test = dataset.y[test_indices]
    train_dataset = Dataset(X=X_train, y=y_train, features=dataset.features, label=dataset.label)
    test_dataset = Dataset(X=X_test, y=y_test, features=dataset.features, label=dataset.label)
    
    return train_dataset, test_dataset