import numpy as np
import itertools
from si.base.model import Model
from si.data.dataset import Dataset
from si.model_selection.cross_validate import k_fold_cross_validation

def randomized_search_cv(model: Model, dataset: Dataset, hyperparameter_grid: dict, cv: int, n_iter: int, scoring: callable = None) -> dict:
    """
    Performs a randomized search cross-validation on a model.
    
    This function implements a parameter optimization strategy that selects a fixed 
    number of random combinations (n_iter) from a hyperparameter grid and evaluates 
    them using Cross-Validation. 

    Parameters
    ----------
    model : Model
        The machine learning model to be tuned.
    dataset : Dataset
        The validation dataset containing features (X) and labels (y).
    hyperparameter_grid : dict
        A dictionary where keys are parameter names (str) and values are lists or 
        distributions of possible values to be sampled.
    cv : int
        The number of folds to use for Cross-Validation.
    n_iter : int
        The number of random hyperparameter combinations to sample and evaluate.
    scoring : callable, optional
        A scoring function. If None, the model's default score method is used.

    Returns
    -------
    dict
        A dictionary containing the results of the search:
        - 'scores': List of mean CV scores for each tested combination.
        - 'hyperparameters': List of the hyperparameter dictionaries corresponding to the scores.
        - 'best_hyperparameters': The dictionary of hyperparameters that achieved the highest score.
        - 'best_score': The highest score obtained.
        
    Raises
    ------
    AttributeError
        If a parameter in hyperparameter_grid does not exist as an attribute in the model.
    """
    # Verfiy that all hyperparameters exist in the model
    for parameter in hyperparameter_grid:
        if not hasattr(model, parameter):
            raise AttributeError(f"Model {type(model).__name__} does not have parameter '{parameter}'.")

    # Obtain random hyperparameter combinations
    combinations = random_combinations(hyperparameter_grid, n_iter)

    results = {'scores': [], 'hyperparameters': []}

    for combination in combinations:
        # Reconstruct hyperparameter dictionary
        parameters = {param: value for param, value in zip(hyperparameter_grid.keys(), combination)}
        
        # Configure model with current hyperparameters
        for param, value in parameters.items():
            setattr(model, param, value)

        # Cross Validation to evaluate model
        scores = k_fold_cross_validation(model=model, dataset=dataset, scoring=scoring, cv=cv)

        # Save results
        results['scores'].append(np.mean(scores))
        results['hyperparameters'].append(parameters)

    # Find the best hyperparameters
    best_idx = np.argmax(results['scores'])
    results['best_hyperparameters'] = results['hyperparameters'][best_idx]
    results['best_score'] = results['scores'][best_idx]

    return results

def random_combinations(hyperparameter_grid: dict, n_iter: int) -> list:
    """
    Randomly select a specified number of hyperparameter combinations.
    """
    all_combinations = list(itertools.product(*hyperparameter_grid.values()))

    # Adjust n_iter if it exceeds the number of combinations
    num_combinations = len(all_combinations)
    if n_iter > num_combinations:
        n_iter = num_combinations
        
    # Select random indices
    random_indices = np.random.choice(num_combinations, size=n_iter, replace=False)
    
    return [all_combinations[i] for i in random_indices]