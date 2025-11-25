import numpy as np
from typing import Callable, Dict, List, Any, Union, Optional
from si.data.dataset import Dataset
from si.metrics.accuracy import accuracy 
from si.model_selection.cross_validate import k_fold_cross_validation

# Type definition for the output dictionary
Results = Dict[str, Union[List[Dict[str, Any]], List[float], Dict[str, Any], float]]


def randomized_search_cv(
    model: Any, 
    dataset: Dataset, 
    hyperparameter_grid: Dict[str, np.ndarray], 
    scoring: Callable = accuracy, 
    cv: int = 5, 
    n_iter: int = 10,
    random_state: Optional[int] = None
) -> Results:
    """
    Implements a parameter optimization strategy with cross-validation using 
    a number of random combinations selected from a distribution of hyperparameters.
    
    This implementation prioritizes scalability by sampling indices directly from 
    the provided hyperparameter distributions.

    Parameters
    ----------
    model: Model
        The model to validate.
    dataset: Dataset
        The validation dataset.
    hyperparameter_grid: Dict[str, np.ndarray]
        Dictionary with hyperparameter name and search values (distributions).
    scoring: Callable, default=accuracy
        The scoring function.
    cv: int, default=5
        Number of folds for cross-validation.
    n_iter: int, default=10
        Number of hyperparameter random combinations to test.
    random_state: Optional[int], default=None
        Seed for reproducibility.

    Returns
    -------
    Dict
        Dictionary with the results of the randomized search cross-validation.
        Includes scores, hyperparameters, best hyperparameters, and best score.
    """

    rng = np.random.RandomState(random_state)
    
    for hyperparameter in hyperparameter_grid:
        if not hasattr(model, hyperparameter):
            raise ValueError(f"Hyperparameter '{hyperparameter}' is not an attribute of model {model.__class__.__name__}.")

    results: Results = {
        'hyperparameters': [],
        'scores': [],
        'best_hyperparameters': {},
        'best_score': -np.inf
    }
    
    # Get n_iter hyperparameter combinations 
    
    random_indices = [
        rng.choice(len(grid), size=n_iter, replace=False) 
        for grid in hyperparameter_grid.values()
    ]
   
    for i in range(n_iter):
        current_hyperparameters = {}

        # Create the current hyperparameter combination
        for j, (param_name, param_grid) in enumerate(hyperparameter_grid.items()):
            index = random_indices[j][i]
            value = param_grid[index]
            current_hyperparameters[param_name] = value
            
        # Set the model hyperparameters with the current combination
        for param_name, value in current_hyperparameters.items():
            setattr(model, param_name, value)
            
        # Cross validate the model 
        cv_scores = k_fold_cross_validation(model, dataset, scoring, cv)
        
        # Save the mean of the scores and respective hyperparameters
        mean_score = np.mean(cv_scores)
        
        results['hyperparameters'].append(current_hyperparameters)
        results['scores'].append(mean_score)
        
        # Save the best score and respective hyperparameters
        if mean_score > results['best_score']:
            results['best_score'] = mean_score
            results['best_hyperparameters'] = current_hyperparameters
            
    # Return the dictionary with all results
    return results