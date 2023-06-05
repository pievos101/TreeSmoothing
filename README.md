# Bayesian post-hoc regulariation of random forests

## Usage
Both classes inherit from `ShrinkageEstimator`, which extends `sklearn.base.BaseEstimator`.
Usage of these two classes is entirely analogous, and works just like any other `sklearn` estimator:
- `__init__()` parameters:
    - `base_estimator`: the estimator around which we "wrap" hierarchical shrinkage. This should be a tree-based estimator: `DecisionTreeClassifier`, `RandomForestClassifier`, ... (analogous for `Regressor`s)
    - `shrink_mode`: 2 options:
        - `"hs"`: classical Hierarchical Shrinkage (from Agarwal et al. 2022)
        - `"beta"`: Bayesian post-hoc regulariation
    - `lmb`: lambda hyperparameter
    - `alpha`: alpha hyperparameter
    - `beta`: beta hyperparameter
    - `random_state`: random state for reproducibility
- Other functions: `fit(X, y)`, `predict(X)`, `predict_proba(X)`, `score(X, y)` work just like with any other `sklearn` estimator.

## Notebooks
