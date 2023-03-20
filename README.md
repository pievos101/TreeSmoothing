# Scikit-Learn-compatible implementation of Augmented Hierarchical Shrinkage
This directory contains an implementation of Augmented Hierarchical Shrinkage in the [aughs](aughs) directory that is compatible with Scikit-Learn. It exports 2 classes:
- `ShrinkageClassifier`
- `ShrinkageRegressor`

## Basic API
Both classes inherit from `ShrinkageEstimator`, which extends `sklearn.base.BaseEstimator`.
Usage of these two classes is entirely analogous, and works just like any other `sklearn` estimator:
- `__init__()` parameters:
    - `base_estimator`: the estimator around which we "wrap" hierarchical shrinkage. This should be a tree-based estimator: `DecisionTreeClassifier`, `RandomForestClassifier`, ... (analogous for `Regressor`s)
    - `shrink_mode`: 4 options:
        - `"hs"`: classical Hierarchical Shrinkage (from Agarwal et al. 2022)
        - `"hs_entropy"`: Augmented Hierarchical Shrinkage with added entropy term in the numerator of the fraction.
        - `"hs_entropy_2"`: Augmented Hierarchical Shrinkage with added entropy term outside of the fraction.
        - `"hs_log_cardinality"`: Augmented Hierarchical Shrinkage with log of cardinality term in numerator of the fraction.
    - `lmb`: lambda hyperparameter
    - `random_state`: random state for reproducibility
- Other functions: `fit(X, y)`, `predict(X)`, `predict_proba(X)`, `score(X, y)` work just like with any other `sklearn` estimator.

## Notebooks
[aughs_cv_titanic.ipynb](aughs_cv_titanic.ipynb) demonstrates how we can tune hyperparameters for shrinkage models using cross-validation.