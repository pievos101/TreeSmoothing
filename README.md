# Bayesian post-hoc regularization for random forests

## Description
All classes inherit from `ShrinkageEstimator`, which extends `sklearn.base.BaseEstimator`.
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

## Usage
Import of main function
```python
from beta import ShrinkageClassifier
```


Other imports

```python
from imodels.util.data_util import get_clean_dataset
import numpy as np
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import roc_auc_score

import sys
```

Example

```python
clf_datasets = [
    ("breast-cancer", "breast_cancer", "imodels")
]


# scoring
#sc = "balanced_accuracy"
sc = "roc_auc"

# number of trees 
ntrees = 10

# Read in data set
X, y, feature_names = get_clean_dataset(id, data_source=source)

scores = {}
print(ds_name)

scores["vanilla"] = []
scores["hs"] = []
scores["beta"] = []

# vanilla RF ##########################################
print("Vanilla Mode")
shrink_mode="vanilla"
#######################################################

clf = RandomForestClassifier(n_estimators=ntrees) 
clf.fit(X_train, y_train)
if sc == "balanced_accuracy":
    pred_vanilla = clf.predict(X_test)
    scores[shrink_mode].append(balanced_accuracy_score(y_test, pred_vanilla))    
if sc == "roc_auc":
    pred_vanilla = clf.predict_proba(X_test)[:,1]
    scores[shrink_mode].append(roc_auc_score(y_test, pred_vanilla))    

# hs - Hierarchical Shrinkage #########################
print("HS Mode")
shrink_mode="hs"
#######################################################

param_grid = {
"lmb": [0.001, 0.01, 0.1, 1, 10, 25, 50, 100, 200],
"shrink_mode": ["hs"]}

grid_search = GridSearchCV(ShrinkageClassifier(RandomForestClassifier(n_estimators=ntrees)), 
param_grid, cv=5, n_jobs=-1, scoring=sc)

grid_search.fit(X_train, y_train)

best_params = grid_search.best_params_
print(best_params)

clf = ShrinkageClassifier(RandomForestClassifier(n_estimators=ntrees),shrink_mode=shrink_mode, 
lmb=best_params.get('lmb'))

clf.fit(X_train, y_train)
if sc == "balanced_accuracy":
    pred_hs = clf.predict(X_test)
    scores[shrink_mode].append(balanced_accuracy_score(y_test, pred_hs))      
if sc == "roc_auc":
    pred_hs = clf.predict_proba(X_test)[:,1]
    scores[shrink_mode].append(roc_auc_score(y_test, pred_hs))    

# beta - Bayesian post-hoc regularization #########################
print("Beta Shrinkage")
shrink_mode="beta"
###################################################################

param_grid = {
"alpha": [1500, 1000, 800, 500, 100, 50, 30, 10, 1],
"beta": [1500, 1000, 800, 500, 100, 50, 30, 10, 1],
"shrink_mode": ["beta"]}

grid_search = GridSearchCV(ShrinkageClassifier
(RandomForestClassifie(n_estimators=ntrees)), param_grid, cv=5,
n_jobs=-1, scoring=sc)

grid_search.fit(X, y)

best_params = grid_search.best_params_
print(best_params)
clf = ShrinkageClassifier(RandomForestClassifier(n_estimators=ntrees),shrink_mode=shrink_mode, 
alpha=best_params.get('alpha'), beta=best_params.get('beta'))
clf.fit(X_train, y_train)

if sc == "balanced_accuracy":
    pred_beta = clf.predict(X_test)
    scores[shrink_mode].append(balanced_accuracy_score(y_test, pred_beta))      
if sc == "roc_auc":
    pred_beta = clf.predict_proba(X_test)[:,1]
    scores[shrink_mode].append(roc_auc_score(y_test, pred_beta))    


RES = np.vstack([scores['vanilla'],scores['hs'],scores['beta']])
print(RES)

```