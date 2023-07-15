<p align="center">
<img src="https://github.com/pievos101/TreeSmoothing/blob/main/blurredForest.jpg" width="400">
</p>

This above picture is from (https://www.blackandwhite.ie/mononeil/blurred-forest)

# Bayesian post-hoc regularization for random forests

## Method Description
Random Forests are powerful ensemble learning algorithms widely used in various machine learning tasks. However, they have a tendency to overfit noisy or irrelevant features, which can result in decreased generalization performance. Post-hoc regularization techniques aim to mitigate this issue by modifying the structure of the learned ensemble after its training.

Here, we propose Bayesian post-hoc regularization to leverage the reliable patterns captured by leaf nodes closer to the root, while potentially reducing the impact of more specific and potentially noisy leaf nodes deeper in the tree. This approach allows for a form of pruning that does not alter the general structure of the trees but rather adjusts the influence of leaf nodes based on their proximity to the root node. We have evaluated the performance of our method on various machine learning data sets. Our approach demonstrates competitive performance with the state-of-the-art methods and, in certain cases, surpasses them in terms of predictive accuracy and generalization.

## Code Description
All classes inherit from `ShrinkageEstimator`, which extends `sklearn.base.BaseEstimator`.
Usage of these two classes is entirely analogous, and works just like any other `sklearn` estimator:
- `__init__()` parameters:
    - `base_estimator`: the estimator around which we "wrap" hierarchical shrinkage. This should be a tree-based estimator: `DecisionTreeClassifier`, `RandomForestClassifier`, ... (analogous for `Regressor`s)
    - `shrink_mode`: 2 options:
        - `"hs"`: classical Hierarchical Shrinkage (from Agarwal et al. 2022)
        - `"beta"`: Bayesian post-hoc regularization (from Pfeifer 2023)
    - `lmb`: lambda hyperparameter
    - `alpha`: alpha hyperparameter
    - `beta`: beta hyperparameter
    - `random_state`: random state for reproducibility
- Other functions: `fit(X, y)`, `predict(X)`, `predict_proba(X)`, `score(X, y)` work just like with any other `sklearn` estimator.


## Usage

Install the Python package treesmoothing via pip

```python
pip install treesmoothing
```

and import the ShrinkageClassifier as 

```python
from treesmoothing import ShrinkageClassifier
```

or install locally import of main function from source 

```python
pip install ./treesmoothing
from treesmooting import ShrinkageClassifier
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

Example data set 

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
X, y, feature_names = get_clean_dataset('breast_cancer', data_source='imodels')

# train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)

scores = {}

scores["vanilla"] = []
scores["hs"] = []
scores["beta"] = []
```

### Vanilla Random Forest 

```python
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
```

### Hierarchical Shrinkage from Agarwal et al. 2022

```python
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
```
### Bayesian post-hoc regularization from Pfeifer 2023

```python
# beta - Bayesian post-hoc regularization #########################
print("Beta Shrinkage")
shrink_mode="beta"
###################################################################

param_grid = {
"alpha": [1500, 1000, 800, 500, 100, 50, 30, 10, 1],
"beta": [1500, 1000, 800, 500, 100, 50, 30, 10, 1],
"shrink_mode": ["beta"]}

grid_search = GridSearchCV(ShrinkageClassifier
(RandomForestClassifier(n_estimators=ntrees)), param_grid, cv=5,
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
```

Print the results

```python
print(scores)
```

## Acknowledgement 
The TreeSmoothing Python code was written by Bastian Pfeifer and Arne Gevaert. 
It is based on the Hierarchical Shrinkage implementation within the Python package imodels (https://github.com/csinva/imodels).

## Citation
If you find the Bayesian post-hoc method useful please cite

```
@misc{pfeifer2023bayesian,
      title={Bayesian post-hoc regularization of random forests}, 
      author={Bastian Pfeifer},
      year={2023},
      eprint={2306.03702},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```

### Bibtex References
```
@inproceedings{agarwal2022hierarchical,
  title={Hierarchical Shrinkage: Improving the accuracy and interpretability of tree-based models.},
  author={Agarwal, Abhineet and Tan, Yan Shuo and Ronen, Omer and Singh, Chandan and Yu, Bin},
  booktitle={International Conference on Machine Learning},
  pages={111--135},
  year={2022},
  organization={PMLR}
}

```