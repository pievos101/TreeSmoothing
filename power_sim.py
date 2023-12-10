# power sim

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
import numpy as np
from tqdm import trange
from sklearn.model_selection import GridSearchCV
from treesmoothing import ShrinkageClassifier

def simulate_data(n_samples: int, relevance: float):
    X = np.zeros((n_samples, 5))
    X[:, 0] = np.random.normal(0, 1, n_samples)
    n_categories = [2, 4, 10, 20]
    for i in range(1, 5):
        X[:, i] = np.random.choice(
            a=n_categories[i - 1],
            size=n_samples,
            p=np.ones(n_categories[i - 1]) / n_categories[i - 1],
        )
    y = np.zeros(n_samples)
    y[X[:, 1] == 0] = np.random.binomial(1, 0.5 - relevance, np.sum(X[:, 1] == 0))
    y[X[:, 1] == 1] = np.random.binomial(1, 0.5 + relevance, np.sum(X[:, 1] == 1))
    return X, y

sc = "roc_auc"
ntrees = 100
relevances = [0.0, 0.05, 0.1, 0.15, 0.2]
#relevances = [0.15]

for rel in relevances:

    iterations = np.arange(0, 30, 1)

   
    X, y = simulate_data(500, rel)
    scores = {}
    scores["vanilla"] = []
    scores["hs"] = []
    scores["beta"] = []

    for xx in iterations:

        # vanilla
        print("Vanilla Mode")
        shrink_mode="vanilla"
        #scores[shrink_mode] = []
        clf = RandomForestClassifier(n_estimators=ntrees) #DecisionTreeClassifier() #RandomForestClassifier(n_estimators=1) ## DecisionTreeClassifier() #
        clf.fit(X, y)
        scores[shrink_mode].append(clf.feature_importances_)    

        # hs
        print("HS Mode")
        shrink_mode="hs"
        #scores[shrink_mode] = []
        param_grid = {
        "lmb": [0.001, 0.01, 0.1, 1, 10, 25, 50, 100, 200],
        "shrink_mode": ["hs"]}

        grid_search = GridSearchCV(ShrinkageClassifier(RandomForestClassifier(n_estimators=ntrees)), param_grid, cv=5, n_jobs=-1, scoring=sc)
        
        grid_search.fit(X, y)
        best_params = grid_search.best_params_
        print(best_params)

        clf = ShrinkageClassifier(RandomForestClassifier(n_estimators=ntrees),shrink_mode=shrink_mode, lmb=best_params.get('lmb'))
        clf.fit(X, y)
        scores[shrink_mode].append(clf.estimator_.feature_importances_)    

        # beta
        print("Beta Shrinkage")
        shrink_mode="beta"
        #scores[shrink_mode] = []
        param_grid = {
        "alpha": [8000, 5000, 4000, 2000, 1000, 800, 500, 100, 50, 30, 10, 1],
        "beta": [8000, 5000, 4000, 2000, 1000, 800, 500, 100, 50, 30, 10, 1],
        "shrink_mode": ["beta"]}

        grid_search = GridSearchCV(ShrinkageClassifier(RandomForestClassifier(n_estimators=ntrees)), param_grid, cv=5, n_jobs=-1, scoring=sc)
        grid_search.fit(X, y)

        best_params = grid_search.best_params_
        print(best_params)
        clf = ShrinkageClassifier(RandomForestClassifier(n_estimators=ntrees),shrink_mode=shrink_mode, alpha=best_params.get('alpha'), beta=best_params.get('beta'))
        clf.fit(X, y)
        scores[shrink_mode].append(clf.estimator_.feature_importances_)    
        
        print(scores)
            
    RES = np.vstack([scores['vanilla'],scores['hs'],scores['beta']])
    print(RES)

    np.savetxt(str(rel),RES, delimiter='\t')

