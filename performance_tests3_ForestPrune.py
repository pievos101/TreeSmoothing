from imodels.util.data_util import get_clean_dataset
import numpy as np
from treesmoothing import ShrinkageClassifier
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import roc_auc_score

import sys

import ForestPrune 

clf_datasets = [
    ("heart", "heart", "imodels"),
    ("breast-cancer", "breast_cancer", "imodels"), 
    ("haberman", "haberman", "imodels"), 
    ("ionosphere", "ionosphere", "pmlb"),
    ("diabetes-clf", "diabetes", "pmlb"),
    ("german", "german", "pmlb"),
    ("juvenile", "juvenile_clean", "imodels"),
    ("recidivism", "compas_two_year_clean", "imodels")
]

clf_datasets = [
    ("heart", "heart", "imodels"), 
    ("breast-cancer", "breast_cancer", "imodels"),
    ("haberman", "haberman", "imodels"), 
    ("ionosphere", "ionosphere", "pmlb"),
    ("diabetes-clf", "diabetes", "pmlb"),
    ("german", "german", "pmlb")
]

clf_datasets = [
    ("haberman", "haberman", "imodels") 
]


clf_datasets = [
    ("breast-cancer", "breast_cancer", "imodels")
]

clf_datasets = [
    ("heart", "heart", "imodels")
]

clf_datasets = [
     ("diabetes-clf", "diabetes", "pmlb")
]

####
clf_datasets = [
      ("heart", "heart", "imodels")
]


# scoring
#sc = "balanced_accuracy"
sc = "roc_auc"

#ntrees = 100

for ntrees in [1, 2, 5, 10, 50, 100]:
    iterations = np.arange(0, 20, 1)

    for ds_name, id, source in clf_datasets:
        X, y, feature_names = get_clean_dataset(id, data_source=source)
        scores = {}
        print(ds_name)

        
        scores["vanilla"] = []
        scores["hs"] = []
        scores["beta"] = []
            
        for xx in iterations:

            # train-test split
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)

            # vanilla
            print("Vanilla Mode")
            shrink_mode="vanilla"
            #scores[shrink_mode] = []
            clf = RandomForestClassifier(n_estimators=ntrees) 
            clf.fit(X_train, y_train)
            if sc == "balanced_accuracy":
                pred_vanilla = clf.predict(X_test)
                scores[shrink_mode].append(balanced_accuracy_score(y_test, pred_vanilla))    
            if sc == "roc_auc":
                pred_vanilla = clf.predict_proba(X_test)[:,1]
                scores[shrink_mode].append(roc_auc_score(y_test, pred_vanilla))    
            # hs
            print("HS Mode")
            shrink_mode="hs"
            #scores[shrink_mode] = []
            param_grid = {
            "lmb": [0.001, 0.01, 0.1, 1, 10, 25, 50, 100, 200],
            "shrink_mode": ["hs"]}

            grid_search = GridSearchCV(ShrinkageClassifier(RandomForestClassifier(n_estimators=ntrees)), param_grid, cv=5, n_jobs=-1, scoring=sc)
            
            grid_search.fit(X_train, y_train)
            best_params = grid_search.best_params_
            print(best_params)

            clf = ShrinkageClassifier(RandomForestClassifier(n_estimators=ntrees),shrink_mode=shrink_mode, lmb=best_params.get('lmb'))
            #print(clf)
            clf.fit(X_train, y_train)
            if sc == "balanced_accuracy":
                pred_hs = clf.predict(X_test)
                scores[shrink_mode].append(balanced_accuracy_score(y_test, pred_hs))      
            if sc == "roc_auc":
                pred_hs = clf.predict_proba(X_test)[:,1]
                scores[shrink_mode].append(roc_auc_score(y_test, pred_hs))    

            # beta
            print("Beta Shrinkage")
            shrink_mode="beta"
            #scores[shrink_mode] = []
            param_grid = {
            "alpha": [8000, 5000, 2000, 1000, 500, 100, 50, 30, 10, 1],
            "beta": [8000, 5000, 2000, 1000,  500, 100, 50, 30, 10, 1],
            "shrink_mode": ["beta"]}

            grid_search = GridSearchCV(ShrinkageClassifier(RandomForestClassifier(n_estimators=ntrees)), param_grid, cv=5, n_jobs=-1, scoring=sc)
            grid_search.fit(X_train, y_train)

            best_params = grid_search.best_params_
            print(best_params)
            clf = ShrinkageClassifier(RandomForestClassifier(n_estimators=ntrees),shrink_mode=shrink_mode, alpha=best_params.get('alpha'), beta=best_params.get('beta'))
            clf.fit(X_train, y_train)
            
            if sc == "balanced_accuracy":
                pred_beta = clf.predict(X_test)
                scores[shrink_mode].append(balanced_accuracy_score(y_test, pred_beta))      
            if sc == "roc_auc":
                pred_beta = clf.predict_proba(X_test)[:,1]
                scores[shrink_mode].append(roc_auc_score(y_test, pred_beta))    

            #ForestPrune

            
            print(scores)

    RES = np.vstack([scores['vanilla'],scores['hs'],scores['beta']])
    print(RES)

    np.savetxt(str(ntrees),RES, delimiter='\t')

