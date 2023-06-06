from imodels.util.data_util import get_clean_dataset
import numpy as np
from beta import ShrinkageClassifier
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier
import sys

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
     ("german", "german", "pmlb")
]

# ionosphere --> bad performance for beta

# scoring
sc = "balanced_accuracy"
#sc = "roc_auc"

#ntrees = 10

for ntrees in [1, 2, 5, 10, 50, 100]:
    iterations = np.arange(0, 20, 1)

    for ds_name, id, source in clf_datasets:
        X, y, feature_names = get_clean_dataset(id, data_source=source)
        scores = {}
        print(ds_name)
        #for shrink_mode in ["hs", "hs_entropy", "hs_entropy_2", "hs_log_cardinality"]:
        #    scores[shrink_mode] = []
        #    for lmb in lmbs:
        #        clf = ShrinkageClassifier(shrink_mode=shrink_mode, lmb=lmb)
        #        scores[shrink_mode].append(
        #            cross_val_score(clf, X, y, cv=10, n_jobs=-1,
        #                            scoring="balanced_accuracy").mean())        

        scores["vanilla"] = []
        scores["hs"] = []
        scores["beta"] = []

        for xx in iterations:

            # vanilla
            print("Vanilla Mode")
            shrink_mode="vanilla"
            #scores[shrink_mode] = []
            clf = RandomForestClassifier(n_estimators=ntrees) #DecisionTreeClassifier() #RandomForestClassifier(n_estimators=1) ## DecisionTreeClassifier() #
            scores[shrink_mode].append(cross_val_score(clf, X, y, cv=5, n_jobs=-1, scoring=sc).mean())    

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
            #print(clf)
            scores[shrink_mode].append(cross_val_score(clf, X, y, cv=5, n_jobs=-1, scoring=sc).mean())    

            # beta
            print("Beta Shrinkage")
            shrink_mode="beta"
            #scores[shrink_mode] = []
            param_grid = {
            "alpha": [2000, 1500, 1000, 800, 500, 100, 50, 30, 10, 1],
            "beta": [2000, 1500, 1000, 800, 500, 100, 50, 30, 10, 1],
            "shrink_mode": ["beta"]}

            grid_search = GridSearchCV(ShrinkageClassifier(RandomForestClassifier(n_estimators=ntrees)), param_grid, cv=5, n_jobs=-1, scoring=sc)
            grid_search.fit(X, y)

            best_params = grid_search.best_params_
            print(best_params)
            clf = ShrinkageClassifier(RandomForestClassifier(n_estimators=ntrees),shrink_mode=shrink_mode, alpha=best_params.get('alpha'), beta=best_params.get('beta'))
            #print(clf)
            scores[shrink_mode].append(cross_val_score(clf, X, y, cv=5, n_jobs=-1, scoring=sc).mean())    
            
            print(scores)
            #for key in scores:
            #    #plt.plot(lmbs, scores[key], label=key)
            #    plt.boxplot(scores[key], labels=key)

    RES = np.vstack([scores['vanilla'],scores['hs'],scores['beta']])
    print(RES)

    np.savetxt(str(ntrees),RES, delimiter='\t')


