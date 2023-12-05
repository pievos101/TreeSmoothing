import sys
#sys.path.append("../arne/")

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
import numpy as np
from treesmoothing import ShrinkageClassifier
from treesmoothing import importance
from tqdm import trange
import joblib
import pandas as pd
from shap import TreeExplainer
from sklearn.datasets import make_classification
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import roc_auc_score

FI_no_hsc = list()
FI_hsc = list()
FI_beta = list()

for xx in range(0,30):
    print(xx)
    X, y = make_classification(n_samples=500,
                                n_features=100, 
                                shuffle=False, 
                                n_informative=20,
                                n_redundant=2,
                                class_sep=3,
                                #flip_y = 0.3)
                                weights=[0.95])

    ntrees = 500
    #sc = "roc_auc"
    sc = "balanced_accuracy"

    # Compute importances for classical RF/DT
    clf = RandomForestClassifier(n_estimators=ntrees).fit(X, y)
    FI_no_hsc.append(clf.feature_importances_)
    #np.savetxt("FI_no_hsc",FI_no_hsc, delimiter='\t')

    # SHAP
    #explainer = TreeExplainer(clf, X, check_additivity=False)
    #shap_values = np.array(explainer.shap_values(X))
    #shap_values = abs(shap_values[0,:,:])+abs(shap_values[1,:,:])

    #importances = []
    #for i in range(shap_values.shape[1]):
    #    importances.append(np.mean(np.abs(shap_values[:, i])))         

    #np.savetxt("FI_no_hsc_SHAP",importances, delimiter='\t')


    ###################################
    # Hierarchical Shrinkage
    ###################################

    lambdas = [0., 0.1, 1.0, 10.0, 25.0, 50.0, 100.0]

    shrink_mode = "hs"

    param_grid = {
        "lmb": [0.001, 0.01, 0.1, 1, 10, 25, 50, 100, 200],
        "shrink_mode": ["hs"]
    }

    grid_search = GridSearchCV(ShrinkageClassifier(RandomForestClassifier(n_estimators=ntrees)), param_grid, cv=5, n_jobs=-1, scoring=sc)

    grid_search.fit(X, y)
    best_params = grid_search.best_params_
    #print(best_params)

    clf = ShrinkageClassifier(RandomForestClassifier(n_estimators=ntrees),shrink_mode=shrink_mode, lmb=best_params.get('lmb'))
    #print(clf)
    clf.fit(X, y)
    FI_hsc.append(clf.estimator_.feature_importances_)
    #np.savetxt("FI_hsc",FI_hsc, delimiter='\t')

    # SHAP
    #explainer = TreeExplainer(hsc.estimator_, X, check_additivity=False)
    #shap_values = np.array(explainer.shap_values(X))
    #shap_values = abs(shap_values[0,:,:])+abs(shap_values[1,:,:])
    #importances = []
    #for i in range(shap_values.shape[1]):
    #    importances.append(np.mean(np.abs(shap_values[:, i])))         

    #np.savetxt("FI_hsc_SHAP",importances, delimiter='\t')

    #########################################
    # BETA-based Hierarchical Shrinkage
    #########################################

    shrink_mode="beta"
    #scores[shrink_mode] = []
    param_grid = {
    "alpha": [2000, 1000, 700, 500, 200, 100, 50, 10, 1],
    "beta": [2000, 1000, 700, 500, 200, 100, 50, 10, 1],
    "shrink_mode": ["beta"]}

    grid_search = GridSearchCV(ShrinkageClassifier(RandomForestClassifier(n_estimators=ntrees)), param_grid, cv=5, n_jobs=-1, scoring=sc)
    grid_search.fit(X, y)

    best_params = grid_search.best_params_
    print(best_params)

    clf2 = ShrinkageClassifier(RandomForestClassifier(n_estimators=ntrees),shrink_mode=shrink_mode, alpha=best_params.get('alpha'), beta=best_params.get('beta'))
    clf2.fit(X, y)

    #FI_beta = clf2.estimator_.feature_importances_
    FI_beta.append(clf2.estimator_.feature_importances_)

    #FI_beta.append(importance(clf2))

    #np.savetxt("FI_beta",FI_beta, delimiter='\t')

    # SHAP
    #explainer = TreeExplainer(ehsc.estimator_, X, check_additivity=False)
    #shap_values = np.array(explainer.shap_values(X))
    #shap_values = abs(shap_values[0,:,:])+abs(shap_values[1,:,:])
    #importances = []
    #for i in range(shap_values.shape[1]):
    #    importances.append(np.mean(np.abs(shap_values[:, i])))         

    #np.savetxt("FI_ehsc_SHAP",importances, delimiter='\t')

    np.savetxt("FI_no_hsc",FI_no_hsc, delimiter='\t')
    np.savetxt("FI_hsc",FI_hsc, delimiter='\t')
    np.savetxt("FI_beta",FI_beta, delimiter='\t')