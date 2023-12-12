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
import pandas as pd
from scipy.stats import beta as BETA
from sklearn.ensemble import VotingClassifier
import sys

ntrees = 50
sc = "balanced_accuracy"
#sc = "roc_auc"

X = pd.read_csv("OMICS.txt", sep='\t')
X = np.array(X)
y = pd.read_csv("omics_target.txt", sep='\t') 
y = np.array(y).flatten()

PERF = []
PERF_w = []


# train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)


for iter in range(0,20):

    # beta
    print("Beta Shrinkage")
    shrink_mode="beta"
    #scores[shrink_mode] = []
    param_grid = {
    "alpha": [500, 200, 150, 100, 50, 30, 10, 5, 1],
    "beta": [500, 200, 150, 100, 50, 30, 10, 5, 1],
    "shrink_mode": ["beta"]}

    grid_search = GridSearchCV(ShrinkageClassifier(RandomForestClassifier(n_estimators=ntrees)), param_grid, cv=5, n_jobs=-1, scoring=sc)
    grid_search.fit(X_train, y_train)

    best_params = grid_search.best_params_
    print(best_params)
    clf = ShrinkageClassifier(RandomForestClassifier(n_estimators=ntrees),shrink_mode=shrink_mode, alpha=best_params.get('alpha'), beta=best_params.get('beta'))
    clf.fit(X_train, y_train)

    if sc == "balanced_accuracy":
        pred_beta = clf.predict(X_test)
        perf = balanced_accuracy_score(y_test, pred_beta)     
    if sc == "roc_auc":
        pred_beta = clf.predict_proba(X_test)[:,1]
        perf = roc_auc_score(y_test, pred_beta)    


    ##############################
    ENTROPY = []
    ENTROPY_ALL = []
    for xx in range(0,len(clf.estimator_.estimators_)):    
        ENTROPY = [] 
        for yy in range(0, len(y_test)):
            leaf_id = clf.estimator_.estimators_[xx].apply(X_test)[yy] # first patient
            N = clf.estimator_.estimators_[xx].tree_.n_node_samples[leaf_id]
            prob = clf.estimator_.estimators_[xx].tree_.value[leaf_id]
            a  = (N*prob)[0][0]
            b  = (N*prob)[0][1]
            ENTROPY.append(BETA.pdf(a/(a+b),a,b))
        ENTROPY_ALL.append(np.array(ENTROPY).mean())

    ENTROPY_ALL = np.array(ENTROPY_ALL)

    #RES = np.vstack(ENTROPY_ALL)
    #np.savetxt("ENTROPY",RES, delimiter='\t')


    # Weighted Majority Vote
    PRED = []
    for xx in range(0,len(clf.estimator_.estimators_)): 
        PRED.append(clf.estimator_.estimators_[xx].predict(X_test))

    PRED = np.vstack(PRED)
    W = np.average(PRED,0, weights=ENTROPY_ALL)#/ENTROPY_ALL.sum())
    W[W>0.5] = 1
    W[W<=0.5] = 0

    if sc == "balanced_accuracy":
        pred_beta = W
        perf2 = balanced_accuracy_score(y_test, pred_beta)     
    if sc == "roc_auc":
        pred_beta = W
        perf2 = roc_auc_score(y_test, pred_beta)    

    PERF.append(perf)
    PERF_w.append(perf2)
    print(PERF)
    print(PERF_w)

    np.savetxt("PERF",np.vstack(PERF), delimiter='\t')
    np.savetxt("PERF_w",np.vstack(PERF_w), delimiter='\t')
    
