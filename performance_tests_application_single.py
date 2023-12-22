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
from imodels.util.data_util import get_clean_dataset

#X = pd.read_csv("OMICS.txt", sep='\t')
#X = np.array(X)
#y = pd.read_csv("omics_target.txt", sep='\t') 
#y = np.array(y).flatten()


clf_datasets = [
    ("breast-cancer", "breast_cancer", "imodels")
]

for ds_name, id, source in clf_datasets:
    X, y, feature_names = get_clean_dataset(id, data_source=source)

ntrees = 10
#sc = "balanced_accuracy"
sc = "roc_auc"
PERF = []
PERF_w = []


for iter in range(0,20):

    # train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)
   
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

    a_p=best_params.get('alpha')
    b_p=best_params.get('beta')
    a_r = len(y_train) - y_train.sum()
    b_r = y_train.sum()
    a_minus = a_p + a_r
    b_minus = b_p + b_r

    if sc == "balanced_accuracy":
        pred_beta = clf.predict(X_test)
        perf = balanced_accuracy_score(y_test, pred_beta)     
    if sc == "roc_auc":
        pred_beta = clf.predict_proba(X_test)[:,1]
        perf = roc_auc_score(y_test, pred_beta)    


    ##############################
    ENTROPY = []
    ENTROPY_ALL = []
    ENTROPY_ALL2 = []
    for xx in range(0,len(clf.estimator_.estimators_)):    
        ENTROPY = [] 
        for yy in range(0, len(y_test)):
            leaf_id = clf.estimator_.estimators_[xx].apply(X_test)[yy] # first patient
            N = clf.estimator_.estimators_[xx].tree_.n_node_samples[leaf_id]
            prob = clf.estimator_.estimators_[xx].tree_.value[leaf_id]
            
            # Get length of path
            node_indicator = clf.estimator_.estimators_[xx].decision_path(X_test)
            sample_id = yy
            # obtain ids of the nodes `sample_id` goes through, i.e., row `sample_id`
            node_index = node_indicator.indices[
                node_indicator.indptr[sample_id] : node_indicator.indptr[sample_id + 1]
            ]

            a  = (N*prob)[0][0]/len(node_index)#/len(y_train) # class 0
            b  = (N*prob)[0][1]/len(node_index)#/#len(y_train) # class 1
            
            #if a >= b:
            ENTROPY.append(BETA.pdf(a/(a+b), a, b))
            #if a < b: 
            #    ENTROPY.append(BETA.pdf(b/(a+b), a, b))
            
            #ENTROPY.append(1-BETA.var(a,b))
            #ENTROPY.append(np.abs(BETA.entropy(a,b)))
   
        ENTROPY_ALL.append((np.array(ENTROPY)).mean()) #mean()
        ENTROPY_ALL2.append((np.array(ENTROPY)))

    ENTROPY_ALL = np.array(ENTROPY_ALL)
    ENTROPY_ALL2 = np.array(ENTROPY_ALL2)
    
    print(ENTROPY_ALL2)
    #print(ENTROPY_ALL)
    #RES = np.vstack(ENTROPY_ALL)
    #np.savetxt("ENTROPY",RES, delimiter='\t')


    # Weighted Majority Vote
    if sc == "balanced_accuracy":
        PRED = []
        for xx in range(0,len(clf.estimator_.estimators_)): 
            PRED.append(clf.estimator_.estimators_[xx].predict(X_test))

        PRED = np.vstack(PRED)
        W = np.average(PRED,0, weights=ENTROPY_ALL)
        W[W>0.5] = 1
        W[W<=0.5] = 0

    # version 1
    if sc == "roc_auc":
        PRED = []
        for xx in range(0,len(clf.estimator_.estimators_)): 
            PRED.append(clf.estimator_.estimators_[xx].predict_proba(X_test)[:,1])

        PRED = np.vstack(PRED)
        W = np.average(PRED,0, weights=ENTROPY_ALL)
        
    #######################################
    # version 2 - the regularization for each sample!
    if sc == "roc_auc":
        W = []
        for xx in range(0,len(y_test)): 
            myW = ENTROPY_ALL2[:,xx]
            W.append(np.average(PRED[:,xx],0, weights=myW))
        W = np.array(W)
    #    W[W>0.5] = 1
    #    W[W<=0.5] = 0
    #######################################
        
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
    
