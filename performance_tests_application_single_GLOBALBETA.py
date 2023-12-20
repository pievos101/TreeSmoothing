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

X = pd.read_csv("OMICS.txt", sep='\t')
X = np.array(X)
y = pd.read_csv("omics_target.txt", sep='\t') 
y = np.array(y).flatten()


#clf_datasets = [
#    ("breast-cancer", "breast_cancer", "imodels")
#]

#for ds_name, id, source in clf_datasets:
#    X, y, feature_names = get_clean_dataset(id, data_source=source)

ntrees = 50
#sc = "balanced_accuracy"
sc = "roc_auc"
PERF = []
PERF_w = []


for iter in range(0,1):

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
        pred_beta1 = clf.predict(X_test)
        perf = balanced_accuracy_score(y_test, pred_beta1)     
    if sc == "roc_auc":
        pred_beta1 = clf.predict_proba(X_test)[:,1]
        perf = roc_auc_score(y_test, pred_beta1)    


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
            
            a  = (N*prob)[0][0] # class 0
            b  = (N*prob)[0][1] # class 1
            
            #if a >= b:
            #    ENTROPY.append(BETA.pdf(a/(a+b), a, b))
            #if a < b: 
            #    ENTROPY.append(BETA.pdf(b/(a+b), a, b))
    
            #ENTROPY.append(1-BETA.var(a,b))
            #ENTROPY.append(np.abs(BETA.entropy(a,b)))
            ENTROPY.append(np.array([a,b])/len(y_train))

        ENTROPY_ALL.append((np.array(ENTROPY))) 
        
    ENTROPY_ALL = np.array(ENTROPY_ALL)

    #ENTROPY_ALL2 = np.array(ENTROPY_ALL2)
    
    print(ENTROPY_ALL)
    #print(ENTROPY_ALL)
    #RES = np.vstack(ENTROPY_ALL)
    #np.savetxt("ENTROPY",RES, delimiter='\t')

    W = ENTROPY_ALL[:,:,:].sum(0)
    W = W[:,1]/(W[:,0]+W[:,1])

    if sc == "balanced_accuracy":
        pred_beta2 = W
        perf2 = balanced_accuracy_score(y_test, pred_beta2)     
    if sc == "roc_auc":
        pred_beta2 = W
        perf2 = roc_auc_score(y_test, pred_beta2)    

    PERF.append(perf)
    PERF_w.append(perf2)
    print(PERF)
    print(PERF_w)

    np.savetxt("PRED",np.vstack(pred_beta1), delimiter='\t')
    np.savetxt("PRED_w",np.vstack(pred_beta2), delimiter='\t')
    np.savetxt("y_test",np.vstack(y_test), delimiter='\t')
