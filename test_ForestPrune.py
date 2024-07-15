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
import sklearn
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import tree

import ForestPrune_utils
from ForestPrune_utils import nodes_per_layer
from ForestPrune_utils import get_node_depths
from ForestPrune_utils import total_nodes
from ForestPrune_utils import difference_array_list
from ForestPrune_utils import solve_weighted
from ForestPrune_utils import prune_polish
from ForestPrune_utils import evaluate_test_error_polished
from ForestPrune_utils import get_node_count


####
clf_datasets = [
      ("heart", "heart", "imodels")
]


ntrees=100
learning_rate = 1/ntrees

for ds_name, id, source in clf_datasets:
        X, y, feature_names = get_clean_dataset(id, data_source=source)
        scores = {}
        print(ds_name)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.10)


xTrain = X_train
yTrain = y_train
xVal = X_val
yVal = y_val
xTest = X_test
yTest = y_test

model = RandomForestClassifier(n_estimators=ntrees,max_depth=20).fit(xTrain,yTrain)

tree_list = np.array(model.estimators_)
W_array = nodes_per_layer(tree_list)
normalization = total_nodes(tree_list)

base_err = sklearn.metrics.mean_squared_error(yTest,model.predict(xTest))
val_err = sklearn.metrics.mean_squared_error(yVal,model.predict(xVal))

base_rf_nodes = 0
for tree1 in tree_list:
    base_rf_nodes = base_rf_nodes + tree1.tree_.node_count


diff_array_list = difference_array_list(xTrain,tree_list)
diff_test_array_list = difference_array_list(xTest,tree_list)
diff_val_array_list = difference_array_list(xVal,tree_list)

# til here ok, including max.depth hack

alphas = []
warm_start = []
results = []

for alpha in np.flip(np.logspace(-2,1.5,50)):
    
    #print(alpha)
    res = solve_weighted(yTrain,tree_list,diff_array_list,
                            alpha,learning_rate,W_array,normalization, warm_start = warm_start)
    
    if res == False:
        print("Error: No convergency.\n") 
        sys.exit(1)


    vars1 = res[0]
    iters = res[1]

    warm_start = vars1
    coef = prune_polish(diff_array_list,yTrain,vars1,learning_rate)
    pred = evaluate_test_error_polished(diff_val_array_list,yVal,vars1,
                                                    coef,learning_rate)
    err = np.square(np.subtract(yVal, pred)).mean()
    results.append([alpha,err,iters])
    #print(results)

results_df = pd.DataFrame(results,columns = ['alpha', 'err','iter'])


err_min = np.min(results_df['err'])
best_alpha = results_df[results_df['err']==err_min]['alpha']
best_alpha = np.array(best_alpha)[0]

vars_best,_ = solve_weighted(yTrain,tree_list,diff_array_list,best_alpha,learning_rate,
                                W_array,normalization,)

coef_best = prune_polish(diff_array_list,yTrain,vars_best,learning_rate)

pred = evaluate_test_error_polished(diff_test_array_list,yTest,vars_best,
                                                        coef_best,learning_rate)
PRED = np.abs(np.vstack((pred-0, pred-1)))
pred_final = PRED.argmin(0)

print(sklearn.metrics.balanced_accuracy_score(model.predict(xTest), yTest))
print(sklearn.metrics.balanced_accuracy_score(pred_final, yTest))




########################
list_errs = []
list_nodes = []
list_threshold = []

for i in [0.01,0.025,0.05]:
    UL = val_err + i
    best_alpha = np.max(results_df[results_df['err']<UL]['alpha'])
    if np.isnan(best_alpha):
        best_alpha = results_df['alpha'].min()
    print(best_alpha)
    
    
    vars_best,_ = solve_weighted(yTrain,tree_list,diff_array_list,best_alpha,learning_rate,
                                W_array,normalization,)

    coef_best = prune_polish(diff_array_list,yTrain,vars_best,learning_rate)

    pruned_err = evaluate_test_error_polished(diff_test_array_list,yTest,vars_best,
                                                        coef_best,learning_rate)
    pruned_rf_nodes = get_node_count(tree_list,vars_best)
    nodes_limit = pruned_rf_nodes
    prune_score = pruned_err
            