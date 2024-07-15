
import sklearn

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import tree

import ForestPrune
from ForestPrune import nodes_per_layer
from ForestPrune import get_node_depths
from ForestPrune import total_nodes
from ForestPrune import difference_array_list

def run_FP(X_train, y_train, ntrees):

    xTrain = X_train
    yTrain = y_train
    xTest = X_test
    yTest = y_test

    model = RandomForestClassifier(n_estimators=ntrees,max_depth=20).fit(xTrain,yTrain)

    tree_list = np.array(model.estimators_)
    W_array = nodes_per_layer(tree_list)
    normalization = total_nodes(tree_list)

    base_err = sklearn.metrics.mean_squared_error(yTest,model.predict(xTest))
    #val_err = sklearn.metrics.mean_squared_error(yVal,model.predict(xVal))

    base_rf_nodes = 0
    for tree1 in tree_list:
        base_rf_nodes = base_rf_nodes + tree1.tree_.node_count


    diff_array_list = difference_array_list(xTrain,tree_list)
    diff_test_array_list = difference_array_list(xTest,tree_list)
    diff_val_array_list = difference_array_list(xVal,tree_list)

    alphas = []
    warm_start = []

    results = []

    for alpha in np.flip(np.logspace(-2,1.5,50)):
        
        vars1, iters = solve_weighted(yTrain,tree_list,diff_array_list,
                                alpha,learning_rate,W_array,normalization, warm_start = warm_start)
        warm_start = vars1
        coef = prune_polish(diff_array_list,yTrain,vars1,learning_rate)
        err = evaluate_test_error_polished(diff_val_array_list,yVal.values,vars1,
                                                        coef,learning_rate)
        results.append([alpha,err,iters])

    results_df = pd.DataFrame(results,columns = ['alpha', 'err','iter'])


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

        pruned_err = evaluate_test_error_polished(diff_test_array_list,yTest.values,vars_best,
                                                            coef_best,learning_rate)