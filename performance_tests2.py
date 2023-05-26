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

#clf_datasets = [
#    ("haberman", "haberman", "imodels")
#]

# scoring
sc = "balanced_accuracy"
#sc = "roc_auc"

iterations = np.arange(0, 10, 1)
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
        clf = RandomForestClassifier(n_estimators=50) #DecisionTreeClassifier() #RandomForestClassifier(n_estimators=1) ## DecisionTreeClassifier() #
        scores[shrink_mode].append(cross_val_score(clf, X, y, cv=5, n_jobs=-1, scoring=sc).mean())    

        # hs
        print("HS Mode")
        shrink_mode="hs"
        #scores[shrink_mode] = []
        param_grid = {
        "lmb": [0.001, 0.01, 0.1, 1, 10, 25, 50, 100, 200],
        "shrink_mode": ["hs"]}

        grid_search = GridSearchCV(ShrinkageClassifier(), param_grid, cv=5, n_jobs=-1, scoring=sc)
        grid_search.fit(X, y)
        best_params = grid_search.best_params_
        print(best_params)

        clf = ShrinkageClassifier(shrink_mode=shrink_mode, lmb=best_params.get('lmb'))
        #print(clf)
        scores[shrink_mode].append(cross_val_score(clf, X, y, cv=5, n_jobs=-1, scoring=sc).mean())    

        # beta
        print("Beta Shrinkage")
        shrink_mode="beta"
        #scores[shrink_mode] = []
        param_grid = {
        "alpha": [0, -10, -50, -100, 0, 10, 50, 100],
        "beta": [0, -10, -50, -100, 0, 10, 50, 100],
        "shrink_mode": ["beta"]}

        grid_search = GridSearchCV(ShrinkageClassifier(), param_grid, cv=5, n_jobs=-1, scoring=sc)
        grid_search.fit(X, y)

        best_params = grid_search.best_params_
        print(best_params)
        clf = ShrinkageClassifier(shrink_mode=shrink_mode, alpha=best_params.get('alpha'), beta=best_params.get('beta'))
        #print(clf)
        scores[shrink_mode].append(cross_val_score(clf, X, y, cv=5, n_jobs=-1, scoring=sc).mean())    
        
        print(scores)
        #for key in scores:
        #    #plt.plot(lmbs, scores[key], label=key)
        #    plt.boxplot(scores[key], labels=key)

# plots 
import numpy as np
import matplotlib.pyplot as plt
fig, ax = plt.subplots()
data = list([scores['vanilla'], scores['hs'], 
    scores['beta']])
# basic plot
ax.boxplot(data, notch=False)
ax.set_ylim([0.5, 1])

ax.set_title(ds_name)
ax.set_xlabel('')
ax.set_ylabel(sc)
xticklabels=['vanilla','hs', 'beta']
ax.set_xticklabels(xticklabels)
plt.xticks(fontsize=7)#, rotation=45)
# add horizontal grid lines
#ax.yaxis.grid(True)

for i, d in enumerate(data):
    y = np.array(data)[i]
    x = np.random.normal(i + 1, 0.04, len(y))
    plt.scatter(x, y, s=[5])

plt.savefig(ds_name)
# show the plot
plt.show()


