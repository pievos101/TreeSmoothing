from imodels.util.data_util import get_clean_dataset
import numpy as np
from beta import ShrinkageClassifier
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
from sklearn.metrics import RocCurveDisplay
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
import numpy as np
import matplotlib.pyplot as plt

clf_datasets = [
    ("breast-cancer", "breast_cancer", "imodels") 
]


# scoring
#sc = "balanced_accuracy"
sc = "roc_auc"

lmbs = np.arange(0, 100, 1)
for ds_name, id, source in clf_datasets:
    X, y, feature_names = get_clean_dataset(id, data_source=source)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
    scores = {}
    print(ds_name)
    
    # vanilla
    print("Vanilla")
    shrink_mode="vanilla"
    scores[shrink_mode] = []
    clf = RandomForestClassifier(n_estimators=100) ## DecisionTreeClassifier() #
    clf.fit(X_train, y_train)
    pred_vanilla = clf.predict_proba(X_test)[:,1]
    

    # hs
    print("HS")
    shrink_mode="hs"
    scores[shrink_mode] = []
    param_grid = {
    "lmb": [0.001, 0.01, 0.1, 1, 10, 20, 100],
    "shrink_mode": ["hs"]}

    grid_search = GridSearchCV(ShrinkageClassifier(), param_grid, cv=5, n_jobs=-1, scoring=None, refit=True)
    grid_search.fit(X_train, y_train)
    best_params = grid_search.best_params_
    print(best_params)

    clf = ShrinkageClassifier(shrink_mode=shrink_mode, lmb=best_params.get('lmb'))
    clf.fit(X_train, y_train)
    pred_hs = clf.predict_proba(X_test)[:,1]

# beta
print("beta")
shrink_mode="beta"
scores[shrink_mode] = []
param_grid = {
"alpha": [-1, -10, -20, -50, -100, 1, 10, 20, 50, 100],
"beta": [-1, -10, -20, -50, -100, 1, 10, 20, 50, 100],
"shrink_mode": ["beta"]}

grid_search = GridSearchCV(ShrinkageClassifier(), param_grid, cv=5, n_jobs=-1, scoring=None, refit=True)
grid_search.fit(X_train, y_train)
best_params = grid_search.best_params_
print(best_params)

clf = ShrinkageClassifier(shrink_mode=shrink_mode, alpha=best_params.get('alpha'), beta=best_params.get('beta'))
clf.fit(X_train, y_train)
pred_beta = clf.predict_proba(X_test)[:,1]

#plot
#set up plotting area
plt.figure(0).clf()

fpr, tpr, _ = metrics.roc_curve(y_test, pred_vanilla)
auc = round(metrics.roc_auc_score(y_test, pred_vanilla), 4)
plt.plot(fpr,tpr,label="Vanilla, AUC="+str(auc))

fpr, tpr, _ = metrics.roc_curve(y_test, pred_hs)
auc = round(metrics.roc_auc_score(y_test, pred_hs), 4)
plt.plot(fpr,tpr,label="HS, AUC="+str(auc))

fpr, tpr, _ = metrics.roc_curve(y_test, pred_beta)
auc = round(metrics.roc_auc_score(y_test, pred_beta), 4)
plt.plot(fpr,tpr,label="BETA, AUC="+str(auc))

#add legend
plt.legend()
plt.show()