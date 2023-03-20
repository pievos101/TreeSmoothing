from imodels.util.data_util import get_clean_dataset
import numpy as np
from beta import ShrinkageClassifier
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt


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


lmbs = np.arange(0, 100, 2)
for ds_name, id, source in clf_datasets:
    X, y, feature_names = get_clean_dataset(id, data_source=source)
    scores = {}
    for shrink_mode in ["hs"]:
        scores[shrink_mode] = []
        for lmb in lmbs:
            clf = ShrinkageClassifier(shrink_mode=shrink_mode, lmb=lmb)
            scores[shrink_mode].append(
                cross_val_score(clf, X, y, cv=10, n_jobs=-1,
                                scoring="balanced_accuracy").mean())        
    for key in scores:
        plt.plot(lmbs, scores[key], label=key)
        
    plt.legend()
    plt.xlabel("$\lambda$")
    plt.ylabel("Balanced accuracy")
    plt.title(ds_name)
    plt.show()