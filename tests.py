from beta import ShrinkageClassifier
from sklearn.model_selection import GridSearchCV
from data import get_titanic


X_train, X_test, y_train, y_test = get_titanic("../raw_data/titanic/titanic_train.csv")
clf = ShrinkageClassifier(shrink_mode='beta', alpha=1,beta=1)
clf.fit(X_train.to_numpy(), y_train.to_numpy())


