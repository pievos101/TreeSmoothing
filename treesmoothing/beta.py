"""
:authors: Bastian Pfeifer and Arne Gevaert 

"""

from abc import abstractmethod
from copy import deepcopy
from typing import Tuple, List

import numpy as np
import scipy
from scipy.stats import beta as BETA
from numpy import typing as npt
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin, clone
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.validation import check_X_y, check_is_fitted
from sklearn.utils.estimator_checks import check_estimator


import scipy.stats

def make_beta(alpha, beta):
    """Makes a beta object."""
    dist = scipy.stats.beta(alpha, beta)
    dist.alpha = alpha
    dist.beta = beta
    return dist

def _check_fit_arguments(X, y, feature_names) -> Tuple[npt.NDArray, npt.NDArray,
                                                       List[str]]:
    if feature_names is None:
        if hasattr(X, "columns"):
            feature_names = X.columns
        else:
            X, y = check_X_y(X, y)
            feature_names = [f"feature_{i}" for i in range(X.shape[1])]
    else:
        assert len(feature_names) == X.shape[1],\
            "Number of feature names must match number of features"
    X, y = check_X_y(X, y)
    return X, y, feature_names


def _shrink_tree_rec(dt, shrink_mode, lmb=0, 
                     alpha=1, beta=1,
                     alpha_p=1, beta_p =1, 
                     X_train=None,
                     X_train_parent=None,
                     node=0, parent_node=None, parent_val=None, cum_sum=None):
    """
    Go through the tree and shrink contributions recursively
    Don't call this function directly, use shrink_forest or shrink_tree
    """

    left = dt.tree_.children_left[node]
    right = dt.tree_.children_right[node]
    feature = dt.tree_.feature[node]
    threshold = dt.tree_.threshold[node]
    parent_num_samples = dt.tree_.n_node_samples[parent_node]
    node_num_samples = dt.tree_.n_node_samples[node]
    parent_feature = dt.tree_.feature[parent_node]

    if isinstance(dt, RegressorMixin):
        value = dt.tree_.value[node, :, :]
    else:
        # Normalize to probability vector
        if shrink_mode =="beta":
            value = deepcopy(dt.tree_.value[node, :, :])  
            #print(value)
        else:
            value = deepcopy(dt.tree_.value[node, :, :]/ dt.tree_.weighted_n_node_samples[node])
        
    # cum_sum contains the value of the telescopic sum
    # If root: initialize cum_sum to the value of the root node
    if parent_node is None:
        cum_sum = value
        alpha_p = alpha
        beta_p = beta
        alpha = alpha + value[0][0] 
        beta = beta + value[0][1]
    else:
        # If not root: update cum_sum based on the value of the current node and the parent node
        reg = 1
        if shrink_mode == "hs":
            # Classic hierarchical shrinkage
            reg = 1 + (lmb / parent_num_samples)
        else:
            # parent_split_feature = X_train_parent[:, parent_feature]
            if shrink_mode =="beta":
                alpha = alpha + value[0][0]
                beta = beta + value[0][1]
        cum_sum += (value - parent_val) / reg
        
    # Set the value of the node to the value of the telescopic sum
    assert not np.isnan(cum_sum).any(), "Cumulative sum is NaN"
    dt.tree_.value[node, :, :] = cum_sum

    # HS: updated impurity
    if shrink_mode == "hs":
        dt.tree_.impurity[node] = 1 - np.sum(np.power(cum_sum, 2))

    if shrink_mode == 'beta':
        dt.tree_.value[node, :, :] = [alpha, beta]
        
        p1 = value[0][0]/(alpha+beta)
        p2 = value[0][1]/(alpha+beta)
        prob = [p1, p2]
        dt.tree_.impurity[node] = 1 - np.sum(np.power(prob, 2))
        # -------------------------------------------

    # Update the impurity of the node
    assert not np.isnan(dt.tree_.impurity[node]), "Impurity is NaN"
    
    # If not leaf: recurse
    if not (left == -1 and right == -1):
        X_train_left = deepcopy(X_train[X_train[:, feature] <= threshold])
        X_train_right = deepcopy(X_train[X_train[:, feature] > threshold])
        _shrink_tree_rec(dt, shrink_mode, lmb, deepcopy(alpha), deepcopy(beta), 
                            deepcopy(alpha_p), deepcopy(beta_p), X_train_left, X_train, left,
                            node, value, deepcopy(cum_sum))
        _shrink_tree_rec(dt, shrink_mode, lmb, deepcopy(alpha), deepcopy(beta), 
                            deepcopy(alpha_p), deepcopy(beta_p), X_train_right, X_train, right, 
                            node, value, deepcopy(cum_sum))
    else:
        if shrink_mode == 'beta':
            dt.tree_.value[node, :, :] = [alpha/(alpha + beta), beta/(alpha + beta)]
            dt.tree_.n_node_samples[node] = alpha + beta

class ShrinkageEstimator(BaseEstimator):
    def __init__(self, base_estimator: BaseEstimator = None,
                 shrink_mode: str = "hs", lmb: float = 1, alpha: float=1, beta: float=1,
                 alpha_p: float=1, beta_p: float=1,
                 random_state=None):
        self.base_estimator = base_estimator
        self.shrink_mode = shrink_mode
        self.lmb = lmb
        self.random_state = random_state
        self.alpha = alpha
        self.beta = beta
        self.alpha_p = alpha_p
        self.beta_p = beta_p

    
    @abstractmethod
    def get_default_estimator(self):
        raise NotImplemented
    
    def fit(self, X, y, **kwargs):
        X, y = self._validate_arguments(X, y, kwargs.pop("feature_names", None))

        if self.base_estimator is not None:    
            self.estimator_ = clone(self.base_estimator)
            #print("Yes, is set")
        else:
            self.estimator_ = self.get_default_estimator()

        self.estimator_.set_params(random_state=self.random_state)
        self.estimator_.fit(X, y, **kwargs)

        self.shrink(X)

        return self

    def shrink(self, X):
        if hasattr(self.estimator_, "estimators_"):  # Random Forest
            for estimator in self.estimator_.estimators_:
                #print("Its a Random Forest")
                #print(self.alpha)
                _shrink_tree_rec(estimator, self.shrink_mode, self.lmb, self.alpha, self.beta, self.alpha_p, self.beta_p, X)
        else:  # Single tree
            _shrink_tree_rec(self.estimator_, self.shrink_mode, self.lmb, self.alpha, self.beta, self.alpha_p, self.beta_p, X)

    def _validate_arguments(self, X, y, feature_names):
        if self.shrink_mode not in ["hs","beta"]:
            raise ValueError("Invalid choice for shrink_mode")
        X, y, feature_names = _check_fit_arguments(
            X, y, feature_names=feature_names)
        self.n_features_in_ = X.shape[1]
        self.feature_names_in_ = feature_names
        return X, y

    def predict(self, X, *args, **kwargs):
        check_is_fitted(self)
        return self.estimator_.predict(X, *args, **kwargs)

    def predict_proba(self, X, *args, **kwargs):
        check_is_fitted(self)
        return self.estimator_.predict_proba(X, *args, **kwargs)


    def score(self, X, y, *args, **kwargs):
        check_is_fitted(self)
        return self.estimator_.score(X, y, *args, **kwargs)


class ShrinkageClassifier(ShrinkageEstimator, ClassifierMixin):
    def get_default_estimator(self):
        return RandomForestClassifier(n_estimators=10) # DecisionTreeClassifier()# ### # # #

    def fit(self, X, y, **kwargs):
        super().fit(X, y, **kwargs)
        self.classes_ = self.estimator_.classes_
        return self
    
    def predict_proba(self, X, *args, **kwargs):
        check_is_fitted(self)
        return self.estimator_.predict_proba(X, *args, **kwargs)


class ShrinkageRegressor(ShrinkageEstimator, RegressorMixin):
    def get_default_estimator(self):
        return DecisionTreeRegressor()


if __name__ == "__main__":
    check_estimator(ShrinkageClassifier())
    check_estimator(ShrinkageRegressor())