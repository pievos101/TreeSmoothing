
def importance(model):
    #
    clf2 = model
    IMP = clf2.estimator_.estimators_[0].tree_.compute_feature_importances()
    IMP[IMP<0] = 0

    for xx in range(1,len(clf2.estimator_.estimators_)-1):

        imp = clf2.estimator_.estimators_[xx].tree_.compute_feature_importances()
        imp[imp<0] = 0
        IMP = IMP + imp
        
    # normalize
    IMP = IMP/len(clf2.estimator_.estimators_)
    return IMP
