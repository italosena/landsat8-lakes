from sklearn.svm import SVC
import numpy as np
import csv

"""
SVC Class
    C=1.0
    cache_size=200
    class_weight=None
    coef0=0.0
    degree=3,
    gamma=0.0
    kernel='rbf'
    max_iter=-1
    probability=False,
    random_state=None
    shrinking=True
    tol=0.001
    verbose=False
"""

clf = SVC()
clf.fit(X, y)
clf.predict(X)
clf.score(X, y)
