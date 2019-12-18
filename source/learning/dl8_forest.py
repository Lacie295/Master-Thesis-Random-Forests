from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y
from sklearn.utils import resample
import numpy as np
from dl85 import ODTClassifier
import sys, os

from source.utils import file_manager
from source.learning import learning

NAME = "DL8Forest"
FILE = "dl8forest"


class DL8Forest(learning.Learning):
    def __init__(self, data_set: file_manager.DataSet, percent=0.5, b=False, **kwargs):
        super().__init__(data_set, percent=percent, b=b)
        self.t = Forest(**kwargs)
        self.FILE = FILE
        self.NAME = NAME

    def build(self):
        super().build()

    def run(self):
        super().run()

    def write_to_file(self):
        super().write_to_file()

    def read_from_file(self):
        super().read_from_file()


class Forest(BaseEstimator, ClassifierMixin):
    def __init__(self,
                 n_estimators=10,
                 tree_class=ODTClassifier,
                 n_samples=10,
                 sampling_type="%",
                 **kwargs):
        self.estimators = []
        self.n_estimators = n_estimators
        self.tree_class = tree_class
        self.n_samples = n_samples
        self.sampling_type = sampling_type
        self.kwargs = kwargs
        self.is_fitted = False

    def fit(self, X, y):
        check_X_y(X, y)
        sample_size = round(self.n_samples/100 * len(X)) if self.sampling_type == "%" else self.n_samples
        for i in range(self.n_estimators):
            tree = self.tree_class(**self.kwargs)
            self.estimators.append(tree)
            sample, classes = resample(X, y, n_samples=sample_size)
            tree.fit(sample, classes)
        self.is_fitted = True
        return self

    def predict(self, X):
        lst = np.array([t.predict(X) for t in self.estimators])
        return [np.argmax(np.bincount(lst[:, i])) for i in range(len(lst[0]))]

    def check_is_fitted(self):
        return self.is_fitted
