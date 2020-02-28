from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y
from sklearn.utils import resample
import numpy as np
from dl85 import DL85Classifier


class Forest(BaseEstimator, ClassifierMixin):
    def __init__(self,
                 n_estimators=10,
                 tree_class=DL85Classifier,
                 method="random",
                 attributes="all",
                 n_samples=25,
                 sampling_type="%",
                 **kwargs):
        self.estimators = []
        self.n_estimators = n_estimators
        self.tree_class = tree_class
        self.n_samples = n_samples
        self.sampling_type = sampling_type
        self.kwargs = kwargs
        self.is_fitted = False
        self.unanimity = None
        self.method = method
        self.attributes = attributes

    def fit(self, X, y):
        check_X_y(X, y)
        self.estimators = []

        sample_size = round(self.n_samples / 100 * len(X)) if self.sampling_type == "%" else self.n_samples
        column_size = round(75 / 100 * len(X[0]))

        for i in range(self.n_estimators):
            tree = self.tree_class(**self.kwargs)
            self.estimators.append(tree)

            if self.method == "random":
                sample, classes = resample(X, y, n_samples=sample_size)
            else:
                sample, classes = X, y

            if self.attributes == "random":
                # TODO: fix
                sample = resample(*sample, n_samples=column_size, replace=False)

            tree.fit(sample, classes)

            if self.attributes == "progressive":
                root = tree.tree_

                attr = root['feat']
                for i in range(len(X)):
                    X[i][attr] = 1

            if self.attributes == "random_progressive":
                root = tree.tree_
                pred = None
                while 'class' not in root:
                    c = np.random.randint(0, 2)
                    pred = root
                    if c == 0:
                        break
                    elif c == 1:
                        root = root['left']
                    else:
                        root = root['right']

                attr = pred['feat']
                for i in range(len(X)):
                    X[i][attr] = 1

        self.is_fitted = True
        return self

    def predict(self, X):
        lst = np.array([t.predict(X) for t in self.estimators])
        pred = [np.argmax(np.bincount(lst[:, i])) for i in range(len(lst[0]))]
        self.unanimity = [np.count_nonzero(lst[:, i] == pred[i]) for i in range(len(lst[0]))]
        return pred

    def check_is_fitted(self):
        return self.is_fitted

    def get_depth_map(self):
        if self.tree_class == DL85Classifier:
            depth_map = {}
            for t in self.estimators:
                tree = t.tree_

                def build_depth_map(curr_tree, n=1):
                    if n not in depth_map:
                        depth_map[n] = {}
                    d = depth_map[n]
                    if 'class' not in curr_tree:
                        f = curr_tree['feat']
                        if f not in d:
                            d[f] = 1
                        else:
                            d[f] += 1
                        build_depth_map(curr_tree['left'], n + 1)
                        build_depth_map(curr_tree['right'], n + 1)

                build_depth_map(tree)
            return depth_map
        else:
            return None

    def get_unanimity(self):
        return self.unanimity

    def get_n_estimators(self):
        return self.n_estimators
