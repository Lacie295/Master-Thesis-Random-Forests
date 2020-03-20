from source.utils import file_manager
from source.utils.forest_core import Forest
from source.learning import learning
import time
import numpy as np

NAME = "OptDL8Forest"
FILE = "optdl8forest"


class OptDL8Forest(learning.Learning):
    def __init__(self, data_set: file_manager.DataSet, percent=0.5, b=False, **kwargs):
        super().__init__(data_set, percent=percent, b=b)
        self.t = []
        self.FILE = FILE
        self.NAME = NAME
        self.depth_map = {}
        self.unanimity = []
        self.n_estimators = []
        self.kwargs = kwargs

    def build(self):
        self.size = len(self.data_set.train)
        t = time.time()
        f = Forest(optimised=True, **self.kwargs)
        self.t.append(f)
        f.fit(self.data_set.train, self.data_set.train_classes)
        run_t = time.time() - t
        self.avg_time = (self.avg_time * self.n_builds + run_t) / (self.n_builds + 1)
        self.n_builds += 1
        self.depth_map[self.n_builds - 1] = f.get_depth_map()
        self.n_estimators.append(f.get_n_estimators())

    def run(self):
        f = self.t[self.n_builds - 1]
        self.predict = np.array(f.predict(self.data_set.test))
        d = np.array(self.data_set.test_classes)
        acc = (d == self.predict).sum() / len(d)
        self.avg_acc = (self.avg_acc * self.n_runs + acc) / (self.n_runs + 1)
        self.n_runs += 1
        self.unanimity.append(f.get_unanimity())

    def write_to_file(self):
        super().write_to_file()

    def read_from_file(self):
        super().read_from_file()

    def check_acc_with_n_trees(self, n):
        d = self.data_set.test_classes
        pred = [f.predict_first_n_trees(self.data_set.test, n) for f in self.t]
        return [sum([1 if pred[t][i] == d[i] else 0 for i in range(len(d))]) / len(d)
                for t in range(len(self.t))]

    def check_train_acc_with_n_trees(self, n):
        d = self.data_set.train_classes
        pred = [f.predict_first_n_trees(self.data_set.train, n) for f in self.t]
        return [sum([1 if pred[t][i] == d[i] else 0 for i in range(len(d))]) / len(d)
                for t in range(len(self.t))]
