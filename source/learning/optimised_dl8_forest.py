from source.utils import file_manager
from source.utils.forest_core import Forest
from source.learning import learning
import time

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
        self.predict = f.predict(self.data_set.test)
        y = self.data_set.test_classes
        acc = sum([1 if self.predict[i] == y[i] else 0 for i in range(len(y))]) / len(y)
        self.avg_acc = (self.avg_acc * self.n_runs + acc) / (self.n_runs + 1)
        self.n_runs += 1
        self.unanimity.append(f.get_unanimity())

    def write_to_file(self):
        super().write_to_file()

    def read_from_file(self):
        super().read_from_file()

    def check_acc_with_n_trees(self, n, test=True):
        y = self.data_set.test_classes if test else self.data_set.train_classes
        X = self.data_set.test if test else self.data_set.train
        pred = [self.t[i].predict_first_n_trees(X, n, slot=0 if test else 1) if n <= self.t[i].n_estimators else -1
                for i in range(len(self.t))]
        return [sum([1 if pred[t][i] == y[i] else 0 for i in range(len(y))]) / len(y)
                for t in range(len(self.t))]
