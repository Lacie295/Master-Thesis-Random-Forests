from sklearn.ensemble import GradientBoostingClassifier
from source.utils import file_manager
from source.learning import learning
import time


class GradientBoosting(learning.Learning):
    def __init__(self, data_set: file_manager.DataSet, b=False, **kwargs):
        super().__init__(data_set, b=b)
        self.g = GradientBoostingClassifier(**kwargs)
        self.predict = []
        self.n_builds = 0
        self.n_runs = 0
        self.avg_acc = 0
        self.avg_time = 0

    def build(self):
        t = time.time()
        self.g.fit(self.data_set.train(), self.data_set.train_classes())
        run_t = time.time() - t
        self.avg_time = (self.avg_time * self.n_builds + run_t) / (self.n_builds + 1)
        self.n_builds += 1

    def run(self):
        self.predict = self.g.predict(self.data_set.test())
        d = self.data_set.test_classes()
        count = 0
        for i in range(len(self.predict)):
            if self.predict[i] == d[i]:
                count += 1
        acc = count / len(self.predict)
        self.avg_acc = (self.avg_acc * (self.n_runs + 1) + acc) / (self.n_runs + 1)
        self.n_runs += 1
        print("Gradient boosting prediction rate on " + self.data_set.file + ": " + str(self.avg_acc))
        print("Build time on " + self.data_set.file + ": " + str(self.avg_time) + "s")

    def write_to_file(self):
        pass

    def read_from_file(self):
        return False
