from sklearn.ensemble import RandomForestClassifier
from source.utils import file_manager
from source.learning import learning
import time


class RandomForest(learning.Learning):
    def __init__(self, data_set: file_manager.DataSet, b=False, **kwargs):
        super().__init__(data_set, b=b)
        self.r = RandomForestClassifier(**kwargs)
        self.predict = []
        self.n_builds = 0
        self.n_runs = 0
        self.avg_acc = 0
        self.avg_time = 0

    def build(self):
        t = time.time()
        self.r.fit(self.data_set.train(), self.data_set.train_classes())
        run_t = time.time() - t
        self.avg_time = (self.avg_time * self.n_builds + run_t) / (self.n_builds + 1)
        self.n_builds += 1

    def run(self):
        self.predict = self.r.predict(self.data_set.test())
        d = self.data_set.test_classes()
        count = 0
        for i in range(len(self.predict)):
            if self.predict[i] == d[i]:
                count += 1
        acc = count / len(self.predict)
        self.avg_acc = (self.avg_acc * (self.n_runs + 1) + acc) / (self.n_runs + 1)
        self.n_runs += 1
        print("Random forest prediction rate on " + self.data_set.file + ": " + str(self.avg_acc))
        print("Build time on " + self.data_set.file + ": " + str(self.avg_time) + "s")

    def write_to_file(self):
        file_manager.write_to_file("results/random_forests.json", self.data_set.file,
                                   {"time": self.avg_time, "acc": self.avg_acc})

    def read_from_file(self):
        return False
