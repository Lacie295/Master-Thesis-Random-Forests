from abc import ABC, abstractmethod
from source.utils import file_manager
import time


class Learning(ABC):
    def __init__(self, data_set: file_manager.DataSet, b=False, **kwargs):
        self.data_set = data_set
        self.b = b
        self.t = None
        self.predict = []
        self.n_builds = 0
        self.n_runs = 0
        self.avg_acc = 0
        self.avg_time = 0
        self.done = False
        self.FILE = "DEFAULT"
        self.NAME = "DEFAULT"

    @abstractmethod
    def build(self):
        if not self.done:
            t = time.time()
            self.t.fit(self.data_set.train(), self.data_set.train_classes())
            run_t = time.time() - t
            self.avg_time = (self.avg_time * self.n_builds + run_t) / (self.n_builds + 1)
            self.n_builds += 1

    @abstractmethod
    def run(self):
        if not self.done:
            self.predict = self.t.predict(self.data_set.test())
            d = self.data_set.test_classes()
            count = 0
            for i in range(len(self.predict)):
                if self.predict[i] == d[i]:
                    count += 1
            acc = count / len(self.predict)
            self.avg_acc = (self.avg_acc * self.n_runs + acc) / (self.n_runs + 1)
            self.n_runs += 1

    @abstractmethod
    def write_to_file(self):
        print(self.NAME + " prediction rate on " + self.data_set.file + ": " + str(self.avg_acc))
        print("Build time on " + self.data_set.file + ": " + str(self.avg_time) + "s")
        if not self.done:
            file_manager.write_to_db(self.FILE, self.data_set.file,
                                     {"time": self.avg_time, "acc": self.avg_acc,
                                      "train_size": len(self.data_set.train_indices),
                                      "test_size": len(self.data_set.data) - len(
                                          self.data_set.train_indices)})

    @abstractmethod
    def read_from_file(self):
        data = file_manager.read_from_db(self.FILE, self.data_set.file)
        if data:
            self.avg_acc = data["acc"]
            self.avg_time = data["time"]
            self.done = True
            print("file found!")
        else:
            print("file not found or outdated, doing calculations")
