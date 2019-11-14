from sklearn.ensemble import RandomForestClassifier
from source.utils import file_manager
from source.learning import learning


class RandomForest(learning.Learning):
    def __init__(self, data_set: file_manager.DataSet, b=False, **kwargs):
        super().__init__(data_set, b=b)
        self.r = RandomForestClassifier()
        self.predict = []

    def build(self):
        self.r.fit(self.data_set.train(), self.data_set.train_classes())

    def run(self):
        self.predict = self.r.predict(self.data_set.test())

    def write_to_file(self):
        d = self.data_set.test_classes()
        count = 0
        for i in range(len(self.predict)):
            if self.predict[i] == d[i]:
                count += 1
        print("Random Forest prediction rate on " + self.data_set.file + ": " + str(count / len(self.predict)))

    def read_from_file(self):
        return False
