from sklearn.ensemble import GradientBoostingClassifier
from source.utils import file_manager
from source.learning import learning


class GradientBoosting(learning.Learning):
    def __init__(self, data_set: file_manager.DataSet, b=False, **kwargs):
        super().__init__(data_set, b=b)
        self.g = GradientBoostingClassifier(kwargs)
        self.predict = []

    def build(self):
        self.g.fit(self.data_set.train(), self.data_set.train_classes())

    def run(self):
        self.predict = self.g.predict(self.data_set.test())

    def write_to_file(self):
        print(self.predict)
        print(asarray(self.data_set.test_classes()))

    def read_from_file(self):
        return False
