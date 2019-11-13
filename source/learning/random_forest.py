from sklearn.ensemble import RandomForestClassifier
from source.utils import file_manager
from source.learning import learning


class RandomForest(learning.Learning):
    def __init__(self, data_set: file_manager.DataSet, b=False, **kwargs):
        super().__init__(data_set, b=b)
        self.r = RandomForestClassifier(kwargs)

    def build(self):
        self.r.fit(self.data_set.data, self.data_set.classes)

    def run(self, data_set: file_manager.DataSet):
        self.r.predict(data_set.data)

    def write_to_file(self):
        pass

    def read_from_file(self):
        return False
