from sklearn.ensemble import RandomForestClassifier
from source.utils import file_manager
from source.learning import learning


class RandomForest(learning.Learning):
    def __init__(self, data_set: file_manager.DataSet, b=False, **kwargs):
        super().__init__(data_set, b=b)
        self.r = RandomForestClassifier(kwargs)

    def build(self):
        pass

    def run(self, data_set):
        pass

    def write_to_file(self):
        pass

    def read_from_file(self):
        return False
