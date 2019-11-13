from sklearn import tree
from source.utils import file_manager
from source.learning import learning


class DecisionTree(learning.Learning):
    def __init__(self, data_set: file_manager.DataSet, b=False, **kwargs):
        super().__init__(data_set, b=b)
        self.t = tree.DecisionTreeClassifier()

    def build(self):
        self.t.fit(self.data_set.data, self.data_set.classes)

    def run(self, data_set: file_manager.DataSet):
        return self.t.predict(data_set.data)

    def write_to_file(self):
        pass

    def read_from_file(self):
        return False
