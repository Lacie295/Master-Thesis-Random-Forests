from sklearn import tree
from source.utils import file_manager
from source.learning import learning

NAME = "Decision Tree"
FILE = "decision_tree"


class DecisionTree(learning.Learning):
    def __init__(self, data_set: file_manager.DataSet, b=False, **kwargs):
        super().__init__(data_set, b=b, **kwargs)
        self.t = tree.DecisionTreeClassifier(**kwargs)
        self.FILE = FILE
        self.NAME = NAME

    def build(self):
        super().build()

    def run(self):
        super().run()

    def write_to_file(self):
        super().write_to_file()

    def read_from_file(self):
        super().read_from_file()
