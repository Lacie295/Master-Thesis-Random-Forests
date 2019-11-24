from sklearn.ensemble import GradientBoostingClassifier
from source.utils import file_manager
from source.learning import learning

NAME = "Gradient Boosting"
FILE = "results/gradient_boosting.json"


class GradientBoosting(learning.Learning):
    def __init__(self, data_set: file_manager.DataSet, b=False, **kwargs):
        super().__init__(data_set, b=b)
        self.t = GradientBoostingClassifier(**kwargs)
        self.FILE = FILE
        self.NAME = NAME
        if not b:
            self.read_from_file()

    def build(self):
        super().build()

    def run(self):
        super().run()

    def write_to_file(self):
        super().write_to_file()

    def read_from_file(self):
        super().read_from_file()
