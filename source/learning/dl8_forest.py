from source.utils import file_manager
from source.utils.forest_core import Forest
from source.learning import learning

NAME = "DL8Forest"
FILE = "dl8forest"


class DL8Forest(learning.Learning):
    def __init__(self, data_set: file_manager.DataSet, percent=0.5, b=False, **kwargs):
        super().__init__(data_set, percent=percent, b=b)
        self.t = Forest(**kwargs)
        self.FILE = FILE + (kwargs["max_depth"] if "depth" in kwargs else "")
        self.NAME = NAME + (kwargs["max_depth"] if "depth" in kwargs else "")
        self.depth_map = {}
        self.unanimity = []
        self.n_estimators = []

    def build(self):
        super().build()
        self.depth_map[self.n_builds - 1] = self.t.get_depth_map()

    def run(self):
        super().run()
        self.unanimity.append(self.t.get_unanimity())
        self.n_estimators.append(self.t.get_n_estimators())

    def write_to_file(self):
        super().write_to_file()

    def read_from_file(self):
        super().read_from_file()
