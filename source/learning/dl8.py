from source.utils import file_manager
from source.learning import learning
from dl85 import DL85Classifier

NAME = "DL8"
FILE = "dl8"


class DL8(learning.Learning):
    def __init__(self, data_set: file_manager.DataSet, percent=0.5, b=False, **kwargs):
        super().__init__(data_set, percent=percent, b=b)
        self.t = DL85Classifier(**kwargs)
        self.FILE = FILE + (kwargs["max_depth"] if "depth" in kwargs else "")
        self.NAME = NAME + (kwargs["max_depth"] if "depth" in kwargs else "")

    def build(self):
        super().build()

    def run(self):
        super().run()

    def write_to_file(self):
        super().write_to_file()

    def read_from_file(self):
        super().read_from_file()
