from source.utils import file_manager
from source.learning import learning
from dl85 import ODTClassifier

NAME = "DL8"
FILE = "dl8"


class DL8(learning.Learning):
    def __init__(self, data_set: file_manager.DataSet, percent=0.5, b=False, **kwargs):
        super().__init__(data_set, percent=percent, b=b)
        self.t = ODTClassifier(**kwargs)
        self.FILE = FILE
        self.NAME = NAME

    def build(self):
        super().build()
        print(self.t.error_)

    def run(self):
        super().run()

    def write_to_file(self):
        super().write_to_file()

    def read_from_file(self):
        super().read_from_file()
