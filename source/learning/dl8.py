from source.utils import file_manager
from source.learning import learning


class DL8(learning.Learning):
    def __init__(self, data_set: file_manager.DataSet, b=False, **kwargs):
        super().__init__(data_set, b=b)

    def build(self):
        pass

    def run(self):
        pass

    def write_to_file(self):
        pass

    def read_from_file(self):
        return False
