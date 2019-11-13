from abc import ABC, abstractmethod
from source.utils import file_manager


class Learning(ABC):
    def __init__(self, data_set: file_manager.DataSet, b=False):
        self.data_set = data_set
        self.b = b

    @abstractmethod
    def build(self):
        pass

    @abstractmethod
    def run(self, data_set):
        pass

    @abstractmethod
    def write_to_file(self):
        pass

    @abstractmethod
    def read_from_file(self):
        pass