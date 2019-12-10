from source.utils import file_manager
from source.learning import learning
import subprocess

NAME = "CP Tree"
FILE = "cp_tree"
TRAIN = "temp/train.txt"
TEST = "temp/test.txt"
ARGS = (
    "/Users/Sami/Downloads/helene_verhaeghe-classificationtree-f08622212045/classificationtree/target/pack/bin/cp-decision-tree",
    TRAIN, "-format=binarypre", "-t", TEST)


class CPTree(learning.Learning):
    def __init__(self, data_set: file_manager.DataSet, percent=0.5, b=False, **kwargs):
        super().__init__(data_set, percent=percent, b=b)
        self.FILE = FILE
        self.NAME = NAME

    def build(self):
        if not self.done:
            self.size = len(self.data_set.train)

    def run(self):
        file = open(TRAIN, "w+")
        for i in range(len(self.data_set.train)):
            s = str(self.data_set.train_classes[i])
            row = self.data_set.train[i]
            for value in row:
                s += " " + str(value)
            file.write(s + "\n")
        file.close()

        file = open(TEST, "w+")
        for i in range(len(self.data_set.test)):
            s = str(self.data_set.test_classes[i])
            row = self.data_set.test[i]
            for value in row:
                s += " " + str(value)
            file.write(s + "\n")
        file.close()

        process = subprocess.Popen(args=ARGS, stdout=subprocess.PIPE)
        process.wait()
        output = process.stdout.readlines()
        acc = float(str(output[-1]).split(" ")[-1][:-3])
        self.avg_acc = (self.avg_acc * self.n_runs + acc) / (self.n_runs + 1)
        self.n_runs += 1

    def write_to_file(self):
        super().write_to_file()

    def read_from_file(self):
        super().read_from_file()
