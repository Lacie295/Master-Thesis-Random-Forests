import re
import json
from sklearn.model_selection import train_test_split
import os
import glob

JSON_VERSION = 1.2
s = "0"


class DataSet:
    def __init__(self, file):
        self.file = file
        self.data = []
        self.converted_data = []
        self.classes = []
        self.train = []
        self.test = []
        self.train_classes = []
        self.test_classes = []
        self.size = 0
        self.n_params = 0

    def convert_data(self):
        if not self.converted_data:
            self.converted_data = [[0 for _ in range(self.n_params)] for _ in range(self.size)]
            for i in range(self.size):
                for j in self.data[i]:
                    self.converted_data[i][j] = 1

    def get_converted_data(self):
        self.convert_data()
        return self.converted_data

    def split(self, percent):
        i = list(range(len(self.data)))
        self.train, self.test, self.train_classes, self.test_classes = \
            train_test_split(self.get_converted_data(), self.classes, train_size=percent)


data_sets = {}


def read(files):
    for f in files:
        for g in glob.glob(f):
            print("Reading " + g + ".")
            file = open(g)
            data_sets[g] = parse(file)
            d = data_sets
            pass


def parse(file):
    # Parse the data files into a DataSet object
    lines = file.readlines()
    d = DataSet(file.name)
    n_params = -1
    for line in lines:
        if line.strip():
            tokens = line.split()
            if re.match(r"^[0-9]+$", tokens[0]):
                data = []
                for token in tokens[:-1]:
                    i = int(token)
                    data.append(i)
                    if i >= n_params:
                        n_params = i + 1
                d.data.append(data)
                d.classes.append(int(tokens[-1]))
            else:
                raise RuntimeError("Incorrect formatting!")
    d.size = len(d.data)
    d.n_params = n_params
    return d


def get_file(file):
    return data_sets[file]


def get_data(file):
    return get_file(file).data


def get_classes(file):
    return get_file(file).classes


def get_class_labels(file):
    return get_file(file).class_labels


def get_labels(file):
    return get_file(file).labels


def get_converted(file):
    d = get_file(file)
    return d.get_converted_data()


def write_to_db(file, category, d):
    file = file_db(file)
    if os.path.exists(file):
        f = open(file, "r")
        data = json.load(f)
        if "version" not in data or data["version"] < JSON_VERSION:
            data = {"version": JSON_VERSION}
    else:
        data = {}
    f = open(file, "w+")
    data[category] = d
    json.dump(data, f)


def read_from_db(file, category):
    file = file_db(file)
    if os.path.exists(file):
        f = open(file, "r")
        data = json.load(f)
        if "version" not in data or data["version"] < JSON_VERSION:
            return {}
        elif category in data:
            return data[category]
        else:
            return {}
    else:
        return {}


def file_db(file):
    return "results/" + file + ".json"


def set_split(split):
    global s
    s = split
