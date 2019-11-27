import re
import json
from sklearn.model_selection import train_test_split
import os
import glob

JSON_VERSION = 1.1


class DataSet:
    file = ""
    class_labels = {}
    labels = {}
    data = []
    converted_data = []
    classes = []
    train = []
    test = []
    train_classes = []
    test_classes = []

    def __init__(self, file):
        self.file = file

    def convert_data(self):
        if not self.converted_data:
            self.converted_data = [[0 for _ in range(len(self.labels))] for _ in range(len(self.data))]
            for i in range(len(self.data)):
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
            file = open(g)
            data_sets[g] = parse(file)


def parse(file):
    lines = file.readlines()
    d = DataSet(file.name)
    reading_data = False
    for line in lines:
        if line.strip():
            tokens = line.strip().split()
            if re.match(r"^@[0-9]+:$", tokens[0]) and not reading_data:
                n = int(tokens[0][1:-1])
                label = line[len(tokens[0]):].strip()
                d.labels[n] = label
            elif re.match(r"^@class:$", tokens[0]) and not reading_data:
                for token in tokens[1:]:
                    t = token.split("=")
                    n = int(t[0])
                    c = t[1]
                    c = c[2:-2] if re.match(r"'{.*}'", c) else c[1:-1]
                    d.class_labels[n] = c
            elif re.match(r"^@data$", tokens[0]) and not reading_data:
                reading_data = True
            elif re.match(r"^@.*$", tokens[0]):
                pass
            elif re.match(r"^[0-9]+$", tokens[0]) and reading_data:
                data = []
                for token in tokens[:-1]:
                    i = int(token)
                    data.append(i)
                d.data.append(data)
                d.classes.append(int(tokens[-1]))
            else:
                raise RuntimeError("Incorrect formatting!")
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
