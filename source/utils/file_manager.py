import re

data_sets = {}


class DataSet:
    file = ""
    class_labels = {}
    labels = {}
    data = []
    converted_data = []
    classes = []

    def __init__(self, file):
        self.file = file

    def convert_data(self):
        self.converted_data = [[0 for _ in range(len(self.labels))] for _ in range(len(self.data))]
        for i in range(len(self.data)):
            for j in self.data[i]:
                self.converted_data[i][j] = 1


def read(files):
    for f in files:
        file = open(f)
        data_sets[f] = parse(file)


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
                d.classes.apped(int(tokens[-1]))
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
    if not d.converted_data:
        d.convert_data()
    return d.converted_data
