import re

data_sets = {}


def read(files):
    for f in files:
        file = open(f)
        data_sets[f] = parse(file)


def parse(file):
    lines = file.readlines()
    data = []
    labels = {}
    classes = {}
    reading_data = False
    for line in lines:
        if line:
            tokens = line.strip().split(" ")
            if re.match(r"^@[0-9]+:$", tokens[0]) and not reading_data:
                n = int(tokens[0][1:-1])
                label = line[len(tokens[0]):].strip()
                labels[n] = label
            elif re.match(r"^@class:$", tokens[0]) and not reading_data:
                for token in tokens[1:]:
                    t = token.split("=")
                    n = int(t[0])
                    c = t[1]
                    c = c[2:-2] if re.match(r"'{.*}'", c) else c[1:-1]
                    classes[n] = c
            elif re.match(r"^@data:$", tokens[0]) and not reading_data:
                reading_data = True
            elif re.match(r"^@.*$", tokens[0]):
                pass
            elif re.match(r"^[0-9]+$", tokens[0]) and reading_data:
                d = []
                for token in tokens:
                    i = int(token)
                    d.append(i)
                data.append(d)
            else:
                raise RuntimeError("Incorrect formatting!")
    return classes, labels, data


def get_file(file):
    return data_sets[file]
