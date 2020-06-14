import random

from source.utils import file_manager
from source.learning import decision_tree, random_forest, cp_tree, dl8, gradient_boosting, dl8_forest, \
    optimised_dl8_forest

algo_names = {
    "D-tree": decision_tree.DecisionTree,
    "R-forest": random_forest.RandomForest,
    "CP-tree": cp_tree.CPTree,
    "DL8": dl8.DL8,
    "DL8-forest": dl8_forest.DL8Forest,
    "OptDL8-forest": optimised_dl8_forest.OptDL8Forest,
    "OptDL8-forest2": optimised_dl8_forest.OptDL8Forest,
    "OptDL8-forest3": optimised_dl8_forest.OptDL8Forest,
    "G-boosting": gradient_boosting.GradientBoosting
}

MAX_DEPTH = 3

kwargs = {
    "D-tree": {'min_samples_leaf': 2, 'max_depth': MAX_DEPTH},
    "R-forest": {'n_estimators': 20, 'max_depth': MAX_DEPTH},
    "CP-tree": {'max_depth': MAX_DEPTH},
    "DL8": {'max_depth': MAX_DEPTH},
    "DL8-forest": {'n_estimators': 10, 'max_depth': MAX_DEPTH, 'method': "all", 'attributes': "random"},
    "OptDL8-forest": {'n_estimators': 1, 'max_depth': MAX_DEPTH, 'method': "all", 'attributes': "progressive",
                      'tree_limit': 10, "time_limit": 60, "error_weight": 1},
    "OptDL8-forest2": {'n_estimators': 1, 'max_depth': MAX_DEPTH, 'method': "all", 'attributes': "progressive",
                       'tree_limit': 10, "time_limit": 60, "error_weight": 0.5},
    "OptDL8-forest3": {'n_estimators': 1, 'max_depth': MAX_DEPTH, 'method': "all", 'attributes': "progressive",
                       'tree_limit': 10, "time_limit": 60, "error_weight": 0.2},
    "G-boosting": {'n_estimators': 20, 'max_depth': MAX_DEPTH}
}

discriminants = {}


def build_algorithms(algos, b=False, percent=0.5, noise=0):
    for algo in algos:
        discriminants[algo] = {}

    for data_set in file_manager.data_sets.values():
        for algo in algos:
            d = algo_names[algo](data_set, b=b, percent=percent, **(kwargs[algo]))
            discriminants[algo][data_set.file] = d
            if not b:
                d.read_from_file()
        print(data_set.file)
        for i in range(10):
            data_set.split(percent)

            data_set.train = [[j if random.randint(0, 100) >= noise else 1-j for j in i] for i in data_set.train]

            for algo in algos:
                d = discriminants[algo][data_set.file]
                if not d.done:
                    print("Pass #" + str(i + 1) + " on " + d.NAME)
                    d.build()
                    d.run()
        for algo in algos:
            d = discriminants[algo][data_set.file]
            d.write_to_file()


def build_all(b=False, percent=0.5, noise=0):
    build_algorithms(algo_names.keys(), b=b, percent=percent, noise=noise)
