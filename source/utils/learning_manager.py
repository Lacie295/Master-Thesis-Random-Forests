from source.utils import file_manager
from source.learning import decision_tree, random_forest, cp_tree, dl8, gradient_boosting

algo_names = {
    "D-tree": decision_tree.DecisionTree,
    "R-forest": random_forest.RandomForest,
    "CP-tree": cp_tree.CPTree,
    "DL8": dl8.DL8,
    "G-boosting": gradient_boosting.GradientBoosting
}

kwargs = {
    "D-tree": {'min_samples_leaf': 2},
    "R-forest": {'n_estimators': 100},
    "CP-tree": {},
    "DL8": {},
    "G-boosting": {'n_estimators': 100}
}

discriminants = {}


def build_algorithms(algos, b=False, percent=0.5):
    for algo in algos:
        discriminants[algo] = {}

    for data_set in file_manager.data_sets.values():
        print(data_set.file)
        for algo in algos:
            d = algo_names[algo](data_set, b=b, percent=percent, **(kwargs[algo]))
            discriminants[algo][data_set.file] = d
            if not b:
                d.read_from_file()
            if not d.done:
                for i in range(10):
                    print("Pass #" + str(i + 1) + " on " + d.NAME)
                    data_set.split(percent)
                    d.build()
                    d.run()
            d.write_to_file()


def build_all(b=False, percent=0.5):
    build_algorithms(algo_names.keys(), b=b, percent=percent)
