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
    "D-tree": {},
    "R-forest": {},
    "CP-tree": {},
    "DL8": {},
    "G-boosting": {}
}

discriminants = {}


def build_algorithms(algos, b=False):
    for algo in algos:
        discriminants[algo] = {}

    for data_set in file_manager.data_sets:
        for algo in algos:
            # TODO: split in train and test
            d = algo_names[algo](data_set, b=b, **kwargs[algo])
            discriminants[algo][data_set.file] = d
            d.build()


def build_all(b=False):
    build_algorithms(algo_names.keys(), b=b)
