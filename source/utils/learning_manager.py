from source.utils import file_manager
from source.learning import decision_tree, random_forest, cp_tree, dl8, gradient_boosting

algo_names = {
    "D-tree": decision_tree,
    "R-forest": random_forest,
    "CP-tree": cp_tree,
    "DL8": dl8,
    "G-boosting": gradient_boosting
}


def build_algorithms(algos, b=False):
    for algo in algos:
        for data_set in file_manager.data_sets:
            # TODO: split in train and test
            algo_names[algo].build(data_set, b=b)
            algo_names[algo].run(data_set)


def build_all(b=False):
    build_algorithms(algo_names.keys(), b=b)
