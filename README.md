# Master Thesis: Building Optimal Forests of Decision trees

This code is given for the Master Thesis done at the Université Catholique de Louvain by Sami Bosch in the academic year 2019-2020. This thesis was supervised by Hélène Verhaeghe and Siegfried Nijssen.

In order to run the code, the file "core/run.py" has to be executed. Multiple parameters can be given to it:

- -i [list of files], the files on which the algorithms should be ran. This argument is mandatory. The files are taken as glob, so formulations such as "data/\*" will be recognised.
- -f, if this parameter is given, it will forcefully reconstruct all classifiers instead of reprinting old data. 
- -g, if this parameter is given, various graphs given the performance of the algorithms will be created. This parameter has to be ran with -f if the optimal forest or dl8 forest are being tested.
- -t, if this parameter is given, a latex compliant table will be printed at the end, resuming all the accuracies of the classifiers.
- -s [n], specifies how large the testing set will be in comparison to the original set. n must be between 0 and 1, and is taken as a ratio.
- -n [n], if this parameter is specified, each value in the datasets will have a 100\*n% chance of being corrupted. This is used to test noise.
- -m [method names], a list of classifiers to test. The list is defined in "utils/learning_manager.py" and can be extended accordingly in code. If this parameter isn't given, -a must be specified.
- -a, runs the code on all classifiers defined in "utils/learning_manager.py".

In order to edit the way the optimal or dl8 forests behave, we can edit it where it's declared, in "utils/learning_manager.py". The following parameters can be changed:

- n_estimators: the number of trees to start with for the optimal forest, or the amount of trees that will be built for the dl8 one.
- tree_class: the classifier to use to build the forest. By default DL8. Must be of the same internal structure as the DL8.5 implementation to work.
- method: can be "random" or "all". Specifies whether to train the trees on the whole dataset or on a random subset.
- attributes: can be "random", "all", "progressive" or "progressive_random". Defines how attributes are selection to build each tree, either randomly, always using all, removing always the previous best attribute, or randomly removing attributes.
- optimised: false if we're working with a dl8 forest and true if we want an optimal forest.
- tree_limit: -1 for unlimited. Will stop the optimal forest a given amount of trees after the optimum has passed 0.
- error_weight: How highly the error margin is weighted in the optimal forest.

Any arguments that can be given to the tree classifier given in tree_class can also be added at the end.
