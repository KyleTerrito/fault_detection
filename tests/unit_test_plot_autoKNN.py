from autocluster.autoKNN import AutoKNN
import pandas as pd

from autocluster.plotters import Plotters
import pickle


def test():

    solutions = pd.read_fwf('tests/autoKNNresults/optKNNoutput.csv',
                            header=None)
    print(solutions)

    ensembles = [i for i in solutions[0][4:-1]]
    solutions_list = [i for i in solutions[1][4:-1]]
    accuracies_list = [i for i in solutions[3][4:-1]]

    res_dict = {
        z[0]: list(z[1:])
        for z in zip(ensembles, solutions_list, accuracies_list)
    }

    hyperparameters_list = [
        "['n_clusters', 'num_neighbors']",
        "['eps', 'min_samples', 'num_neighbors']",
        "['n_comp', 'n_clusters', 'num_neighbors']",
        "['n_comp', 'eps', 'min_samples', 'num_neighbors']"
    ]

    plotter = Plotters()
    plotter.plot_performance(res_dict, hyperparameters_list)

    return None


# ensembles =

# res_dict = {
#         z[0]: list(z[1:])
#         for z in zip(ensembles, solutions_list, accuracies_list)
#     }