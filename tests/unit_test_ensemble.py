from copyreg import pickle
from src.autoML import Solvers
from src.dataPreprocessing import DataPreprocessing
from src.plotters import Plotters
from src.autoML import Metrics
import numpy as np
import pandas as pd
from tabulate import tabulate
from datetime import datetime, date
import pickle


def test():
    '''
    Tests the optimized ensembles of DR+CL methods + kNN
    Accuracy corresponds to kNN against CL labels.
    Stores resutls in /tests/results.txt

    '''
    '''
    Start - Compute accuracies------------------------------------
    '''

    # Load data
    preprocessor = DataPreprocessing()

    X_train, X_test, y_train, y_test = preprocessor.load_data(
        path='data/processed/TEP_Selected_Faults.xlsx')

    # Save data to use later in metrics
    X_train_file = open('tests/ensemble_test_results/X_train_file.pkl', 'wb')
    pickle.dump(X_train, X_train_file)

    # Set up methods
    dr_methods = ['NO DR', 'PCA']  # , 'UMAP']
    cl_methods = ['KMEANS', 'DBSCAN']  # , 'HDBSCAN']

    ensembles = []
    hyperparameters_list = []
    solutions_list = []
    accuracies_list = []

    for dr_method in dr_methods:
        for cl_method in cl_methods:

            print('===================================================')
            print(f'         Ensemble: {dr_method} + {cl_method}      ')
            print('===================================================')

            methods = [dr_method, cl_method]
            solver = Solvers()
            res, hyperparameters = solver.genSolver(
                train_data=np.asarray(X_train),
                test_data=X_test,
                true_labels=y_test,
                methods=methods)

            ensembles.append((dr_method, cl_method))
            hyperparameters_list.append(hyperparameters)
            solutions_list.append((res.X))
            accuracies_list.append((-1 * res.F))

    h_file = open('tests/ensemble_test_results/h_list.pkl', 'wb')
    pickle.dump(hyperparameters_list, h_file)
    '''
    End - Compute accuracies---------------------------------------
    '''
    '''
    Start - Print results -------------------------- --------------
    '''

    res_dict = {
        z[0]: list(z[1:])
        for z in zip(ensembles, solutions_list, accuracies_list)
    }

    res_file = open('tests/ensemble_test_results/res_dict.pkl', 'wb')
    pickle.dump(res_dict, res_file)

    table = []
    for key, value in res_dict.items():
        table.extend([[key, value[0], value[1]]])

    today = date.today()
    d = today.strftime("%b-%d-%Y")
    f = open(f'tests/ensemble_test_results/result_unit_test_ensemble{d}.txt',
             'w')

    now = datetime.now()

    print(now.strftime("%Y/%m/%d %H:%M:%S"), file=f)

    print(tabulate(table,
                   headers=['ensemble', 'solutions', 'accuracies'],
                   tablefmt="rst"),
          file=f)

    print(
        tabulate(table,
                 headers=['ensemble', 'solutions', 'accuracies'],
                 tablefmt="rst"))
    '''
    End - Print results ---------------------------------------------
    '''
    '''
    Start - Plot results---------------------------------------------
    '''
    plotter = Plotters()

    plotter.plot_performance(res_dict, hyperparameters_list)

    res_dict_test = pickle.load(
        open('tests/ensemble_test_results/res_dict.pkl', 'rb'))

    metrics = Metrics()

    rc_error, sil_scores, CH_scores, DBI_scores, n_clusters = metrics.get_metrics(
        res_dict=res_dict_test, X_train=X_train)

    plotter.plot_metrics(res_dict=res_dict,
                         reconstruction_errors=rc_error,
                         sil_scores=None,
                         CH_scores=None,
                         DBI_scores=None,
                         n_clusters=None)

    plotter.plot_metrics(res_dict=res_dict,
                         reconstruction_errors=None,
                         sil_scores=sil_scores,
                         CH_scores=None,
                         DBI_scores=None,
                         n_clusters=None)

    plotter.plot_metrics(res_dict=res_dict,
                         reconstruction_errors=None,
                         sil_scores=None,
                         CH_scores=CH_scores,
                         DBI_scores=None,
                         n_clusters=None)

    plotter.plot_metrics(res_dict=res_dict,
                         reconstruction_errors=None,
                         sil_scores=None,
                         CH_scores=None,
                         DBI_scores=DBI_scores,
                         n_clusters=None)

    plotter.plot_metrics(res_dict=res_dict,
                         reconstruction_errors=None,
                         sil_scores=None,
                         CH_scores=None,
                         DBI_scores=None,
                         n_clusters=n_clusters)
    '''
    End - Plot results---------------------------------------------
    '''
    return X_train
