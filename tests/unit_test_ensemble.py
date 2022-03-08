from src.autoML import Solvers
from src.dataPreprocessing import DataPreprocessing
import numpy as np
import pandas as pd
from tabulate import tabulate
from datetime import datetime


def test():
    '''
    Tests the optimized ensembles of DR+CL methods + kNN
    Accuracy corresponds to kNN against CL labels.
    Stores resutls in /src/results.txt
    
    '''

    preprocessor = DataPreprocessing()

    X_train, X_test, y_train, y_test = preprocessor.load_data()

    dr_methods = ['PCA', 'UMAP']
    cl_methods = ['KMEANS', 'DBSCAN', 'HDBSCAN']

    ensembles = []
    accuracies_list = []

    f = open('results.txt', 'w')

    for dr_method in dr_methods:
        for cl_method in cl_methods:

            print('===================================================')
            print(f'         Ensemble: {dr_method} + {cl_method}      ')
            print('===================================================')

            methods = [dr_method, cl_method]
            solver = Solvers()
            res = solver.genSolver(train_data=np.asarray(X_train),
                                   test_data=X_test,
                                   true_labels=y_test,
                                   methods=methods)

            ensembles.append((dr_method, cl_method))
            accuracies_list.append((-1 * res.F))
    ''' print results -------------------------- -----------------------'''

    res_dict = {z[0]: list(z[1:]) for z in zip(ensembles, accuracies_list)}

    table = []
    for key, value in res_dict.items():
        table.extend([[key, value[0]]])

    now = datetime.now()

    print(now.strftime("%Y/%m/%d %H:%M:%S"), file=f)

    print(tabulate(table, headers=['ensemble', 'accuracies'], tablefmt="rst"),
          file=f)

    print(tabulate(table, headers=['ensemble', 'accuracies'], tablefmt="rst"))
    '''---------------------------------------------------------------'''
    return None