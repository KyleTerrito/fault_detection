from autocluster.dataPreprocessing import DataPreprocessing
from autocluster.autoML import Solvers, Metrics
from autocluster.plotters import Plotters
import numpy as np
from tabulate import tabulate
from datetime import datetime, date
import csv
import pickle
import pandas as pd
import itertools


class AutoKNN(DataPreprocessing, Solvers, Metrics, Plotters):
    '''
    Collects methods for optimization and visualization of DR+Clustering ensembles.
    See examples of usage in tests/opt_full_tep.py and opt_metrics_pyro.py
    
    '''
    def __init__(self, *args, **kwargs):
        super(AutoKNN, self).__init__(*args, **kwargs)

    def get_data(self, path, exp='default', p_method = 'mean'):
        X_train, X_test, y_train, y_test = self.load_data(
            path=path, normalize_method=p_method)

        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.exp = exp

        return None

    def optimize(self,
                 p_methods, 
                 dr_methods,
                 cl_methods,
                 termination='test',
                 mode='supervised',
                 data_path = None, 
                 exp = 'default'):

        self.exp = exp
        self.ensembles = []
        self.hyperparameters_list = []
        self.solutions_list = []
        self.accuracies_list = []
        self.n_labels_list = []

        permutations = list(itertools.product(*[p_methods, dr_methods, cl_methods]))
        
        for ensemble in permutations:

            p_method = ensemble[0]
            dr_method = ensemble[1]
            cl_method = ensemble[2]

            self.get_data(path=data_path, p_method=p_method, exp=self.exp)

            print('======================================================')
            print(f' Ensemble: {p_method} + {dr_method} + {cl_method} ')
            print('======================================================')

            methods = [p_method, dr_method, cl_method]

            res, hyperparameters, n_labels, sil_scores, ch_scores, dbi_scores, it_accuracies, it_clusters = self.genSolver(
                train_data=np.asarray(self.X_train),
                test_data=self.X_test,
                true_labels=self.y_test,
                methods=methods,
                termination=termination,
                mode=mode)

            self.ensembles.append((p_method, self.dr_method, self.cl_method))
            self.hyperparameters_list.append(hyperparameters)
            res.F = -1 * res.F

            self.solutions_list.append((res.X))
            self.accuracies_list.append((res.F))
            self.n_labels_list.append(n_labels)

            metrics = [
                    sil_scores, ch_scores, dbi_scores, it_accuracies,
                    it_clusters
                ]

            # print('---------------------------')
            # print(metrics)

            metrics = pd.DataFrame(metrics,
                                    index=[
                                        f'{dr_method}_{cl_method}_sil',
                                        f'{dr_method}_{cl_method}_ch',
                                        f'{dr_method}_{cl_method}_dbi',
                                        f'{dr_method}_{cl_method}_acc',
                                        f'{dr_method}_{cl_method}_cl',
                                    ])

            metrics_file = open(
                f'tests/ensemble_test_results/{self.exp}metrics_{p_method}_{dr_method}_{cl_method}.pkl',
                'wb')
            pickle.dump(metrics, metrics_file)

        h_file = open(f'tests/ensemble_test_results/{self.exp}_h_list.pkl',
                      'wb')
        pickle.dump(self.hyperparameters_list, h_file)
        print(self.accuracies_list)

        res_dict = {
            z[0]: list(z[1:])
            for z in zip(self.ensembles, self.solutions_list,
                         self.accuracies_list)
        }

        return res_dict, self.hyperparameters_list

    def show_solutions(self):
        self.res_dict = {
            z[0]: list(z[1:])
            for z in zip(self.ensembles, self.solutions_list,
                         self.n_labels_list, self.accuracies_list)
        }

        table = []

        for key, value in self.res_dict.items():
            table.extend([[key, value[0], value[1], -1 * value[2]]])

        today = date.today()
        d = today.strftime("%b-%d-%Y")
        f = open(
            f'tests/ensemble_test_results/{self.exp}result_unit_test_ensemble{d}.txt',
            'w')

        now = datetime.now()

        print(now.strftime("%Y/%m/%d %H:%M:%S"), file=f)

        print(tabulate(
            table,
            headers=['ensemble', 'solutions', 'n of clusters', 'accuracies'],
            tablefmt="rst"),
              file=f)

        print(
            tabulate(table,
                     headers=[
                         'ensemble', 'solutions', 'n of clusters', 'accuracies'
                     ],
                     tablefmt="rst"))

        return None

    def plot_results(self):
        rc_error, sil_scores, CH_scores, DBI_scores, n_clusters = self.get_metrics(
            res_dict=self.res_dict, X_train=self.X_train)

        self.plot_metrics(res_dict=self.res_dict,
                          reconstruction_errors=rc_error,
                          sil_scores=None,
                          CH_scores=None,
                          DBI_scores=None,
                          n_clusters=None)

        self.plot_metrics(res_dict=self.res_dict,
                          reconstruction_errors=None,
                          sil_scores=sil_scores,
                          CH_scores=None,
                          DBI_scores=None,
                          n_clusters=None)

        self.plot_metrics(res_dict=self.res_dict,
                          reconstruction_errors=None,
                          sil_scores=None,
                          CH_scores=CH_scores,
                          DBI_scores=None,
                          n_clusters=None)

        self.plot_metrics(res_dict=self.res_dict,
                          reconstruction_errors=None,
                          sil_scores=None,
                          CH_scores=None,
                          DBI_scores=DBI_scores,
                          n_clusters=None)

        self.plot_metrics(res_dict=self.res_dict,
                          reconstruction_errors=None,
                          sil_scores=None,
                          CH_scores=None,
                          DBI_scores=None,
                          n_clusters=n_clusters)
        return None
