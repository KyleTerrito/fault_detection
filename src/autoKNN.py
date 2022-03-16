from src.dataPreprocessing import DataPreprocessing
from src.autoML import Solvers, Metrics
from src.plotters import Plotters
import numpy as np
from tabulate import tabulate
from datetime import datetime, date
import csv
import pickle
import pandas as pd


class AutoKNN(DataPreprocessing, Solvers, Metrics, Plotters):
    def __init__(self, *args, **kwargs):
        super(AutoKNN, self).__init__(*args, **kwargs)

    def get_data(self, path):
        X_train, X_test, y_train, y_test = self.load_data(
            path=path, normalize_method="mean")

        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test

        return None

    def optimize(self, dr_methods, cl_methods, termination='test'):
        # dr_methods = ['NO DR', 'PCA']  # , 'UMAP']
        # cl_methods = ['KMEANS', 'DBSCAN']  # , 'HDBSCAN']

        self.ensembles = []
        self.hyperparameters_list = []
        self.solutions_list = []
        self.accuracies_list = []
        self.n_labels_list = []

        for dr_method in dr_methods:
            for cl_method in cl_methods:

                print('===================================================')
                print(f'         Ensemble: {dr_method} + {cl_method}      ')
                print('===================================================')

                methods = [dr_method, cl_method]
                #solver = Solvers()
                res, hyperparameters, n_labels, sil_scores, ch_scores, dbi_scores, it_accuracies = self.genSolver(
                    train_data=np.asarray(self.X_train),
                    test_data=self.X_test,
                    true_labels=self.y_test,
                    methods=methods,
                    termination=termination)
                #best_X, best_f = self.get_best(res)

                self.ensembles.append((self.dr_method, self.cl_method))
                self.hyperparameters_list.append(hyperparameters)
                self.solutions_list.append((res.X))
                self.accuracies_list.append((-1 * res.F))
                self.n_labels_list.append(n_labels)

                metrics = [sil_scores, ch_scores, dbi_scores, it_accuracies]

                # print('---------------------------')
                # print(metrics)

                metrics = pd.DataFrame(metrics,
                                       index=[
                                           f'{dr_method}_{cl_method}_sil',
                                           f'{dr_method}_{cl_method}_ch',
                                           f'{dr_method}_{cl_method}_dbi',
                                           f'{dr_method}_{cl_method}_acc'
                                       ])

                metrics_file = open(
                    f'tests/ensemble_test_results/metrics{dr_method}_{cl_method}.pkl',
                    'wb')
                pickle.dump(metrics, metrics_file)

        h_file = open('tests/ensemble_test_results/h_list.pkl', 'wb')
        pickle.dump(self.hyperparameters_list, h_file)
        print(self.accuracies_list)
        res_dict = {
            z[0]: list(z[1:])
            for z in zip(self.ensembles, self.solutions_list,
                         self.accuracies_list)
        }

        res_file = open('tests/ensemble_test_results/res_dict.pkl', 'wb')
        pickle.dump(res_dict, res_file)

        # output = [
        #     self.ensembles, self.hyperparameters_list, self.solutions_list,
        #     self.accuracies_list, self.n_labels_list
        # ]

        # with open("tests/autoKNNresults/optKNNoutput.csv", "w",
        #           newline="") as f:
        #     writer = csv.writer(f)
        #     writer.writerows(output)

        return None

    def show_solutions(self):
        self.res_dict = {
            z[0]: list(z[1:])
            for z in zip(self.ensembles, self.solutions_list,
                         self.n_labels_list, self.accuracies_list)
        }

        table = []
        for key, value in self.res_dict.items():
            table.extend([[key, value[0], value[1], value[2]]])

        today = date.today()
        d = today.strftime("%b-%d-%Y")
        f = open(
            f'tests/ensemble_test_results/result_unit_test_ensemble{d}.txt',
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
