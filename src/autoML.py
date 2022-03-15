"""
-Standalone module

-Provides methods for AutoML via NSGA2 optimization

-Tunable methods for dimensionality reduction, clustering and fault detection 
are imported from src.faultDetection

-A new class is needed for each dr/clustering/fd method to 
define the optimization problem. See pcaTuner() for example.
-Also a new function in Solvers() is needed.

"""
import pickle
import warnings
from math import floor, sqrt
from telnetlib import EL

import numpy as np
import sklearn
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.core.problem import ElementwiseProblem
from pymoo.factory import (get_crossover, get_mutation, get_sampling,
                           get_termination)
from pymoo.operators.mixed_variable_operator import (MixedVariableCrossover,
                                                     MixedVariableMutation,
                                                     MixedVariableSampling)
from pymoo.optimize import minimize
from pymoo.util.display import Display
from pymoo.util.termination.default import SingleObjectiveDefaultTermination
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from tabulate import tabulate

from src.dataPreprocessing import DataPreprocessing
from src.faultDetection import DR, Clustering, FaultDetection

warnings.filterwarnings("ignore")
"""-------------------------------------------------------------------------------"""


class MyDisplay(Display):
    def _do(self, problem, evaluator, algorithm):
        super()._do(problem, evaluator, algorithm)
        #self.output.append("metric_a", -1 * (algorithm.pop.get("F")))
        self.output.append("metric_a", min(algorithm.pop.get("F")))


"""------------------Optimization problems----------------------------------------"""
# TODO: update individual dr/cl methods for new data naming convention


class genTuner(ElementwiseProblem):
    def __init__(self, train_data, test_data, true_labels, methods):

        self.n_var = 0
        self.xl = []
        self.xu = []

        self.dr_method = methods[0]
        self.cl_method = methods[1]

        self.setDRhyper(methods=self.dr_method, train_data=train_data)
        self.setCLhyper(methods=self.cl_method, train_data=train_data)

        # add hyperparameter for kNN
        self.n_var += 1
        self.xl.extend([1])
        self.xu.extend([min(floor(sqrt(len(train_data))), 15)])

        # print(self.n_var)

        # print(self.xl)
        # print(self.xu)

        super().__init__(n_var=self.n_var,
                         n_obj=1,
                         n_constr=0,
                         xl=self.xl,
                         xu=self.xu)

        self.train_data = train_data
        self.test_data = test_data

        le = preprocessing.LabelEncoder().fit(true_labels)
        true_labels = le.transform(true_labels)
        self.true_labels = np.asarray(true_labels)

    def setDRhyper(self, methods, train_data):

        if 'PCA' in methods:
            self.dr_index = 1
            self.n_var = 1
            self.xl = [1]
            self.xu = [min(len(train_data[:, 0]), len(train_data[0, :]))]

        elif 'UMAP' in methods:
            self.dr_index = 3
            self.n_var = 3
            self.xl = [2, 0.1, 2]
            self.xu = [10, 0.99, 5]

        else:
            pass

    def setCLhyper(self, methods, train_data):

        if 'KMEANS' in methods:
            self.cl_index = -1
            self.n_var += 1

            self.xl.extend([1])

            self.xu.extend([min(len(train_data[:, 0]), len(train_data[0, :]))])

        elif 'HDBSCAN' in methods:
            self.cl_index = -3
            self.n_var += 3

            self.xl.extend([2, 1, 0.1])
            self.xu.extend([100, 100, 0.99])

        elif 'H' not in methods and 'DBSCAN' in methods:
            self.cl_index = -2
            self.n_var += 2

            self.xl.extend([1, 3])

            self.xu.extend([100, 50])

        else:
            print('Please select at least on clustering method')
            quit()

    def _evaluate(self, x, out, *args, **kwargs):

        # DR

        dr = DR()
        try:

            dr_model, dr_data = dr.performGEN(self.dr_method, self.train_data,
                                              x[:self.cl_index])

            # rc_data = dr.reconstructGEN(self.dr_method, dr_model, dr_data)

        except:
            dr_data = self.train_data
            # rc_data = self.train_data

        # mse = sklearn.metrics.mean_squared_error(self.train_data, rc_data)

        # Clustering

        cl = Clustering()

        cl_train_labels = cl.performGEN(self.cl_method, dr_data,
                                        x[(self.cl_index - 1):-1])

        n_train_labels = len(set(cl_train_labels))

        # if len(set(cl_train_labels)) > 2:
        #     sil_score = cl.silmetric(dr_data, cl_train_labels)
        # else:
        #     sil_score = -1

        # Fault detection

        fd = FaultDetection()
        # dp = DataPreprocessing()

        cl_X_train, cl_X_test, cl_y_train, cl_y_test = train_test_split(
            dr_data, cl_train_labels, test_size=0.2)

        knn_model = fd.trainKNN(train_data=cl_X_train,
                                labels=cl_y_train,
                                hyperparameters=x[-1])
        ''''
        Test kNN training against CL labels
        '''

        knn_y_test_predicted = fd.predict(knn_model=knn_model,
                                          test_data=cl_X_test)

        # kNN_accuracy = fd.accuracy(true_labels=cl_y_test,
        #                            predicted_labels=knn_y_test_predicted)
        '''
        Test kNN model against ground truth labels
        '''
        try:
            reduced_test_data = dr_model.transform(self.test_data)
        except RecursionError:
            out["F"] = [0]
            return
        except:
            reduced_test_data = self.test_data

        real_y_test_predicted = fd.predict(knn_model=knn_model,
                                           test_data=reduced_test_data)

        aligned_predicted_labels, self.n_labels = fd.alignLabels(
            self.true_labels,
            real_y_test_predicted,
            majority_threshold_percentage=1.0,
            print_reassignments=False)

        #self.n_labels = len(set(aligned_predicted_labels))

        # cl_y_test are the labels from CL, self.true_labels are the ground truth labels
        confusion, accuracy = fd.accuracy(
            true_labels=self.true_labels,
            predicted_labels=aligned_predicted_labels)

        out["F"] = [-1 * accuracy, n_train_labels]


class pcaTuner(ElementwiseProblem):
    '''
    n_var is the number of tunable hyperparameters for the method
    n_obj is the number of objective functions (1 for most of the optimization problems here)
    n_constr is always zero
    xl and xu are lists containing the lower/upper limits allowed for each hyperparameter
        These are based on either intuition or some recommendation for the method.
    '''
    def __init__(self, data):
        super().__init__(n_var=1,
                         n_obj=1,
                         n_constr=0,
                         xl=[1],
                         xu=[min(len(data[:, 0]), len(data[0, :]))])

        self.data = data

    def _evaluate(self, x, out, *args, **kwargs):
        '''
        x is the vector of hyperparameters

        '''
        # Initialize a DR object
        dr = DR()

        # Use the corresponing DR method
        model, dr_data = dr.performPCA(self.data, x)

        rc_data = dr.reconstructPCA(model, dr_data)

        # Compute performance metric
        mse = sklearn.metrics.mean_squared_error(self.data, rc_data)

        # Return value of objective function
        out["F"] = [mse]


class umapTuner(ElementwiseProblem):
    def __init__(self, data):
        super().__init__(n_var=3,
                         n_obj=1,
                         n_constr=0,
                         xl=[2, 2, 0],
                         xu=[25, 5, 0.99])

        self.data = data

    def _evaluate(self, x, out, *args, **kwargs):
        '''
        x is the vector of hyperparameters

        '''

        dr = DR()

        model, dr_data = dr.performUMAP(self.data, x)

        rc_data = dr.reconstructUMAP(model, dr_data)

        mse = sklearn.metrics.mean_squared_error(self.data, rc_data)

        out["F"] = [mse]


class kmeansTuner(ElementwiseProblem):
    def __init__(self, data):
        super().__init__(n_var=1,
                         n_obj=1,
                         n_constr=0,
                         xl=[1],
                         xu=[min(len(data[:, 0]), len(data[0, :]))])

    def _evaluate(self, x, out, *args, **kwargs):

        cl = Clustering()

        labels = cl.performKMEANS(self.data, x)

        if len(set(labels)) > 2:
            sil_score = cl.silmetric(self.data, labels)
        else:
            sil_score = -1

        out["F"] = [-sil_score]


class dbscanTuner(ElementwiseProblem):
    def __init__(self, data):
        super().__init__(n_var=2, n_obj=1, n_constr=0, xl=[1, 3], xu=[100, 50])

        self.data = data

    def _evaluate(self, x, out, *args, **kwargs):
        '''
        x is the vector of hyperparameters

        '''

        cl = Clustering()

        labels = cl.performDBSCAN(self.data, x)

        if len(set(labels)) > 2:
            sil_score = cl.silmetric(self.data, labels)
        else:
            sil_score = -1

        out["F"] = [-sil_score]


class hdbscanTuner(ElementwiseProblem):
    def __init__(self, data):
        super().__init__(n_var=3,
                         n_obj=1,
                         n_constr=0,
                         xl=[2, 1, 0.1],
                         xu=[100, 100, 0.99])

        self.data = data

    def _evaluate(self, x, out, *args, **kwargs):
        '''
        x is the vector of hyperparameters

        '''
        cl = Clustering()

        labels = cl.performHDBSCAN(self.data, x)

        if len(set(labels)) > 2:
            sil_score = cl.silmetric(self.data, labels)
        else:
            sil_score = -1

        out["F"] = [-sil_score]


"""---------------------Solvers----------------------------------------------------"""


class Solvers(ElementwiseProblem):
    '''
    Standalone class

    Contains methods to auto tune fault detection methods

    '''
    def __init__(self):
        super(ElementwiseProblem, self).__init__()

    def masker(self, mask):
        '''
        mask: list containing the type of variable for each hyperparameter
            it can be 'int' for integer variables (i.e., dimensions, min_samples) or
            'real' for real variables (i.e., eps, min_distance).
        '''

        sampling = MixedVariableSampling(mask, {
            "real": get_sampling("real_random"),
            "int": get_sampling("int_random")
        })

        crossover = MixedVariableCrossover(
            mask, {
                "real": get_crossover("real_sbx", prob=1.0, eta=3.0),
                "int": get_crossover("int_sbx", prob=1.0, eta=3.0)
            })

        mutation = MixedVariableMutation(
            mask, {
                "real": get_mutation("real_pm", eta=3.0),
                "int": get_mutation("int_pm", eta=3.0)
            })

        return sampling, crossover, mutation

    def setDRmethodmask(self, methods):
        if 'PCA' in methods:
            self.mask_names.append('n_comp')
            self.mask.append('int')

        elif 'UMAP' in methods:
            self.mask_names.extend(['n_neighbors', 'min_dist', 'n_components'])
            self.mask.extend(['int', 'real', 'int'])

        else:
            pass

    def setCLmethodmask(self, methods):
        if 'KMEANS' in methods:
            self.mask_names.extend(['n_clusters'])
            self.mask.extend(['int'])

        elif 'HDBSCAN' in methods:
            self.mask_names.extend([
                'min_cluster_size', 'min_samples', 'cluster_selection_epsilon'
            ])
            self.mask.extend(['int', 'int', 'real'])

        elif 'H' not in methods and 'DBSCAN' in methods:
            self.mask_names.extend(['eps', 'min_samples'])
            self.mask.extend(['real', 'int'])

        else:
            pass

    def genSolver(self, train_data, test_data, true_labels, methods,
                  termination):
        self.dr_method = methods[0]
        self.cl_method = methods[1]
        self.mask = []
        self.mask_names = []

        self.setDRmethodmask(methods)
        self.setCLmethodmask(methods)

        self.mask_names.append('num_neighbors')
        self.mask.append('int')

        sampling, crossover, mutation = self.masker(mask=self.mask)

        problem = genTuner(train_data, test_data, true_labels,
                           methods)  # use the corresponding problem

        # dv_dict = dict(
        #     zip(self.mask_names, [self.mask, problem.xl, problem.xu]))
        ''' print decision variables for opt problem -----------------------'''

        dv_dict = {
            z[0]: list(z[1:])
            for z in zip(self.mask_names, self.mask, problem.xl, problem.xu)
        }

        table = []
        for key, value in dv_dict.items():
            table.extend([[key, value[0], value[1], value[2]]])

        print(
            tabulate(table,
                     headers=[
                         'hyperparameter', 'type', 'lower limit', 'upper limit'
                     ],
                     tablefmt="rst"))
        '''---------------------------------------------------------------'''

        algorithm = NSGA2(pop_size=10,
                          n_offsprings=5,
                          sampling=sampling,
                          crossover=crossover,
                          mutation=mutation,
                          eliminate_duplicates=True)


        # algorithm = GA(pop_size=10,
        #                n_offsprings=5,
        #                sampling=sampling,
        #                crossover=crossover,
        #                mutation=mutation,
        #                eliminate_duplicates=True)


        if termination == 'test':
            termination = get_termination("n_gen", 2)

        elif termination == 'run':
            termination = SingleObjectiveDefaultTermination(x_tol=1e-8,
                                                            cv_tol=1e-6,
                                                            f_tol=1e-6,
                                                            nth_gen=5,
                                                            n_last=20,
                                                            n_max_gen=1000,
                                                            n_max_evals=100000)

        res = minimize(problem,
                       algorithm,
                       termination=termination,
                       seed=1,
                       save_history=True,
                       verbose=True)

        return res, self.mask_names, problem.n_labels

    def pcaSolver(self, data):

        mask = ['int']

        sampling, crossover, mutation = self.masker(mask=mask)

        problem = pcaTuner(data)  # use the corresponding problem

        algorithm = NSGA2(pop_size=300,
                          n_offsprings=4,
                          sampling=sampling,
                          crossover=crossover,
                          mutation=mutation,
                          eliminate_duplicates=True)

        res = minimize(problem,
                       algorithm,
                       termination=get_termination("n_gen", 10),
                       seed=1,
                       save_history=True,
                       verbose=True)

        return res

    def umapSolver(self, data):

        mask = ['int', 'int', 'real']

        sampling, crossover, mutation = self.masker(mask=mask)

        problem = umapTuner(data)

        algorithm = NSGA2(pop_size=5,
                          n_offsprings=2,
                          sampling=sampling,
                          crossover=crossover,
                          mutation=mutation,
                          eliminate_duplicates=True)

        res = minimize(problem,
                       algorithm,
                       termination=get_termination("n_gen", 10),
                       seed=1,
                       save_history=True,
                       verbose=True)

        return res

    def kmeansSolver(self, data):

        mask = ['real']

        sampling, crossover, mutation = self.masker(mask=mask)

        problem = kmeansTuner(data)

        algorithm = NSGA2(pop_size=5,
                          n_offsprings=2,
                          sampling=sampling,
                          crossover=crossover,
                          mutation=mutation,
                          eliminate_duplicates=True)

        res = minimize(problem,
                       algorithm,
                       termination=get_termination("n_gen", 10),
                       seed=1,
                       save_history=True,
                       display=MyDisplay(),
                       verbose=True)

        return res

    def dbscanSolver(self, data):

        mask = ['real', 'int']

        sampling, crossover, mutation = self.masker(mask=mask)

        problem = dbscanTuner(data)

        algorithm = NSGA2(pop_size=5,
                          n_offsprings=2,
                          sampling=sampling,
                          crossover=crossover,
                          mutation=mutation,
                          eliminate_duplicates=True)

        res = minimize(problem,
                       algorithm,
                       termination=get_termination("n_gen", 10),
                       seed=1,
                       save_history=True,
                       display=MyDisplay(),
                       verbose=True)

        return res

    def hdbscanSolver(self, data):

        mask = ['int', 'int', 'real']

        sampling, crossover, mutation = self.masker(mask=mask)

        problem = hdbscanTuner(data)

        algorithm = NSGA2(pop_size=5,
                          n_offsprings=2,
                          sampling=sampling,
                          crossover=crossover,
                          mutation=mutation,
                          eliminate_duplicates=True)

        res = minimize(problem,
                       algorithm,
                       termination=get_termination("n_gen", 10),
                       seed=1,
                       save_history=True,
                       display=MyDisplay(),
                       verbose=True)

        return res


class Metrics():
    def __init__(self):
        pass

    def get_metrics(self, res_dict, X_train):

        # preprocessor = DataPreprocessing()

        # X_train, X_test, y_train, y_test = preprocessor.load_data()

        # res_dict = pickle.load(
        #     open('tests/ensemble_test_results/res_dict.pkl', 'rb'))

        mse_values = []
        sil_values = []
        CH_values = []
        DBI_values = []
        n_clusters_values = []
        dr = DR()
        cl = Clustering()

        for key, values in res_dict.items():

            methods = [key[0]]

            h_values = [i for i in values[0]]

            for i in range(len(methods)):

                if methods[i] == 'NO DR':
                    mse = 0
                    dr_data = X_train

                    cl_train_labels = cl.performGEN(key[1], dr_data,
                                                    h_values[:-1])

                    non_noise_index = np.where(cl_train_labels != -1)
                    non_noise_cl_train_labels = cl_train_labels[non_noise_index]
                    non_noise_data = dr_data.iloc[non_noise_index]

                    if len(set(cl_train_labels)) > 2:
                        sil_score = cl.silmetric(non_noise_data, non_noise_cl_train_labels)
                        CH_score = cl.CHindexmetric(non_noise_data, non_noise_cl_train_labels)
                        DBI_score = cl.DBImetric(non_noise_data, non_noise_cl_train_labels)
                    else:
                        sil_score = -1
                        CH_score = 0
                        DBI_score = 0

                elif methods[i] == 'PCA':

                    dr_model, dr_data = dr.performGEN(methods[i], X_train,
                                                      h_values[:1])

                    rc_data = dr.reconstructGEN(methods[i], dr_model, dr_data)

                    mse = sklearn.metrics.mean_squared_error(X_train, rc_data)

                    cl_train_labels = cl.performGEN(key[1], dr_data,
                                                    h_values[1:-1])

                    non_noise_index = np.where(cl_train_labels != -1)
                    non_noise_cl_train_labels = cl_train_labels[non_noise_index]
                    non_noise_data = dr_data[non_noise_index]


                    if len(set(cl_train_labels)) > 2:
                        sil_score = cl.silmetric(non_noise_data, non_noise_cl_train_labels)
                        CH_score = cl.CHindexmetric(non_noise_data, non_noise_cl_train_labels)
                        DBI_score = cl.DBImetric(non_noise_data, non_noise_cl_train_labels)
                    else:
                        sil_score = -1
                        CH_score = 0
                        DBI_score = 0

                elif methods[i] == 'UMAP':

                    dr_model, dr_data = dr.performGEN(methods[i], X_train,
                                                      h_values[:3])

                    rc_data = dr.reconstructGEN(methods[i], dr_model, dr_data)

                    mse = sklearn.metrics.mean_squared_error(X_train, rc_data)

                    print(key[1])
                    print(h_values)
                    print(h_values[0:-1])
                    print(h_values[1:-1])
                    print(h_values[2:-1])
                    cl_train_labels = cl.performGEN(key[1], dr_data,
                                                    h_values[3:-1])
                    
                    non_noise_index = np.where(cl_train_labels != -1)
                    non_noise_cl_train_labels = cl_train_labels[non_noise_index]
                    non_noise_data = dr_data[non_noise_index]


                    if len(set(cl_train_labels)) > 2:
                        sil_score = cl.silmetric(non_noise_data, non_noise_cl_train_labels)
                        CH_score = cl.CHindexmetric(non_noise_data, non_noise_cl_train_labels)
                        DBI_score = cl.DBImetric(non_noise_data, non_noise_cl_train_labels)
                    else:
                        sil_score = -1
                        CH_score = 0
                        DBI_score = 0

                #print(f'Labels in metrics: {set(cl_train_labels)}')
                mse_values.append(mse)
                sil_values.append(sil_score)
                CH_values.append(CH_score)
                DBI_values.append(DBI_score)
                n_clusters_values.append(len(set(cl_train_labels)))


        return mse_values, sil_values, CH_values, DBI_values, n_clusters_values


    def get_best(self, res):

        for i in range(len(res.F)):
            if res.F[i, 0] == min(res.F[:, 0]):
                best_x = res.X[i]
                best_f = res.F[i]

        return best_x, best_f

