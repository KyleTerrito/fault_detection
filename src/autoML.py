"""
-Standalone module

-Provides methods for AutoML via NSGA2 optimization

-Tunable methods for dimensionality reduction, clustering and fault detection 
are imported from src.faultDetection

-A new class is needed for each dr/clustering/fd method to 
define the optimization problem. See pcaTuner() for example.
-Also a new function in Solvers() is needed.

"""
from telnetlib import EL
import numpy as np
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.problem import ElementwiseProblem
from pymoo.factory import (get_crossover, get_mutation, get_sampling,
                           get_termination)
from pymoo.operators.mixed_variable_operator import (MixedVariableCrossover,
                                                     MixedVariableMutation,
                                                     MixedVariableSampling)
from pymoo.optimize import minimize
from pymoo.util.display import Display

from src.faultDetection import DR, Clustering, FaultDetection
import sklearn
from math import sqrt

import warnings
warnings.filterwarnings("ignore")
"""-------------------------------------------------------------------------------"""


class MyDisplay(Display):
    def _do(self, problem, evaluator, algorithm):
        super()._do(problem, evaluator, algorithm)
        #self.output.append("metric_a", -1 * (algorithm.pop.get("F")))
        self.output.append("metric_a", min(algorithm.pop.get("F")))


"""------------------Optimization problems----------------------------------------"""
#TODO: add multiobjective optimization for dr+clustering+fault_detection problems


class genTuner(ElementwiseProblem):
    def __init__(self, data, methods):

        self.dr_method = methods[0]
        self.cl_method = methods[1]

        if 'PCA' in methods:
            self.dr_index = 1
            self.n_var = 1
            self.xl = [1]
            self.xu = [min(len(data[:, 0]), len(data[0, :]))]

            if 'KMEANS' in methods:
                self.cl_index = -1
                self.n_var += 1

                self.xl.append(1)

                self.xu.append(min(len(data[:, 0]), len(data[0, :])))

            if 'HDBSCAN' in methods:
                self.cl_index = -3
                self.n_var += 3

                self.xl.append(2)
                self.xl.append(1)
                self.xl.append(0.1)

                self.xu.append(100)
                self.xu.append(100)
                self.xu.append(0.99)

            if 'H' not in methods and 'DBSCAN' in methods:
                self.cl_index = -2
                self.n_var += 2

                self.xl.append(1)
                self.xl.append(3)

                self.xu.append(100)
                self.xu.append(50)

        if 'UMAP' in methods:
            self.dr_index = 3
            self.n_var = 3
            self.xl = [2, 2, 0],
            self.xu = [25, 5, 0.99]

            if 'KMEANS' in methods:
                self.cl_index = -1
                self.n_var += 1

                self.xl.append(1)

                self.xu.append(min(len(data[:, 0]), len(data[0, :])))

            if 'HDBSCAN' in methods:
                self.cl_index = -3
                self.n_var += 3

                self.xl.append(2)
                self.xl.append(1)
                self.xl.append(0.1)

                self.xu.append(100)
                self.xu.append(100)
                self.xu.append(0.99)

            if 'H' not in methods and 'DBSCAN' in methods:
                self.cl_index = -2
                self.n_var += 2

                self.xl.append(1)
                self.xl.append(3)

                self.xu.append(100)
                self.xu.append(50)

        #add hyperparameter for kNN
        self.n_var += 1
        self.xl.append(1)
        self.xu.append(sqrt(len(data[0, :])))

        print(self.n_var)
        print(self.xl)
        print(self.xu)

        super().__init__(n_var=self.n_var,
                         n_obj=2,
                         n_constr=0,
                         xl=self.xl,
                         xu=self.xu)

        self.data = data

    def _evaluate(self, x, out, *args, **kwargs):

        #DR
        dr = DR()
        model, dr_data = dr.performGEN(self.dr_method, self.data,
                                       x[:self.cl_index])

        rc_data = dr.reconstructGEN(self.dr_method, model, dr_data)

        mse = sklearn.metrics.mean_squared_error(self.data, rc_data)

        #Clustering
        cl = Clustering()

        labels = cl.performGEN(self.cl_method, dr_data,
                               x[(self.cl_index - 1):-1])

        if len(set(labels)) > 2:
            sil_score = cl.silmetric(dr_data, labels)
        else:
            sil_score = -1

        #Fault detection
        fd = FaultDetection()

        knn_model = fd.trainKNN(train_data=self.data,
                                labels=labels,
                                hyperparameters=x[-1])

        labels = fd.predict(knn_model=knn_model, test_data=self.data)

        confusion, accuracy = fd.accuracy(true_labels=self.labels,
                                          predicted_labels=labels)

        out["F"] = [accuracy]


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
        #Initialize a DR object
        dr = DR()

        #Use the corresponing DR method
        model, dr_data = dr.performPCA(self.data, x)

        rc_data = dr.reconstructPCA(model, dr_data)

        #Compute performance metric
        mse = sklearn.metrics.mean_squared_error(self.data, rc_data)

        #Return value of objective function
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
        print(x)
        dr = DR()

        model, dr_data = dr.performUMAP(self.data, x)

        rc_data = dr.reconstructUMAP(model, dr_data)

        mse = sklearn.metrics.mean_squared_error(self.data, rc_data)

        # print('------------------------')
        # print(f'mse = {mse}')
        # print('------------------------')

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
        #print(x)
        cl = Clustering()

        labels = cl.performDBSCAN(self.data, x)

        # print('------------------------')
        # print(f'Number of labels = {len(set(labels))}')

        if len(set(labels)) > 2:
            sil_score = cl.silmetric(self.data, labels)
        else:
            sil_score = -1

        # print('------------------------')
        # print(f'silhouette score = {sil_score}')

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
        #print(x)
        cl = Clustering()

        labels = cl.performHDBSCAN(self.data, x)
        #print(labels)

        # print('------------------------')
        # print(f'Number of labels = {len(set(labels))}')

        if len(set(labels)) > 2:
            sil_score = cl.silmetric(self.data, labels)
        else:
            sil_score = -1

        out["F"] = [-sil_score]

        # score = cl.mstscore(self.data, labels)

        # out["F"] = [score]

        # print('------------------------')
        # print(f'silhouette score = {sil_score}')


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

    def genSolver(self, data, methods):
        self.dr_method = methods[0]
        self.cl_method = methods[1]
        mask = []
        if 'PCA' in methods:
            mask.append('int')

            if 'KMEANS' in methods:
                mask.append('int')

            if 'HDBSCAN' in methods:
                mask.append('int')
                mask.append('int')
                mask.append('real')

            if 'H' not in methods and 'DBSCAN' in methods:
                mask.append('real')
                mask.append('int')

        print(mask)

        sampling, crossover, mutation = self.masker(mask=mask)

        problem = genTuner(data, methods)  #use the corresponding problem

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

    def pcaSolver(self, data):

        mask = ['int']

        sampling, crossover, mutation = self.masker(mask=mask)

        problem = pcaTuner(data)  #use the corresponding problem

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