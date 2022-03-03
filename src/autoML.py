"""
-Standalone module

-Provides methods for AutoML via NSGA2 optimization

-Tunable methods for dimensionality reduction, clustering and fault detection 
are imported from src.faultDetection

-A new class is needed for each dr/clustering/fd method to 
define the optimization problem. See pcaTuner() for example.
-Also a new function in Solvers() is needed.

"""
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

from src.faultDetection import DR, Clustering
import sklearn

import warnings
warnings.filterwarnings("ignore")
"""-------------------------------------------------------------------------------"""


class MyDisplay(Display):
    def _do(self, problem, evaluator, algorithm):
        super()._do(problem, evaluator, algorithm)
        #self.output.append("metric_a", -1 * (algorithm.pop.get("F")))
        self.output.append("metric_a", min(algorithm.pop.get("F")))


"""------------------Optimization problems----------------------------------------"""
#TODO: add hdbscanTuner, tsneTuner

#TODO: add multiobjective optimization for dr+clustering+fault_detection problems


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
        model, dr_data = dr.performPCA(self.data, x[0])

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

        # print('------------------------')
        # print(f'Number of labels = {len(set(labels))}')

        # if len(set(labels)) > 2:
        #     sil_score = cl.silmetric(self.data, labels)
        # else:
        #     sil_score = -1

        # out["F"] = [-sil_score]

        score = cl.mstscore(self.data, labels)

        out["F"] = [score]

        # print('------------------------')
        # print(f'silhouette score = {sil_score}')


"""---------------------Solvers----------------------------------------------------"""
#TODO: add hdbscanSolver, tsneSolver


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