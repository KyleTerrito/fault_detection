"""
Standalone module

Provides methods for AutoML via NSGA2 optimization

Contains tunable methods for Dimensionality reduction, clustering and fault detection

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


class MyDisplay(Display):
    def _do(self, problem, evaluator, algorithm):
        super()._do(problem, evaluator, algorithm)
        #self.output.append("metric_a", -1 * (algorithm.pop.get("F")))
        self.output.append("metric_a", -1 * min(algorithm.pop.get("F")))


class pcaTuner(ElementwiseProblem):
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
        dr = DR()

        model, dr_data = dr.performPCA(self.data, x[0])

        rc_data = dr.reconstructPCA(model, dr_data)

        mse = sklearn.metrics.mean_squared_error(self.data, rc_data)

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

        print('------------------------')
        print(f'mse = {mse}')
        print('------------------------')

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


class Solvers(ElementwiseProblem):
    '''
    Standalone class

    Contains methods to auto tune fault detection methods

    '''
    def __init__(self):
        super(ElementwiseProblem, self).__init__()

    def solverPCA(self, data):

        mask = ['int']

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

        problem = pcaTuner(data)

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

    def solverUMAP(self, data):

        mask = ['int', 'int', 'real']

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

    def solverDBSCAN(self, data):

        mask = ['real', 'int']

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