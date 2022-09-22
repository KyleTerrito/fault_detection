from autocluster.autoML import Solvers
import numpy as np
import pandas as pd
from autocluster.faultDetection import DR


def test():

    path = 'data/processed/demethanizer_test_1.xlsx'

    data = np.asarray(pd.read_excel(path, sheet_name='coldata'))

    solver = Solvers()

    res = solver.dbscanSolver(data)

    print('-----------------------------------')
    print(f'Best eps = {(res.X)[0]}')
    print(f'Best min_samples = {(res.X)[1]}')
    print(f'Best silhuette score = {-(res.F)}')
    print('-----------------------------------')

    return None