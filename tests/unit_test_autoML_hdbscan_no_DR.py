from autocluster.autoML import Solvers
import numpy as np
import pandas as pd
from autocluster.faultDetection import DR


def test():

    path = 'data/processed/demethanizer_test_1.xlsx'

    data = np.asarray(pd.read_excel(path, sheet_name='coldata'))

    solver = Solvers()

    res = solver.hdbscanSolver(data)

    print('-----------------------------------')
    print(res.X)
    print(f'Best min_cluster_size = {np.asarray(res.X)[:, 0]}')
    print(f'Best min_samples = {np.asarray(res.X)[:, 1]}')
    print(f'Best cluster_selection_epsilon = {np.asarray(res.X)[:, 2]}')
    print(f'Best sil_score = {np.asarray(-res.F)[:, 0]}')
    print('-----------------------------------')

    return None