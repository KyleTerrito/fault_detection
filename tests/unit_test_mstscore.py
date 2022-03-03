from src.faultDetection import Clustering
from src.autoML import Solvers
import pandas as pd
import numpy as np


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
    print(f'Best silhuette score = {np.asarray(-res.F)[:, 0]}')
    print('-----------------------------------')

    cl = Clustering()

    labels = cl.performHDBSCAN(data, np.asarray(res.X)[0, :])

    score = cl.mstscore(data, labels)

    print('-----------------------------------')
    print(f'HDBSCAN score = {score}')
    print('-----------------------------------')

    score = cl.mstscore(data, labels=None)
    print('-----------------------------------')
    print(f'No clustering score = {score}')
    print('-----------------------------------')

    return None