from src.autoML import Solvers
import numpy as np
import pandas as pd
from src.faultDetection import DR


def test():

    path = 'data/processed/demethanizer_test_1.xlsx'

    data = np.asarray(pd.read_excel(path, sheet_name='coldata'))

    methods = ['PCA', 'HDBSCAN']

    solver = Solvers()

    res = solver.genSolver(data, methods)

    print('-----------------------------------')
    print(res.X)
    print(res.F)
    print('-----------------------------------')

    return None
