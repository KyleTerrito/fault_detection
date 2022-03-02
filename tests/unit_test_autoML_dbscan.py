from src.autoML import Solvers
import numpy as np
import pandas as pd
from src.faultDetection import DR


def test():

    data = np.asarray(
        pd.read_excel(
            'C:/Users/lbrice1/Dropbox/LSU/PSE@LSU/In-house Software/FastMan_Program/datasheet demethanizer_1.xlsx',
            sheet_name='coldata'))

    # dr = DR()

    # pca_model, dr_data = dr.performPCA(data, 3)

    solver = Solvers()

    res = solver.solverDBSCAN(data)

    print('-----------------------------------')
    print(f'Best eps = {(res.X)[0]}')
    print(f'Best min_samples = {(res.X)[1]}')
    print(f'Best silhuette score = {-(res.F)}')
    print('-----------------------------------')

    return None