from src.autoML import Solvers
import numpy as np
import pandas as pd


def test():

    data = np.asarray(
        pd.read_excel(
            'C:/Users/lbrice1/Dropbox/LSU/PSE@LSU/In-house Software/FastMan_Program/datasheet demethanizer_1.xlsx',
            sheet_name='coldata'))

    solver = Solvers()

    res = solver.solverUMAP(data)

    return None