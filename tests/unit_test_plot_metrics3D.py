#from src.autoKNN import AutoKNN
from src.plotters import Plotters
import pickle

import matplotlib.pyplot as plt
import numpy as np


def test():

    plotter = Plotters(exp='pyro')

    plotter.plot_metrics_opt_3d(metrics=['sil_score', 'ch_score', 'dbi_score'])
    plotter.plot_metrics_opt_3d(metrics=['sil_score', 'ch_score', 'dbi_score'])

    return None