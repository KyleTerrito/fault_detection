from src.autoKNN import AutoKNN
from src.plotters import Plotters
import pickle

import matplotlib.pyplot as plt
import numpy as np


def test():

    plotter = Plotters()

    plotter.plot_metrics_opt(metric='sil_score')
    plotter.plot_metrics_opt(metric='ch_score')
    plotter.plot_metrics_opt(metric='dbi_score')

    return None
