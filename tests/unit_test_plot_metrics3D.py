#from src.autoKNN import AutoKNN
from autocluster.plotters import Plotters
import pickle

import matplotlib.pyplot as plt
import numpy as np


def test():
    p_methods = [
                    'Z',
                    'mean',
                    'min_max'
                ]
    
    dr_methods = [
                    'NO DR', 
                    'PCA', 
                    'UMAP'
                ]

    cl_methods = [
                    'KMEANS', 
                    'DBSCAN', 
                    'HDBSCAN'
                ]

    plotter = Plotters(exp='aiche_pyro')

    plotter.plot_metrics_opt_3d(
                            metrics=['sil_score', 'ch_score', 'dbi_score'],
                            p_methods=p_methods, 
                            dr_methods=dr_methods,
                            cl_methods=cl_methods
                            )

    plotter.plot_metrics_opt_3d(
                            metrics=['sil_score', 'ch_score', 'dbi_score'],
                            p_methods=p_methods, 
                            dr_methods=dr_methods,
                            cl_methods=cl_methods
                            )
    return None
