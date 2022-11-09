import sys
import os

#os.chdir('../../')
#print(f'running from: {os.getcwd()}')

from autocluster.autoKNN import AutoKNN
from autocluster.plotters import Plotters


def test():
    knn = AutoKNN()
    #knn.get_data(path=r'C:\\Users\\lbrice1\\Documents\\GitHub Repos\\fault_detection\\data\\processed\\Full_TEP.xlsx', exp='fulltep')

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

    res_dict, hyperparameters_list = knn.optimize(
                        p_methods=p_methods,
                        dr_methods=dr_methods, 
                        cl_methods=cl_methods, 
                        termination='test', 
                        data_path=r'C:\\Users\\lbrice1\\Documents\\GitHub Repos\\fault_detection\\data\\processed\\Full_TEP.xlsx', 
                        exp='test')

    print(res_dict)
    knn.show_solutions()

    plotter = Plotters(exp='test')
    plotter.plot_pareto(res_dict, hyperparameters_list)

    plotter.plot_metrics_opt_3d(
                            metrics=['sil_score', 'ch_score', 'dbi_score'],
                            p_methods=p_methods, 
                            dr_methods=dr_methods,
                            cl_methods=cl_methods
                            )


    # knn.plot_results()

    return None
