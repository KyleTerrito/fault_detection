import sys
import os

os.chdir('../')
os.getcwd()

from autocluster.autoKNN import AutoKNN


def test():
    knn = AutoKNN()
    knn.get_data(path=r'C:\\Users\\lbrice1\\Documents\\GitHub Repos\\fault_detection\\data\\processed\\Full_TEP.xlsx', exp='fulltep')

    dr_methods = [
                    'NO DR', 
                    #'PCA', 
                    #'UMAP'
                ]

    cl_methods = [
                    'KMEANS', 
                    #'DBSCAN', 
                    #'HDBSCAN'
                ]

    res_dict = knn.optimize(dr_methods, cl_methods, termination='test')

    print(res_dict)
    #knn.show_solutions()

    return None

test()