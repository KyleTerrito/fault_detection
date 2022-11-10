from autocluster.autoKNN import AutoKNN
from autocluster.plotters import Plotters

def test():
    knn = AutoKNN()

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
                        termination='run', 
                        data_path=r'C:\\Users\\lbrice1\\Documents\\GitHub Repos\\fault_detection\\data\\processed\\Pyrolysis Reduced 4-15_1500.xlsx', 
                        exp='aiche_pyro')

    print(res_dict)
    knn.show_solutions()


    return None

