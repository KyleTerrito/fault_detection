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
                        termination='run', 
                        data_path=r'C:\\Users\\lbrice1\\Documents\\GitHub Repos\\fault_detection\\data\\processed\\TEP_Selected_Faults.xlsx', 
                        exp='aiche_tep')

    print(res_dict)
    knn.show_solutions()

    return None