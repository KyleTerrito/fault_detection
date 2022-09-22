from autocluster.autoKNN import AutoKNN
import pickle
from filelock import FileLock


def test():
    knn = AutoKNN()
    knn.get_data(path='data/processed/TEP0-2_labeled.xlsx', exp='test3d')

    dr_methods = ['NO DR', 'PCA', 'UMAP']
    cl_methods = ['KMEANS', 'DBSCAN', 'HDBSCAN']

    knn.optimize(dr_methods, cl_methods, termination='run')

    knn.show_solutions()

    return None
