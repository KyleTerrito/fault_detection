from autocluster.autoKNN import AutoKNN


def test():
    knn = AutoKNN()
    knn.get_data(path='data/processed/Pyrolysis Reduced 4-15_1500.xlsx',
                 exp='pyro')

    dr_methods = ['NO DR', 'PCA', 'UMAP']
    cl_methods = ['KMEANS', 'DBSCAN', 'HDBSCAN']

    knn.optimize(dr_methods, cl_methods, termination='run')

    knn.show_solutions()

    return None
