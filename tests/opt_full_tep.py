from autocluster.autoKNN import AutoKNN


def test():
    knn = AutoKNN()
    knn.get_data(path='data/processed/Full_TEP.xlsx', exp='fulltep')

    dr_methods = ['NO DR', 'PCA', 'UMAP']
    cl_methods = ['KMEANS', 'DBSCAN', 'HDBSCAN']

    knn.optimize(dr_methods, cl_methods, termination='run')

    knn.show_solutions()

    return None
