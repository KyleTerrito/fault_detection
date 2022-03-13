from src.autoKNN import AutoKNN


def test():
    knn = AutoKNN()
    knn.get_data(path='data/processed/TEP_Selected_Faults.xlsx')

    dr_methods = ['NO DR', 'PCA']  # , 'UMAP']
    cl_methods = ['KMEANS', 'DBSCAN']  # , 'HDBSCAN']

    knn.optimize(dr_methods, cl_methods, termination='test')

    knn.show_solutions()
    knn.plot_results()

    return None
