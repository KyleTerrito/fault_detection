from src.autoKNN import AutoKNN
import pickle
from filelock import FileLock


def test():
    knn = AutoKNN()
    knn.get_data(path='data/processed/TEP_Selected_Faults.xlsx')

    dr_methods = ['NO DR', 'PCA']  #, 'UMAP']
    cl_methods = ['KMEANS', 'DBSCAN']  #, 'HDBSCAN']

    knn.optimize(dr_methods, cl_methods, termination='test')

    knn.show_solutions()

    # hyperparameters_list = pickle.load(
    #     open('tests/ensemble_test_results/h_list.pkl', 'rb'))

    # # with FileLock('tests/ensemble_test_results/res_dict.pkl'):
    # res_dict = pickle.load(
    #     open('tests/ensemble_test_results/res_dict.pkl', 'rb'))

    # knn.plot_performance(res_dict, hyperparameters_list)

    # knn.plot_results()

    return None
